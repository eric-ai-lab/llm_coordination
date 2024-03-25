import time
import datetime 
import re 
import openai
from openai import OpenAI, AzureOpenAI

import os 

#openai.api_key = os.environ['API_KEY']
#openai.organization = os.environ['ORGANIZATION']

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LLMAgent():
    def __init__(self, player_id, temperature=0.6, do_sample=True, max_new_tokens=1000, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, api_server=True,  cache_dir=os.getenv('HF_HOME')):
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_server = api_server
        self.device = 'cuda'

        # self.model = 'gpt-4-0125'
        # self.model_name = 'gpt-4-0125'
        # self.model = 'gpt-35-turbo'
        # self.model_name = 'gpt-35-turbo'
        # self.model_type = 'openai'
        self.model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        self.model_type = 'mistral'
        self.model = 'mixtral'

        self.player_id = player_id
        self.player_names = ['Alice', 'Bob']
        self.player_name = self.player_names[player_id]
        self.partner_name = self.player_names[1 - player_id]
        self.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.killer_info = ''
        
        self.base_prompt = '''In the game Collab Escape. We must cooperate with each other to repair generators and escape the map. We win even if only one of us escapes the map.

        Environment Details: There are 7 rooms (room 1 to 7). Room 1 and 2 have generators, and room 7 has the exit gate. The rooms are connected by pathways, and you can only move to adjacent rooms. Room 1 is connected to room 2, 5, and 7; room 2 is connected to room 1, 6, and 3; room 3 is connected to room 2, 4, 5, and 7; room 4 is connected to room 3 and 5; room 5 is connected to room 1, 3, 4, and 6; room 6 is connected to room 2, 5, and 7; room 7 is connected to room 1, 3, and 6.

        As a survivor, my goal is to avoid the killer and move to rooms with generators, repair them, and reach the exit gate to escape. To fix a generator, it needs two consecutive repair actions. Survivors must also avoid being in the same room as the killer or an adjacent room, as the killer will move to catch any survivors they see in adjacent rooms. Being in the same room with the Killer results in an immediate loss.

        In each turn, survivors and the killer simultaneously perform one action. I cannot know each other's actions ahead of time but can make educated guesses. To win the game I will need to protect teammate, fix generators, divert the killer, etc. Help me select the best action from the list, considering my priority. First, consider if the current room is safe from the killer and then format your response as "Action: move to room <number>" or "Action: fix generator in room <number>." Do not say anything else.'''

        self.llm_system_prompt = "A chat between a human and an assistant. The assistant is correct and brief at all times."

        self.assistant_response_initial = f'''Got it!'''

        self.action_regex = r"Action:\s*(.*)"

        if self.model_type == 'openai':
            self.message = [
                        {"role": "system", "content": self.llm_system_prompt},
                        {"role": "user", "content": self.base_prompt},
                        {"role": "assistant", "content": self.assistant_response_initial},
                    ]
        else:
            self.message = [
                        {"role": "user", "content": self.base_prompt},
                        {"role": "assistant", "content": self.assistant_response_initial},
                    ]
        
        if self.model_type == 'openai':
            self.akey = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.org = os.getenv("AZURE_OPENAI_API_KEY")
            # self.client = OpenAI(api_key = self.akey, organization = self.org)
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version="2023-05-15"
            )
        else:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8006/v1"
            )
        self.inference_fn = self.run_openai_inference
        self.num_api_calls = 0
        self.all_actions = [f'move to {r}' for r in ["room 1", "room 2", "room 3", "room 4", "room 5", "room 6", "room 7"]]
        self.all_actions += ['fix generator in room 1', 'fix generator in room 2']
        self.all_actions += ['wait']

    def run_openai_inference(self, messages):
        api_call_start = time.time()
        completion = self.client.chat.completions.create(
            messages = messages,
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )
        print(f"{bcolors.FAIL}LLM INFERENCE TIME: {time.time() - api_call_start}{bcolors.ENDC}")
        # print("INFERENCE STRING: ", completion.choices[0].message.content)
        self.num_api_calls += 1
        print(f"{bcolors.OKBLUE}Number of API calls made by {self.player_name}: {bcolors.ENDC}", self.num_api_calls)
        #total_tokens = completion.usage.total_tokens
        #cur_cost =  0.011 * (total_tokens / 1000)
        #self.cost += cur_cost 
        #print(f"COST SO FAR: {self.cost} USD")
        return completion.choices[0].message.content
        
    def _get_available_actions(self, state):
        
        available_actions = []

        current_room = state[self.player_name].current_room
        
        for room in current_room.adjacent_rooms:
            available_actions.append(f'move to {room.name}')
        
        if current_room.name in list(state['Generators'].keys()):
            available_actions.append(f"fix generator in {current_room.name}")
        
        return available_actions + ['wait']
        
    def _state_to_description(self, state, killer_info):
        print(state)
        player_location = state[self.player_name].current_room.name
        partner_location = state[self.partner_name].current_room.name
        state_description = f"My name is {self.player_name}. I am in {player_location}. "
        if self.player_name == 'Alice' and state['Alice'].last_action_is_fixing:  
            if state['Alice'].current_room.generator_fixed:
                state_description += 'I have finished to fix the generator in this room. '
            else:
                state_description += 'I have started to fix the generator in this room. '
        if self.player_name == 'Bob' and state['Bob'].last_action_is_fixing:  
            if state['Bob'].current_room.generator_fixed:
                state_description += 'I have finished to fix the generator in this room. '
            else:
                state_description += 'I have started to fix the generator in this room. '            
        state_description+= f"{self.partner_name} is in {partner_location}. "
        if self.partner_name == 'Alice' and state['Alice'].last_action_is_fixing:  
            state_description += 'Alice in last turn was fixing the generator. '
        if self.partner_name == 'Bob' and state['Bob'].last_action_is_fixing:  
            state_description += 'Bob in last turn was fixing the generator. '
            
        

        state_description += killer_info
        
        
        # adversary_location = state['Adversary'].current_room.name
        # state_description += f"The Killer is in {adversary_location} that connects to"
        
        # for room in state['Adversary'].current_room.adjacent_rooms:
        #     state_description += ' ' + room.name
        # state_description += '. '
        
        generator_status = state['Generators']        
        
        generator_rooms = list(generator_status.keys())
        for room_name in generator_rooms:
            if 2 - generator_status[room_name]['fix_count'] == 0:
                state_description += f"Generator in {room_name} is fixed. "
            else:
                state_description += f"Generator in {room_name} still needs {2 - generator_status[room_name]['fix_count']} fix. "

        
       

        exit_gate_status = state['exit gate']
        if exit_gate_status:
            state_description += f"The exit gate is open for escape. "
        else:
            state_description += f"The exit gate is closed. "

        self.available_actions_list = self._get_available_actions(state)
        available_actions_string = ', '.join(self.available_actions_list)
        state_description += f'Available Actions: {available_actions_string}'

        return state_description
    
    
    def get_next_move(self, state, killer_info):
        state_description = self._state_to_description(state, killer_info)
        response = ''
        print(f"{bcolors.FAIL}{state_description}{bcolors.ENDC}")
        # Running inference here
        try:
            messages = self.message + [{"role": "user", "content": state_description}]
            response = self.inference_fn(messages=messages)
            print(f'''{bcolors.WARNING}LLM RESPONSE: {response}{bcolors.ENDC}''')
            match = re.search(self.action_regex, response.strip())
            if match:
                action = match.group(1).strip().replace('.', '').lower()
            else:
                action = 'wait'
        except Exception as e:
            action = 'wait' 
            print(f'Failed to get response from openai api for player {self.player_id} due to {e}')
        print(self.all_actions)
        
        if action in self.all_actions:
            if action in self.available_actions_list:
                selected_action = action 
            else:
                selected_action = 'wait'
        else:
            selected_action = 'wait'

        return selected_action

