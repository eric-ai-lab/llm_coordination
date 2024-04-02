import time 
import os 
from openai import OpenAI, AzureOpenAI
import datetime 
import re 
from fuzzywuzzy import process 
import numpy as np 

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

class LLMManager:
    def __init__(self, model_name, model_type, cache_dir, temperature=0.6, do_sample=True, max_new_tokens=1000, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, api_server=True):
        self.model_name = model_name 
        self.model_type = model_type
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_server = api_server
        self.device = 'cuda'
        
        self.cost = 0
        if self.model_type == 'openai':
            self.akey = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.org = os.getenv("AZURE_OPENAI_API_KEY")
            # self.client = OpenAI(api_key = self.akey, organization = self.org)
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version="2023-05-15"
            )
            self.inference_fn = self.run_openai_inference
        else:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8000/v1"
            )
            self.inference_fn = self.run_openai_inference                    

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
        total_tokens = completion.usage.total_tokens
        cur_cost =  0.015 * (total_tokens / 1000)
        self.cost += cur_cost 
        print(f"COST SO FAR: {self.cost} USD")
        return completion.choices[0].message.content


class LLMAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.player_names = ['Alice', 'Bob']
        

        # Controls 
        self.DEBUG = False     
        self.enable_cache = False # True    
        self.write_to_cache = False 
        self.save_trajectory = True # True 

        # self.model = 'gpt-4-0125'
        # self.model_name = 'gpt-4-0125'
        # self.model = 'gpt-35-turbo'
        # self.model_name = 'gpt-35-turbo'
        # self.model_type = 'openai'
        self.model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        self.model_type = 'mistral'
        self.model = 'mixtral'

        self.llm = LLMManager(model_name=self.model_name, model_type=self.model_type, cache_dir=os.getenv('HF_HOME'))

        self.experiment_type = 'ai'
        
        ### LOGGING ###
        self.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f'/home/saaket/llm_coordination/logs/hanabi'
        self.traj_dir = f'/home/saaket/llm_coordination/logs/hanabi'
        ####
        ### LOGGING ###
        
        self.num_api_calls = 0 
        self.player_actions = []

        # Set other player ID 
        if int(self.player_id) == 0:
            self.other_player_id = 1
        else:
            self.other_player_id = 0

        self.player_name = self.player_names[self.player_id]
        self.other_player_name = self.player_names[self.other_player_id]

        self.llm_system_prompt = "You are a friendly chat assistant who is correct and brief at all times."
        
        self.rules = f'''
        1. A thief is caught if a player is in the same room as the thief or if the thief and a player in connected rooms move towards each other. 
        2. The environment is fully visible to all three entities. In each turn, player and the thief simultaneously perform one action.
        3. The thief always takes the greediest action to move away from the player closest to him. 
        4. Doors can be either open or closed. 
        5. If the door is closed, no one can pass, creating a dead end.
        6. Pressing the button closes an open door and opens a closed door. 
        7. A button in a room can only be pressed if player is in that room
        8. All rooms are connected by pathways with same length, and everyone can only move to connected rooms in each turn.'''

        # Without COT
        self.base_prompt = f'''I {self.player_name} am playing the game """"Collab Capture"""" with Bob. I want to coordinate with Bob to catch the thief in the environment in the minimum number of steps. Coordination between Bob and Alice is important to trap the thief. 
        Environment Layout:
        - Room 1 directly connects to Room 2 and Room 6.
        - Room 2 directly connects to Room 1, Room 3.
        - Room 3 directly connects to Room 2, Room 4,
        - Room 4 directly connects to Room 3, Room 5.
        - Room 5 directly connects to Room 4, Room 6, Room 7.
        - Room 6 directly connects Room 5, Room 9, Room 1.
        - Room 7 directly connects Room 5, Room 8.
        - Room 8 directly connects to Room 7.
        - Room 9 directly connects Room 6
        There is a door between Room 1 and Room 2 controlled by a button in Room 9. There is a door between Room 3 and Room 4 controlled by a button in Room 8. Help me {self.player_name} select my next action. Format your response as: 
        Analysis: <brief explanation for your next action>.
        Action: <your selected action from the list>.'''

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
            
        self.tokens_used = 0 

        # print(self.all_actions)
        self.log_csv_dict = {}
        self.action_history = []
    
    def _get_available_actions(self, state_for_llm):
        self.available_actions_list = []
        description = ''
        for i, action in enumerate(state_for_llm['available_actions'][self.player_name]):
            
            if str(action).isdigit():
                self.available_actions_list.append(f'{chr(65+i)}. Move to room {action}.')
            else:
                self.available_actions_list.append(f'{chr(65+i)} {action}')
        
        for action in self.available_actions_list:
            description += f'{action}\n'
        return description

    def find_best_match(self, action_string):
        match = re.search(self.action_regex, action_string.strip())
        if match:
            selected_match = match.group(1).strip().lower()

            # Sometimes model repeats Action: withing the action string
            if 'action:' in selected_match.lower():
                updated_action_regex = r"action:\s*(.*)"
                match = re.search(updated_action_regex, selected_match.strip())
                if match:
                    selected_match = match.group(1).strip().lower()
            ####
            for action in self.available_actions_list:
                if selected_match.lower() in action.lower():
                    return action 
            selected_move, score = process.extractOne(selected_match, self.available_actions_list)
        else:
            selected_move = np.random.choice(self.available_actions_list)
        return selected_move

    
    def _state_to_description(self, state_for_llm):
        
        # Where is everyone
        description = f"I {self.player_name} am in room {state_for_llm['player_locs'][self.player_name]}. {self.other_player_name} is in room {state_for_llm['player_locs'][self.other_player_name]}. Thief is in room {state_for_llm['player_locs']['Thief']}. "

        # Which doors are open, which are closed 
        for door in state_for_llm['door_states']:
            description += f"Door between room {door[0]} and room {door[1]} is {state_for_llm['door_states'][door]}. "
        
        description += f'\n{self._get_available_actions(state_for_llm)}'

        return description
    
    def get_next_move(self, state_for_llm):
        
        state_description = self._state_to_description(state_for_llm)
        print(f"{bcolors.OKBLUE}{state_description}{bcolors.ENDC}")
        messages = self.message + [{'role': 'user', 'content': state_description}]

        action_string = self.llm.inference_fn(messages=messages)
        print(f"{bcolors.OKGREEN}LLM Response: {action_string}{bcolors.ENDC}")
        selected_action = self.find_best_match(action_string)

        if selected_action in self.available_actions_list:
            selected_move_idx = self.available_actions_list.index(selected_action)
        print("SELECTED ACTION: ", state_for_llm['available_actions'][self.player_name][selected_move_idx])
        return state_for_llm['available_actions'][self.player_name][selected_move_idx]









