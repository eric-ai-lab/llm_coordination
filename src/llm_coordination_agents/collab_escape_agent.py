import datetime 
import re 
import openai 

import os 
openai.api_key = os.environ['API_KEY']
openai.organization = os.environ['ORGANIZATION']

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
    def __init__(self, player_id):
        self.player_id = player_id
        self.player_names = ['Alice', 'Bob']
        self.player_name = self.player_names[player_id]
        self.partner_name = self.player_names[1 - player_id]
        self.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.killer_info = ''
        
        self.base_prompt = '''In the game Collab Escape. We must cooperate with each other to repair generators and escape the map. We win even if one of us escapes the map.

        Environment Details: There are 7 rooms (room 1 to 7). Room 1 and 2 have generators, and room 7 has the exit gate. The rooms are connected by pathways, and you can only move to adjacent rooms. Room 1 is connected to room 2, 5, and 7; room 2 is connected to room 1, 6, and 3; room 3 is connected to room 2, 4, 5, and 7; room 4 is connected to room 3 and 5; room 5 is connected to room 1, 3, 4, and 6; room 6 is connected to room 2, 5, and 7; room 7 is connected to room 1, 3, and 6.

        As a survivor, my goal is to avoid the killer and move to rooms with generators, repair them, and reach the exit gate to escape. To fix a generator, it needs two consecutive repair actions. Survivors must also avoid being in the same room as the killer or an adjacent room, as the killer will move to catch any survivors they see in adjacent rooms. Being in the same room with the Killer results in an immediate loss.

        In each turn, survivors and the killer simultaneously perform one action. I cannot know each other's actions ahead of time but can make educated guesses. To win the game I will need to protect teammate, fix generators, divert the killer, etc. Help me select the best action from the list, considering my priority. First, consider if the current room is safe from the killer and then format your response as "Action: move to room <number>" or "Action: fix generator in room <number>." Do not say anything else.'''

        self.llm_system_prompt = "A chat between a human and an assistant. The assistant is correct and brief at all times."

        self.assistant_response_initial = f'''Got it!'''

        self.action_regex = r"Action:\s*(.*)"

        self.message = [
                    {"role": "system", "content": self.llm_system_prompt},
                    {"role": "user", "content": self.base_prompt},
                    {"role": "assistant", "content": self.assistant_response_initial},
                ]
        self.model = 'gpt-4'
        # self.model = 'gpt-3.5-turbo'
        self.num_api_calls = 0
        self.all_actions = [f'move to {r}' for r in ["room 1", "room 2", "room 3", "room 4", "room 5", "room 6", "room 7"]]
        self.all_actions += ['fix generator in room 1', 'fix generator in room 2']
        self.all_actions += ['wait']
        
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
        response_string = ''
        message = ''
        print(f"{bcolors.FAIL}{state_description}{bcolors.ENDC}")
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.message + [{"role": "user", "content": state_description}],
                temperature=0.6
            )
            
            self.num_api_calls += 1
            print(f"{bcolors.OKBLUE}Number of API calls made by {self.player_name}: {bcolors.ENDC}", self.num_api_calls)
            response_string = response["choices"][0]["message"]["content"]
            print(f'''{bcolors.WARNING}LLM RESPONSE: {response_string}{bcolors.ENDC}''')

            match = re.search(self.action_regex, response_string.strip())
            if match:
                action = match.group(1).strip().replace('.', '').lower()
            else:
                action = 'wait'
        except:
            selected_action = 'wait' 
            print(f'Failed to get response from openai api for player {self.player_id} due to {e}')
        print(self.all_actions)
        
        if action in self.all_actions:
            if action in self.available_actions_list:
                selected_action = action 
            else:
                selected_action = 'wait'

        return selected_action
