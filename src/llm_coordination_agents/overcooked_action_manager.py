from collections import deque
import numpy as np
import time
import math 
import random 
import datetime
from overcooked_ai_py.mdp.actions import Action, Direction, LLMActionSet
from llm_coordination_agents.overcooked_agent import LLMAgent
import re

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

set_global_seed(42)

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


def extract_location(s):
        pattern = r'([a-z])(\d+)'
        match = re.search(pattern, s)

        # Print the match
        if match:
            return match.group(1), match.group(2)
        else:
            return None, None  
        
NO_COUNTERS_PARAMS = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}

NO_COUNTERS_START_OR_PARAMS = {
    "start_orientations": True,
    "wait_allowed": False,
    "counter_goals": [],
    "counter_drop": [],
    "counter_pickup": [],
    "same_motion_goals": True,
}

storage_counter_locations = {
    'forced_coordination': [(2, 1), (2, 2), (2, 3)],
    'counter_circuit_o_1order': [(2, 2), (3, 2), (4, 2), (5,2)],
    'cramped_room': [],
    'coordination_ring': [],
    'asymmetric_advantages': [],
    'bottleneck': [],
    'centre_pots': [],
    'centre_objects': [],
    'large_room': [],
    'no_counter_door': [],
    'soup_passing_door': [],
    'soup_passing': [(3, 1), (3, 2), (3, 3)],
}


class LLMActionManager(object):
    def __init__(self, mdp, player_name, layout_name): 
        self.layout_name = layout_name
        self.mdp = mdp 
        self.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # self.onion_dispenser_locations = self.mdp.get_onion_dispenser_locations()
        self.object_location = {
            'o': self.mdp.get_onion_dispenser_locations(), 
            'c': self.mdp.get_pot_locations(),
            'p': self.mdp.get_dish_dispenser_locations(),
            'd': self.mdp.get_serving_locations(),
            's':  storage_counter_locations[self.layout_name],
            'k': self.mdp.get_counter_locations(),
            'g': self.mdp.get_gate_locations(),
        }
        # Shared counter should not be counted twice 
        self.object_location['k']  = [k for k in self.object_location['k'] if k not in self.object_location['s']]
        self.trajectory = []
        self.selected_action = None 
        self.pot_state = 0 
        
        self.other_player_position = None 
        self.soup_delivered = False 
        self.player_held_object = {
            'onion': False, 
            'plate': False,
            'soup': False,
        }
        self.pot_state_to_num_onions = {
            'empty': 0,
            '1_items': 1,
            '2_items': 2, 
            '3_items': 3,
            'cooking': 3, 
            'ready': 3,
        }
        self.add_extra_wait_state = False 
        self.player_name = player_name 
        self.player_id = int(self.player_name[-1])
        self.other_player_id = 1 if self.player_id == 0 else 0 
        self.should_get_stage_from_llm = True 
        self.current_stage = None 
        self.moving_away = False 
        self.state_for_llm = {
            self.player_id: {'held_object': 'nothing'},
            self.other_player_id: {'held_object': 'nothing'},
            'distances': {
                'onion_dispenser': [[None, None] for _ in range(len(self.mdp.get_onion_dispenser_locations()))],
                'cooker': [[None, None] for _ in range(len(self.mdp.get_pot_locations()))],
                'plate_dispenser': [[None, None] for _ in range(len(self.mdp.get_dish_dispenser_locations()))],
                'delivery_zone': [[None, None] for _ in range(len(self.mdp.get_serving_locations()))],
                'storage_counter': [[None, None] for _ in range(len(self.object_location['s']))],
                'kitchen_counter': [[None, None] for _ in range(len(self.object_location['k']))],
                'gate': [[None, None] for _ in range(len(self.mdp.get_gate_locations()))]
            },
            'storage_counter_objects': ['empty']*len(self.object_location['s']),
            'kitchen_counter_objects': ['empty']*len(self.object_location['k']),
            'num_onions_in_pot': ['0']*len(self.object_location['c']),
            'cooker_status': ['off']*len(self.object_location['c']),
            'soup_in_cooker_status': ['not cooked']*len(self.object_location['c']),
            'gate_status': ['closed', 'closed'],
            'gate_open_time': [0, 0],
            'soup_in_cooker_remaining_time': [0, 0]
        }
        self.action_set = LLMActionSet[self.layout_name]
        self.message = ''
        self.llm_agent = LLMAgent(self.player_id, self.layout_name) 
        self.save_low_level_trajectory = True
        self.prev_directive = 'wait.'

    

    def get_next_move(self, state, other_player_message):
        print(f"GET NEXT MOVE PLAYER {self.player_id}")
        # Handle player and partner variables 
        self.set_player_pos_or(state) 
        self.message = ''

        # Default Action
        self.selected_action = Action.STAY

        # If player sends a message, current stage may be changed midway. 
        if other_player_message!= '':
            self.should_get_stage_from_llm = True 

        # if previous stage is complete, get new stage from llm 
        if self.should_get_stage_from_llm:
            self.get_stage_from_llm(state, other_player_message)         

        if self.current_stage == 'wait.':
            self.selected_action = Action.STAY 
            self.should_get_stage_from_llm = True 
        
        elif self.current_stage == 'move away.':
            self.selected_action = self.move_away_deterministic()
            self.should_get_stage_from_llm = True

        else:
            location, location_id = extract_location(self.current_stage)
            assert location in list(self.object_location.keys())
            if location_id == None:
                self.selected_action = Action.STAY
            else:
                self.selected_action = self.move_to(location, int(location_id))
                if self.selected_action == Action.REACHED:
                    if 'pick' in self.current_stage:
                        self.selected_action = self.pick_up(state)
                    elif 'place' in self.current_stage or 'put' in self.current_stage:
                        self.selected_action = self.place(state)
                    elif 'load' in self.current_stage:    
                        self.selected_action = self.load_plate(state, int(location_id))
                    elif 'open' in self.current_stage:
                        self.selected_action = self.open(state)
                    # elif 'deliver' in self.current_stage:
                    #     self.selected_action = self.place(state)
                    else:
                        print(f"{bcolors.FAIL}ERROR: Unknown action type in {self.current_stage}{bcolors.FAIL}")
                        self.selected_action = Action.STAY
        
        self.handle_stalemate()
        self.assign_prev_action_directive()

        self.trajectory.append([self.selected_action, self.message])
        return self.selected_action, self.message
    

    def assign_prev_action_directive(self):
        self.prev_selection_action = self.selected_action
        self.prev_directive = self.current_stage
        if self.moving_away:
            self.prev_directive = 'move away deterministic'
        self.prev_position = self.player_position
        self.prev_other_player_position = self.other_player_position
        self.prev_orienation = self.player_orientation

    def handle_stalemate(self):
        # Handle case where loc_agent_a = goal_agent_b and loc_agent_b = goal_agent_a  AND  gate closed 
        # if self.selected_action == Action.STAY and self.current_stage not in ['wait.', 'load soup on plate from c0.', 'load soup on plate from c1.']:
        #     print(f"Case 1 stalemate for player {self.player_id}. info - {self.selected_action}, {self.current_stage}")
        #     self.should_get_stage_from_llm = True 
        
        # Handle case goal_agent_a = goal_agent_b
        if self.prev_directive != 'wait.' and self.prev_selection_action != Action.INTERACT and self.selected_action != Action.INTERACT and self.current_stage not in ['wait.']:
            if self.player_position == self.prev_position and self.prev_orienation == self.player_orientation and self.should_get_stage_from_llm == False:
                if int(self.player_id) == 1:
                    print(f"Case 2 stalemate. info prev_directive - {self.prev_directive} prev_selected_action - {self.prev_selection_action} - current_stage - {self.current_stage}")
                    if self.prev_position == self.player_position and self.prev_directive == 'move away deterministic':
                        self.selected_action = self.move_away_deterministic(avoid_dirs=self.prev_selection_action)
                    else:
                        self.selected_action = self.move_away_deterministic()
        else:
            self.moving_away = False 


        
    def move_away_random(self):
        directions = [Direction.WEST, Direction.EAST, Direction.NORTH, Direction.SOUTH]
        random.shuffle(directions)
        new_pos_arr = [(self.player_position[0] + d[0], self.player_position[1] + d[1]) for d in directions]
        return self.go_to_position(new_pos_arr, self.player_position, self.player_orientation)
    
    def move_away_deterministic(self, avoid_dirs=None):
        self.moving_away = True 
        all_directions = [Direction.WEST, Direction.NORTH, Direction.SOUTH, Direction.EAST]

        # Find direction of approach
        dx = self.player_position[0] - self.other_player_position[0]
        dy = self.player_position[1] - self.other_player_position[1]
        
        preferred_directions = []
        if dx > 0:
            preferred_directions.append((Direction.EAST))
        elif dx < 0:
            preferred_directions.append((Direction.WEST))
        
        if dy > 0:
            preferred_directions.append((Direction.SOUTH))
        elif dy < 0:
            preferred_directions.append((Direction.NORTH))
        
        remaining_directions = [x for x in all_directions if x not in preferred_directions]
        considered_directions = preferred_directions + remaining_directions
        if avoid_dirs != None and avoid_dirs != (0, 0):
            considered_directions.remove(avoid_dirs)
        new_pos_arr = [(self.player_position[0] + d[0], self.player_position[1] + d[1]) for d in considered_directions]
        return self.go_to_position(new_pos_arr, self.player_position, self.player_orientation)
        # for direction in preferred_directions:
        #     if direction != (0, 0):
        #         new_pos = (self.player_position[0] + direction[0], self.player_position[1] + direction[1])
        #         # print(f"INFO: starting position {self.player_position}, moving away to new position {new_pos}, moving in direction {direction}")
        #         if self.mdp.get_terrain_type_at_pos(new_pos) == " ":
        #             self.should_get_stage_from_llm = True
        #             return self.go_to_position(new_pos)
            
        # remaining_directions = [x for x in all_directions if x not in preferred_directions]

        # for direction in remaining_directions:
        #     new_pos = (self.player_position[0] + direction[0], self.player_position[1] + direction[1])
        #     # print(f"INFO: starting position {self.player_position}, moving away to new position {new_pos}, moving in direction {direction}")
        #     if self.mdp.get_terrain_type_at_pos(new_pos) == " ":
        #         self.should_get_stage_from_llm = True
        #         return self.go_to_position(new_pos)
        
       # If no accessible direction is found, wait in place
        self.should_get_stage_from_llm = True
        # print(f"ERROR: No position found to move away to")
        return Action.STAY 


    def move_away(self):
        # Calculate the direction from the other player to the current player
        dx = self.player_position[0] - self.other_player_position[0]
        dy = self.player_position[1] - self.other_player_position[1]

        # Determine the direction to move away from the other player
        if dx > 0:
            x_dir = 1
        elif dx < 0:
            x_dir = -1
        else:
            x_dir = 0

        if dy > 0:
            y_dir = 1
        elif dy < 0:
            y_dir = -1
        else:
            y_dir = 0

        # Try to move in the opposite direction of the other player
        for direction in [(x_dir, 0), (0, y_dir)]:
            new_pos = (self.player_position[0] + direction[0], self.player_position[1] + direction[1])
            # print(f"INFO: starting position {self.player_position}, moving away to new position {new_pos}, moving in direction {direction}")
            if self.mdp.get_terrain_type_at_pos(new_pos) == " ":
                self.should_get_stage_from_llm = True
                return self.go_to_position(new_pos)

        # If no accessible direction is found, wait in place
        self.should_get_stage_from_llm = True
        # print(f"ERROR: No position found to move away to")
        return Action.STAY
    
    def move_to(self, location, location_id=0):
        # target_locs_arr, pseudo_obj_loc_arr = self.get_nearest_unblocked_object_locations(self.player_position[0], self.player_position[1], obj=location)
        location_coords = self.object_location[location][location_id]
        target_locs_arr = self.get_adjacent_empty_points(location_coords)
        if target_locs_arr == None:
            # Another stalemate handler 
            self.should_get_stage_from_llm = True 
            return Action.STAY
        
        # TODO: Make a decision about this 
        elif len(target_locs_arr) == 0:
            self.should_get_stage_from_llm = True 
            return Action.STAY
        if self.player_position in target_locs_arr:
            idx = target_locs_arr.index(self.player_position)
            desired_orientation = self.calculate_desired_orientation(self.player_position, location_coords)
            if desired_orientation!= self.player_orientation:
                return desired_orientation                    
            else:
                self.selected_action = Action.REACHED
            return self.selected_action 
                
        return self.go_to_position(target_locs_arr, self.player_position, self.player_orientation)
    
    def pick_up(self, state):
        state_dict = state.to_dict()
        # Add assertions to check whether player is actually next to item and facing it 
        if state_dict['players'][int(self.player_id)]['held_object'] == None:
            self.should_get_stage_from_llm = True 
            return Action.INTERACT
        else:
            # print(f"ERROR: Already holding object, cannot pick up {item}")
            return Action.STAY
        
    def load_plate(self, state, location_id):
        state_dict = state.to_dict()
        if state_dict['players'][int(self.player_id)]['held_object']['name'] == 'dish':
            pot_states_dict = self.mdp.get_pot_states(state)
            if 'ready' in pot_states_dict['onion']:
                if self.object_location['c'][location_id] in pot_states_dict['onion']['ready']:
                    self.should_get_stage_from_llm = True 
                    return Action.INTERACT
        self.should_get_stage_from_llm = True 
        return Action.STAY


    def place(self, state):
        state_dict = state.to_dict()
        # Add assertions to check whether player is actually next to location and facing it 
        if state_dict['players'][int(self.player_id)]['held_object'] != None:
            self.should_get_stage_from_llm = True 
            return Action.INTERACT
        else:
            # print(f"ERROR: Not holding {item} cannot complete")
            return Action.STAY
    
    def open(self, state):
        state_dict = state.to_dict()
        if state_dict['players'][int(self.player_id)]['held_object'] == None:
            self.should_get_stage_from_llm = True 
            return Action.INTERACT
        

    def _populate_distances(self):
        for k, v in self.state_for_llm['distances'].items():
            for i in range(len(self.state_for_llm['distances'][k])):

                # Find all distances for me 
                adjc_accessible_points = self.get_adjacent_accessible_points(self.object_location[k[0]][i])
                if len(adjc_accessible_points) <= 0:
                    self.state_for_llm['distances'][k][i][0] = 'infinite'
                else:

                    # dest = self._find_closest_point(self.player_position, self.other_player_position, adjc_accessible_points)
                    # distances = self.find_shortest_distance(self.player_position, adjc_accessible_points, self.other_player_position)
                    dist, dest = self.find_shortest_distance(self.player_position, adjc_accessible_points, self.other_player_position)
                    # for point in adjc_accessible_points:
                    #     s_dist = self.find_shortest_distance(self.player_position, point, self.other_player_position)

                    # if dist == None:
                    #     dist = 'infinite'
                    # elif dest == self.other_player_position:
                    #     dist = 'blocked'
                    self.state_for_llm['distances'][k][i][0] = str(dist)

                    adjc_accessible_points, dest, dist = None, None, None 

                # Find all distances for my partner 
                adjc_accessible_points = self.get_adjacent_accessible_points(self.object_location[k[0]][i])
                if len(adjc_accessible_points) <= 0:
                    self.state_for_llm['distances'][k][i][0] = 'infinite'
                else:
                    # dest = self._find_closest_point(self.other_player_position, self.player_position, adjc_accessible_points)
                    # dist = self.find_shortest_distance(self.other_player_position, dest, self.player_position)
                    dist, dest = self.find_shortest_distance(self.other_player_position, adjc_accessible_points, self.player_position)
                    # if dist == None:
                    #     dist = 'infinite'
                    # elif dest == self.player_position:
                    #     dist = 'blocked'
                    self.state_for_llm['distances'][k][i][1] = str(dist)



    def _find_closest_point(self, p1, other_player_position, p_list):
        # closest_point = None
        # closest_distance = math.inf

        # for p in p_list:
        #     distance = math.sqrt((p[0] - p1[0])**2 + (p[1] - p1[1])**2)
        #     if distance < closest_distance:
        #         closest_distance = distance
        #         closest_point = p

        # return closest_point
        sorted_p_list = sorted(p_list, key=lambda p: math.sqrt((p[0] - p1[0])**2 + (p[1] - p1[1])**2))
        if len(sorted_p_list) > 1:
            for point in sorted_p_list:
                if point != other_player_position:
                    return point 
        if len(sorted_p_list) > 0:
            return sorted_p_list[0]
        return None 


    def is_cooking(self, soup):
        soup_type, num_items, cook_time = soup.state 
        return 0 <= cook_time < self.mdp.soup_cooking_time and num_items >= self.mdp.num_items_for_soup

    def is_ready(self, soup):
        soup_type, num_items, cook_time = soup.state 
        return num_items >= self.mdp.num_items_for_soup and cook_time >= self.mdp.soup_cooking_time


    #     @property
    #     def cook_time_remaining(self):
    #         return max(0, self.cook_time - self._cooking_tick)

    #     @property
    #     def is_ready(self):
    #         if self.is_idle:
    #             return False
    #         return self._cooking_tick >= self.cook_time

    

    def _populate_pot_states(self, state):
        for idx, pot_pos in enumerate(self.object_location['c']):
            if not state.has_object(pot_pos):
                # print(f"{bcolors.OKGREEN}SOUP IS NOT COOKING in c{idx} and there are 0 onions{bcolors.ENDC}")
                self.state_for_llm['num_onions_in_pot'][idx] = 0 
                self.state_for_llm['soup_in_cooker_status'][idx] = 'not cooking'
                self.state_for_llm['cooker_status'][idx] = 'off'
            else:
                soup = state.get_object(pot_pos)
                if self.is_ready(soup):
                    # print(f"{bcolors.OKGREEN}SOUP IS READY in c{idx} {bcolors.ENDC}")
                    self.state_for_llm['soup_in_cooker_status'][idx] = 'cooked'
                    self.state_for_llm['cooker_status'][idx] = 'off'
                    self.state_for_llm['num_onions_in_pot'][idx] = 3
                elif self.is_cooking(soup):
                    # print(f"{bcolors.OKGREEN}SOUP IS COOKING in c{idx} {bcolors.ENDC}")
                    self.state_for_llm['soup_in_cooker_status'][idx] = 'still cooking'
                    self.state_for_llm['cooker_status'][idx] = 'on'
                    self.state_for_llm['num_onions_in_pot'][idx] = 3
                    _, _, cook_time = soup.state 
                    self.state_for_llm['soup_in_cooker_remaining_time'][idx] = cook_time
                else:
                    # print(f"{bcolors.OKGREEN}SOUP IS NOT COOKING in c{idx} and there are {self.state_for_llm['num_onions_in_pot'][idx]} onions {bcolors.ENDC}")
                    _, num_ingredients, _ = soup.state 
                    self.state_for_llm['soup_in_cooker_status'][idx] = 'not cooking'
                    self.state_for_llm['cooker_status'][idx] = 'off'
                    self.state_for_llm['num_onions_in_pot'][idx] = num_ingredients

    def _populate_counter_objects(self, state, counter_type='kitchen'):
        counter_objects = {}
        inv_counter_objects = self.mdp.get_counter_objects_dict(state, counter_subset=self.object_location[f'{counter_type[0]}']) 
        if 'onion' in inv_counter_objects:
            for pos in inv_counter_objects['onion']:
                counter_objects[pos] = 'onion'
        if 'dish' in inv_counter_objects:
            for pos in inv_counter_objects['dish']:
                counter_objects[pos] = 'plate'
        if 'soup' in inv_counter_objects:
            for pos in inv_counter_objects['soup']:
                counter_objects[pos] = 'soup in plate'
        for idx, pos in enumerate(self.object_location[f'{counter_type[0]}']):
            if pos in counter_objects:
                self.state_for_llm[f'{counter_type}_counter_objects'][idx] = counter_objects[pos]
            else:
                self.state_for_llm[f'{counter_type}_counter_objects'][idx] = 'empty'
    
    

    def get_stage_from_llm(self, state, other_player_message):
        state_dict = state.to_dict()

        # Deal with objects on the counter
        
        if self.layout_name in ['forced_coordination', 'counter_circuit_o_1order', 'soup_passing']:
            self._populate_counter_objects(state, counter_type='storage')
        
        self._populate_counter_objects(state, counter_type='kitchen')

        # print(f"{bcolors.FAIL}{self.state_for_llm['counter_objects']}{bcolors.ENDC}")
        # What are the 2 players holding? 
        if state_dict['players'][int(self.player_id)]['held_object'] != None:
            if state_dict['players'][int(self.player_id)]['held_object']['name']  == 'dish':
                self.state_for_llm[self.player_id]['held_object'] = 'plate'
            elif state_dict['players'][int(self.player_id)]['held_object']['name'] == 'soup':
               self.state_for_llm[self.player_id]['held_object'] = 'soup in plate' 
            else:
                self.state_for_llm[self.player_id]['held_object'] = state_dict['players'][int(self.player_id)]['held_object']['name']
        else:
            self.state_for_llm[self.player_id]['held_object'] = 'nothing'
        
        if state_dict['players'][int(self.other_player_id)]['held_object'] != None:
            if state_dict['players'][int(self.other_player_id)]['held_object']['name']  == 'dish':
                self.state_for_llm[self.other_player_id]['held_object'] = 'plate'
            elif state_dict['players'][int(self.other_player_id)]['held_object']['name'] == 'soup':
               self.state_for_llm[self.other_player_id]['held_object'] = 'soup in plate' 
            else:
                self.state_for_llm[self.other_player_id]['held_object'] = state_dict['players'][int(self.other_player_id)]['held_object']['name']
        else:
            self.state_for_llm[self.other_player_id]['held_object'] = 'nothing'
        


        
        # pots, num onions in each pot and status of each pot 

        self._populate_distances()
        self._populate_pot_states(state)
        for idx, gate in enumerate(self.object_location['g']):
            x, y = gate
            if self.mdp.terrain_mtx[y][x] == ' ':
                self.state_for_llm['gate_status'][idx] = 'open'
                self.state_for_llm['gate_open_time'][idx] = self.mdp.gate_states[gate][-1]
            else:
                self.state_for_llm['gate_status'][idx] = 'closed'
                self.state_for_llm['gate_open_time'][idx] = 0
        prev_stage = self.current_stage
        # if self.current_stage == 'put onion in c0.' and self.state_for_llm['num_onions_in_pot'][0] == 3:
        #     self.current_stage = 'turn on c0.'
        # elif self.current_stage == 'put onion in c1.' and self.state_for_llm['num_onions_in_pot'][1] == 3:
        #     self.current_stage = 'turn on c1.'
        # else:
        self.current_stage, self.message = self.llm_agent.get_player_action(self.state_for_llm, other_player_message)
        
        # if self.current_stage == 'move away from cooker.':
        #     if prev_stage.startswith('put onion in pot.'):
        #         self.current_stage = 'move away.'
        #     else:
        #         self.current_stage = prev_stage
        # if self.current_stage == 'wait.':
        #     # print("Inside wait: ", self.state_for_llm)
        #     if self.state_for_llm[self.player_id]['held_object'] == 'onion' and (self.state_for_llm['cooker_status'][0] == 'on' or self.state_for_llm['soup_in_cooker_status'][0] == 'cooked'):
        #         # print("MOVING AWAY FOR PLAYER ", self.player_id)
        #         self.current_stage = 'move away.'
        
        self.should_get_stage_from_llm = False 
    
    def generate_state_for_llm(self, state):
        state_dict = state.to_dict()

        # Deal with objects on the counter
        
        if self.layout_name in ['forced_coordination', 'counter_circuit_o_1order']:
            self._populate_counter_objects(state, counter_type='storage')
        
        self._populate_counter_objects(state, counter_type='kitchen')

        # print(f"{bcolors.FAIL}{self.state_for_llm['counter_objects']}{bcolors.ENDC}")
        # What are the 2 players holding? 
        if state_dict['players'][int(self.player_id)]['held_object'] != None:
            if state_dict['players'][int(self.player_id)]['held_object']['name']  == 'dish':
                self.state_for_llm[self.player_id]['held_object'] = 'plate'
            else:
                self.state_for_llm[self.player_id]['held_object'] = state_dict['players'][int(self.player_id)]['held_object']['name']
        else:
            self.state_for_llm[self.player_id]['held_object'] = 'nothing'
        
        if state_dict['players'][int(self.other_player_id)]['held_object'] != None:
            if state_dict['players'][int(self.other_player_id)]['held_object']['name']  == 'dish':
                self.state_for_llm[self.other_player_id]['held_object'] = 'plate'
            else:
                self.state_for_llm[self.other_player_id]['held_object'] = state_dict['players'][int(self.other_player_id)]['held_object']['name']
        else:
            self.state_for_llm[self.other_player_id]['held_object'] = 'nothing'
        
        # pots, num onions in each pot and status of each pot 

        self._populate_distances()
        self._populate_pot_states(state)

        return self.state_for_llm
    
    def process_message(self, state, other_player_message):
        if other_player_message == '':
            # Do nothing 
            return 
        else:
            print(f"[COMMUNICATION] Other player said: {other_player_message}")
            if self.current_stage in self.action_set['cooker']:
                self.current_stage = 'move away.'
            else:
                print("[COMMUNICATION] Carrying on action since it naturally moves me away. ")
                pass
            
    def set_player_pos_or(self, state):
        state_dict = state.to_dict()
        player_info = state_dict['players'][self.player_id]
        self.player_position = player_info['position']
        self.player_orientation = player_info['orientation']

        self.other_player_position = state_dict['players'][self.other_player_id]['position']

    def go_to_position(self, target_locs_arr, player_position, player_orientation):
        for target in target_locs_arr:
            path = self.find_shortest_path(player_position, target)
            if path:
                next_action = self.get_next_action(player_position, player_orientation, path)
                return next_action
            else:
                continue
        return Action.STAY

    def fix_orientation(self, position, orientation, target_position):
        desired_orientation = self.calculate_desired_orientation(position, target_position)
        if orientation == desired_orientation:
            return Action.STAY

        return desired_orientation

    def calculate_desired_orientation(self, position, target_position):
        dx = target_position[0] - position[0]
        dy = target_position[1] - position[1]

        if dx > 0:
            return Direction.EAST
        elif dx < 0:
            return Direction.WEST
        elif dy > 0:
            return Direction.SOUTH
        else:
            return Direction.NORTH

    def calculate_orientation_difference(self, orientation, desired_orientation):
  
        orientation_diff = (desired_orientation[0] - orientation[0], desired_orientation[1] - orientation[1])
        return orientation_diff

    def get_adjacent_empty_points(self, p1):
        x, y = p1
        adjacent_points = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        empty_points = []

        for point in adjacent_points:
            if point[0] >= 0 and point[1] >=0 and point[0] < self.mdp.width and point[1] < self.mdp.height:
                # # print(point, sys.stderr)
                
                if self.mdp.get_terrain_type_at_pos(point) == " " and point != self.other_player_position:
                    # # print('point and other player: ', point, self.other_player_position)    
                    empty_points.append(point)

        return empty_points
    
    def get_adjacent_accessible_points(self, p1):
        x, y = p1
        adjacent_points = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        accessible_points = []

        for point in adjacent_points:
            if point[0] >= 0 and point[1] >=0 and point[0] < self.mdp.width and point[1] < self.mdp.height:
                # # print(point, sys.stderr)
                
                if self.mdp.get_terrain_type_at_pos(point) == " ":
                    # # print('point and other player: ', point, self.other_player_position)    
                    accessible_points.append(point)
                elif point == self.other_player_position:
                    accessible_points.append(point)
        return accessible_points


    def get_nearest_unblocked_object_locations(self, player_x, player_y, obj):
        player_position = np.array([player_x, player_y])
        points_array = []
        pseudo_object_locations = []
        for location in self.object_location[obj]:
           adjc_points = self.get_adjacent_empty_points(location)
           # if points next to all possible objects of the type obj are blocked, just return None, None 
           
           points_array.extend(adjc_points)
           pseudo_object_locations.extend([location for _ in range(len(adjc_points))])
        if len(points_array) == 0:
            return None, None
        distances = np.linalg.norm(points_array - player_position, axis=1)
        locations_sorted_by_distance = [x for _, x in sorted(zip(distances, points_array))]
        pseudo_object_locations = [x for _, x in sorted(zip(distances, pseudo_object_locations))]
        
        return locations_sorted_by_distance, pseudo_object_locations

    def find_shortest_distance(self, p1, p_arr, other_player_position):
        # Given a starting point and an array of points, find the shortest distance to any point in the array
        shortest_distances = {p:math.inf for p in p_arr}
        for p2 in p_arr:
            directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
            queue = deque([(p1, 0)])
            visited = set()

            while queue:
                current_pos, distance = queue.popleft()

                if current_pos == p2:
                    shortest_distances[p2] = distance

                if current_pos in visited:
                    continue

                visited.add(current_pos)

                for dx, dy in directions:
                    new_x = current_pos[0] + dx
                    new_y = current_pos[1] + dy
                    new_pos = (new_x, new_y)

                    if (self.mdp.get_terrain_type_at_pos(new_pos) == " ") and new_pos not in visited:
                        if new_pos == other_player_position:
                            if new_pos == p2:
                                queue.append((new_pos, distance + 1))
                        else:
                            queue.append((new_pos, distance + 1))


        min_dist = math.inf
        min_dest = None
        for p, d in shortest_distances.items():
            if d < min_dist:
                min_dist = d 
                min_dest = p
        
        if min_dest == other_player_position:
            min_dist = 'blocked'
            # TODO: Uncomment this later and see how to fix  
            if len(shortest_distances.keys())>1:
                s_distances = [dd for pp, dd in shortest_distances.items() if (pp != min_dest and dd != math.inf)]
                if len(s_distances)>0:
                    min_dist = min(s_distances)
                
        
        elif min_dist == math.inf:
            min_dist = 'infinite'

            
        return min_dist, min_dest


    # BFS 
    # def find_shortest_path(self, p1, p2, other_player_position):
    #     # Given two points, find the shortest distance between them
    #     directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
    #     queue = deque([(p1, 0)])
    #     visited = set()

    #     while queue:
    #         current_pos, distance = queue.popleft()


    #         if current_pos == p2:
    #             return distance

    #         if current_pos in visited:
    #             continue

    #         visited.add(current_pos)

    #         for dx, dy in directions:
    #             new_x = current_pos[0] + dx
    #             new_y = current_pos[1] + dy
    #             new_pos = (new_x, new_y)

    #             if (self.mdp.get_terrain_type_at_pos(new_pos) == " ") and new_pos!=self.other_player_position and new_pos not in visited:

    #                 queue.append((new_pos, distance + 1))
    # A*
    # def find_shortest_path(self, p1, p2):
        
    #     def heuristic(p1, p2):
    #         # Manhattan distance
    #         return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
    #     directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
    #     open_list = PriorityQueue()
    #     open_list.put((0 + heuristic(p1, p2), (p1, [], 0)))  # f(n), (position, path, g(n))
    #     visited = set()

    #     while not open_list.empty():
    #         _, (current_pos, path, g_n) = open_list.get()

    #         if current_pos == p2:
    #             return path

    #         if current_pos in visited:
    #             continue

    #         visited.add(current_pos)

    #         for dx, dy in directions:
    #             new_x = current_pos[0] + dx
    #             new_y = current_pos[1] + dy
    #             new_pos = (new_x, new_y)

    #             if self.mdp.get_terrain_type_at_pos(new_pos) == " " and new_pos not in visited and new_pos != self.other_player_position:
    #                 new_g_n = g_n + 1  # Assuming a grid, so the cost is 1. Change this if your grid has different costs.
    #                 new_f_n = new_g_n + heuristic(new_pos, p2)
    #                 open_list.put((new_f_n, (new_pos, path + [new_pos], new_g_n)))

    #     return None
    
    def find_shortest_path(self, p1, p2):
        # Given two points, find the shortest path between them
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        queue = deque([(p1, [])])
        visited = set()

        while queue:
            current_pos, path = queue.popleft()

            if current_pos == p2:
                return path

            if current_pos in visited:
                continue

            visited.add(current_pos)

            for dx, dy in directions:
                new_x = current_pos[0] + dx
                new_y = current_pos[1] + dy
                new_pos = (new_x, new_y)

                if self.mdp.get_terrain_type_at_pos(new_pos) == " " and new_pos not in visited and new_pos != self.other_player_position:
                    queue.append((new_pos, path + [new_pos]))
                
                # TODO: Make a decision about this 
                # elif str(self.mdp.get_terrain_type_at_pos(new_pos)) in ["1", "2"]:
                #     self.should_get_stage_from_llm = True 

        return None


    def get_next_action(self, player_position, player_orientation, path):
        # Given a player position, orientation, and path, return the next action to take
        next_position = path[0]
        if next_position == player_position:
            return None  # Player is already at the next position
        dx = next_position[0] - player_position[0]
        dy = next_position[1] - player_position[1]
        if (dx, dy) in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            return (dx, dy)
        
        else:
            return Action.STAY  # Invalid move

