import random 
import time 
from collab_capture_agent import LLMAgent
class Environment:
    def __init__(self):
        self.rooms = {
            1: [2, 6],
            2: [1, 3],
            3: [2, 4],
            4: [3, 5],
            5: [4, 6, 7],
            6: [5, 9, 1],
            7: [5, 8],
            8: [7],
            9: [6]
        }
        self.door_controls = {1: 9, 3: 8}  # Room: Button location
        self.doors_open = {1: True, 3: True}

    def toggle_door(self, control_room):
        # This method now just determines which door to toggle and returns that door's number
        for door, button_room in self.door_controls.items():
            if button_room == control_room:
                return door  # Return door number to be toggled
        return None  # No door is associated with this room

    def is_door_open(self, from_room, to_room):
        if (from_room in self.door_controls and to_room == from_room + 1) or (to_room in self.door_controls and from_room == to_room + 1):
            return self.doors_open.get(min(from_room, to_room), True)
        return True

    def get_accessible_rooms(self, current_room):
        return [r for r in self.rooms[current_room] if self.is_door_open(current_room, r)]

    def get_action_options(self, current_room):
        options = ['Stay'] + self.get_accessible_rooms(current_room)
        if current_room in self.door_controls.values():
            options.append(f"Press button in Room {current_room}")
        return options
    
    def get_state_for_llm(self, alice, bob, thief):
        state_for_llm = {'player_locs': {'Alice': alice.current_room, 'Bob': bob.current_room, 'Thief': thief.current_room}, 'door_states': {}}
        for door in self.door_controls:
            state_for_llm['door_states'][(door, door+1)] = 'open' if self.doors_open[door] else 'closed'
        state_for_llm['available_actions'] = {'Alice': None, 'Bob': None}
        state_for_llm['available_actions']['Alice'] = self.get_action_options(alice.current_room)
        state_for_llm['available_actions']['Bob'] = self.get_action_options(bob.current_room)

        return state_for_llm

class Agent:
    def __init__(self, start_room, id, name, environment):
        self.current_room = start_room
        self.next_room = start_room  # Where the agent plans to move next
        self.name = name
        self.id = id
        self.environment = environment
        self.press_button_next = False  # Whether the agent plans to press a button next
        self.llm_agent = LLMAgent(id)

    def plan_move(self, new_room):
        # This method now sets the next_room without actually moving
        if new_room in self.environment.get_accessible_rooms(self.current_room):
            self.next_room = new_room
        elif new_room == self.current_room:
            self.next_room = self.current_room

    def plan_press_button(self):
        if self.current_room in self.environment.door_controls.values():
            self.press_button_next = True

    def execute_move(self):
        # This method moves the agent to their next_room
        if self.press_button_next:
            door_to_toggle = self.environment.toggle_door(self.current_room)
            if door_to_toggle is not None:
                self.environment.doors_open[door_to_toggle] = not self.environment.doors_open[door_to_toggle]
                print(f"{self.name} pressed a button in Room {self.current_room}. Door {door_to_toggle} has been {'opened' if self.environment.doors_open[door_to_toggle] else 'closed'}.")
            self.press_button_next = False
        self.current_room = self.next_room

class Thief(Agent):
    def __init__(self, start_room, name, environment):
        self.current_room = start_room
        self.next_room = start_room  # Where the agent plans to move next
        self.name = name
        self.id = id
        self.environment = environment
        self.press_button_next = False  # Whether the agent plans to press a button next

    def plan_move_away_from_agents(self, alice_position, bob_position):
        accessible_rooms = self.environment.get_accessible_rooms(self.current_room)
        if not accessible_rooms:
            self.next_room = self.current_room
            return  # No movement if trapped
        # This method now calculates the next_room for the thief without moving
        furthest_room = max(accessible_rooms, key=lambda room: min(abs(room - alice_position), abs(room - bob_position)))
        self.next_room = furthest_room

def get_user_action(agent_name, action_options):
    print(f"{agent_name}, choose an action from the following options: {action_options}")
    choice = input()
    if choice.isdigit():
        return int(choice)
    else:
        return choice

