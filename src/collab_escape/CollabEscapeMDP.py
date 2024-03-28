import random
from llm_coordination_agents.collab_escape_agent import LLMAgent
import re 
import time
import os 

class Room:
    def __init__(self, name, has_generator=False):
        self.name = name
        self.has_generator = has_generator
        self.generator_fixed = False
        self.fix_count = 0  # New attribute to count the number of fixes
        self.adjacent_rooms = []
        
    def fix_generator(self):
        if self.has_generator:
            self.fix_count += 1
            if self.fix_count >= 2:
                self.generator_fixed = True
            return True
        return False

class Player:
    def __init__(self, name, current_room):
        self.name = name
        self.current_room = current_room
        self.last_action_is_fixing = False

    # get room number from selected action
    def _extract_room(self, action_string):
        room_name = re.search(r"move to (.+)", action_string).group(1)
        return room_name
    
    # exeucte chosen action
    def move(self, action_string, rooms):
        if 'move' in action_string:
            target_room = self._extract_room(action_string)
            if rooms[target_room] in self.current_room.adjacent_rooms:
                self.current_room = rooms[target_room]
            else:
                # target room isn't adjacent
                pass
        elif "fix" in action_string:
            self.current_room.fix_generator()
        else:
            # Wait in the same room 
            pass 

class Adversary:
    def __init__(self, current_room):
        self.current_room = current_room
        self.target_name = ''
        self.can_see = {'Alice': False, 'Bob': False}

    def move_randomly(self):
        self.current_room = random.choice(self.current_room.adjacent_rooms)

    def move(self, chosen_room):
        # move to chosen room
        self.current_room = chosen_room
    
    def choose_greedily(self, state):
        room_adj_list =  [room.name for room in self.current_room.adjacent_rooms]
        print(state['Alice'].current_room.name)
        print(state['Bob'].current_room.name)
        
        # Both alice and bob are near, default to pursuing alice
        if state['Alice'].current_room.name in room_adj_list and state['Bob'].current_room.name in room_adj_list:
            self.target_name = 'Alice'
            self.can_see['Alice'] = True
            self.can_see['Bob'] = True
            return state['Alice'].current_room
        
        # Only alice is near, pursue
        elif state['Alice'].current_room.name in room_adj_list:    
            self.target_name = 'Alice'
            self.can_see['Alice'] = True
            self.can_see['Bob'] = False
            return state['Alice'].current_room

        # Only bob is near, pursue
        elif state['Bob'].current_room.name in room_adj_list:
            self.target_name = 'Bob'
            self.can_see['Alice'] = False
            self.can_see['Bob'] = True
            return state['Bob'].current_room

        # Neither Bob nor Alice are nearby, explore by choosing random room from adj rooms
        else:
            return random.choice(self.current_room.adjacent_rooms)
        
#            self.current_room = state['Alice'].current_room
#            self.current_room = state['Bob'].current_room
#
#
#            if self.can_see['Alice'] and self.can_see['Bob']:
#                
#                self.current_room = state[self.target_name].current_room
#            elif self.can_see['Alice']:
#                self.target_name = 'Bob'
#                self.current_room = state['Bob'].current_room
#            elif self.can_see['Bob']:
#                self.target_name = 'Alice'
#                self.current_room = state['Alice'].current_room
#            else:
#                self.target_name = 'Alice'
#                self.current_room = state['Alice'].current_room

        

class Game:
    def __init__(self):
        # Initialize rooms with thematic names
        room_names = ["room 1", "room 2", "room 3", "room 4", "room 5", "room 6", "room 7", "room 8"]
        self.rooms = {name: Room(name, has_generator=(name=="room 1" or name=="room 5")) for name in room_names}
        
        # Set adjacent rooms
        self.rooms["room 1"].adjacent_rooms = [self.rooms["room 2"], self.rooms["room 8"]]
        self.rooms["room 2"].adjacent_rooms = [self.rooms["room 1"], self.rooms["room 3"]]
        self.rooms["room 3"].adjacent_rooms = [self.rooms["room 2"], self.rooms["room 4"], self.rooms["room 7"]]
        self.rooms["room 4"].adjacent_rooms = [self.rooms["room 3"], self.rooms["room 5"]]
        self.rooms["room 5"].adjacent_rooms = [self.rooms["room 4"], self.rooms["room 6"]]
        self.rooms["room 6"].adjacent_rooms = [self.rooms["room 5"], self.rooms["room 7"]]
        self.rooms["room 7"].adjacent_rooms = [self.rooms["room 3"], self.rooms["room 6"], self.rooms["room 8"]]
        self.rooms["room 8"].adjacent_rooms = [self.rooms["room 1"], self.rooms["room 7"]]
        
        # Initialize players and adversary
        self.alice = Player("Alice", self.rooms["room 1"])
        self.bob = Player("Bob", self.rooms["room 2"])
        self.adversary = Adversary(self.rooms["room 6"])
        
        self.game_over = False

        # Initialize state
        self.state = {
            "Alice": self.alice,
            "Bob": self.bob,
            "Adversary": self.adversary,
            "Generators": {name: {"fixed": room.generator_fixed, "fix_count": room.fix_count} for name, room in self.rooms.items() if room.has_generator},
            "exit gate": False
        }

    # saves updated state values only, doesn't progress game state
    def update_state(self):
        self.state["Alice"] = self.alice
        self.state["Bob"] = self.bob
        self.state["Adversary"] = self.adversary
        self.state["Generators"] = {room.name: {"fixed": room.generator_fixed, "fix_count": room.fix_count} for room in self.rooms.values() if room.has_generator}
        self.state["exit gate"] = all(room.generator_fixed for room in self.rooms.values() if room.has_generator)

    # does game state match game over conditions?
    def check_game_over(self):
        if self.adversary.current_room == self.alice.current_room or self.adversary.current_room == self.bob.current_room:
            print("Game Over: Adversary caught a player.")
            self.game_over = True
            return "loss"

        if self.state["exit gate"]:
            if self.alice.current_room.name == "room 7" or self.bob.current_room.name == "room 7":
                print("Game Over: Players win!")
                self.game_over = True
                return "win"

        return "continue"

    def print_readable_state(self):
        state_info = "\nCurrent State:\n"
        state_info += "--------------\n"
        state_info += f"Alice is in {self.alice.current_room.name}\n"
        state_info += f"Bob is in {self.bob.current_room.name}\n"
        state_info += f"Adversary is in {self.adversary.current_room.name}\n"
        state_info += "Generators:\n"
        for room, info in self.state['Generators'].items():
            state_info += f"  - {room}: {info['fix_count']}/2 fixed\n"
        state_info += f"Exit Gate: {'Open' if self.state['exit gate'] else 'Closed'}\n"
        state_info += "--------------\n"

        # Write the state information to a file
        with open('game_state_gpt3.5_ToM5.txt', 'a') as file:
            file.write(state_info)
        # Print the state information to the console
        print(state_info)

    # handle main gameplay logic and flow
    def play(self):
        # create agents
        self.alice_llm_agent = LLMAgent(player_id=0)
        self.bob_llm_agent = LLMAgent(player_id=1)

        self.turn_count = 1
        while not self.game_over:
            current_state = self.state.copy()

            # Adversary's decision on where to move
            adversary_choice = self.adversary.choose_greedily(current_state)
            
            # intel on killer, provided to agents at inference
            killer_info = 'We have information that the killer is currently located in ' + self.adversary.current_room.name + ". "
            if self.adversary.target_name == '':
                killer_info += 'We don\'t know which room the killer will move to next, but we do know that he is unaware of our current location. '
            else:
                killer_info += 'We also have information that the killer will certainly move to the room where ' + self.adversary.target_name + ' is. '

            # Inference for both agent's action selection:
            # Alice's decision
            alice_action_string = self.alice_llm_agent.get_next_move(current_state, killer_info)
            
            # Bob's decision
            bob_action_string = self.bob_llm_agent.get_next_move(current_state, killer_info)
            

            # Reset generators fix count if its fixing stopped before completion (according to latest action decisions)
            reset_generator_list = self.rooms.copy()
            if "fix" in alice_action_string:
                reset_generator_list.pop(self.alice.current_room.name)
                self.alice.last_action_is_fixing = True
            else:
                self.alice.last_action_is_fixing = False
            if 'fix' in bob_action_string and self.bob.current_room.name in reset_generator_list.keys():
                reset_generator_list.pop(self.bob.current_room.name)
                self.bob.last_action_is_fixing = True
            else:
                self.bob.last_action_is_fixing = False
            for room in reset_generator_list.values():
                if room.generator_fixed == False:
                    room.fix_count = 0

            
            # execute action {move, fix, wait} for Alice and Bob
            self.alice.move(alice_action_string, self.rooms)
            self.bob.move(bob_action_string, self.rooms)

            # execute Adversary's move
            self.adversary.move(adversary_choice)
            
            # Update state
            self.update_state()
            self.print_readable_state()
            
            # Check game over conditions
            outcome = self.check_game_over()
            if outcome in ['loss', 'win']:
                return outcome, self.turn_count

            self.turn_count += 1

            time.sleep(2.)

