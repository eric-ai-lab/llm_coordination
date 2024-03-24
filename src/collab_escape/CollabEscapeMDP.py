import random
from llm_coordination_agents.collab_escape_agent import LLMAgent
import re 
import time 

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
    def move(self, target_room):
        if target_room in self.current_room.adjacent_rooms:
            self.current_room = target_room
            return True
        return False

    def fix_generator(self):
        return self.current_room.fix_generator()

class Adversary:
    def __init__(self, current_room):
        self.current_room = current_room
        self.target_name = ''
        self.can_see = {'Alice': False, 'Bob': False}
    def move_randomly(self):
        self.current_room = random.choice(self.current_room.adjacent_rooms)
    
    def move_greedily(self, state):
        room_adj_list =  [room.name for room in self.current_room.adjacent_rooms]
        
        # Both alice and bob are near, pursue one of them
        if state['Alice'].current_room.name in room_adj_list and state['Bob'].current_room.name in room_adj_list:
            if self.can_see['Alice'] and self.can_see['Bob']:
                
                self.current_room = state[self.target_name].current_room
            elif self.can_see['Alice']:
                self.target_name = 'Bob'
                self.current_room = state['Bob'].current_room
            elif self.can_see['Bob']:
                self.target_name = 'Alice'
                self.current_room = state['Alice'].current_room
            else:
                self.target_name = 'Alice'
                self.current_room = state['Alice'].current_room
        
        # Alice is near, pursue
        elif state['Alice'].current_room.name in room_adj_list:    
            self.current_room = state['Alice'].current_room
            self.target_name = 'Alice'
            self.can_see['Alice'] = True
            self.can_see['Bob'] = False
        # Bob is near, pursue
        elif state['Bob'].current_room.name in room_adj_list:
            self.current_room = state['Bob'].current_room
            self.target_name = 'Bob'
            self.can_see['Alice'] = False
            self.can_see['Bob'] = True
        # Neither Bob nor Alice are nearby, explore
        else:
            self.current_room = random.choice(self.current_room.adjacent_rooms) 
            self.target_name = ''
            self.can_see['Alice'] = False
            self.can_see['Bob'] = False
        

class Game:
    def __init__(self):
        # Initialize rooms with thematic names
        room_names = ["room 1", "room 2", "room 3", "room 4", "room 5", "room 6", "room 7"]
        self.rooms = {name: Room(name, has_generator=(name=="room 1" or name=="room 2")) for name in room_names}
        
        # Set adjacent rooms
        self.rooms["room 1"].adjacent_rooms = [self.rooms["room 2"], self.rooms["room 5"], self.rooms["room 7"]]
        self.rooms["room 2"].adjacent_rooms = [self.rooms["room 1"], self.rooms["room 6"], self.rooms["room 3"]]
        self.rooms["room 3"].adjacent_rooms = [self.rooms["room 2"], self.rooms["room 7"], self.rooms["room 5"], self.rooms["room 4"]]
        self.rooms["room 4"].adjacent_rooms = [self.rooms["room 3"], self.rooms["room 5"]]
        self.rooms["room 5"].adjacent_rooms = [self.rooms["room 1"], self.rooms["room 3"], self.rooms["room 4"], self.rooms["room 6"]]
        self.rooms["room 6"].adjacent_rooms = [self.rooms["room 2"], self.rooms["room 5"], self.rooms["room 7"]]
        self.rooms["room 7"].adjacent_rooms = [self.rooms["room 1"], self.rooms["room 3"], self.rooms["room 6"]]
        
        # Initialize players and adversary
        self.alice = Player("Alice", self.rooms["room 1"])
        self.bob = Player("Bob", self.rooms["room 2"])
        self.adversary = Adversary(self.rooms["room 3"])
        
        self.game_over = False

        # Initialize state
        self.state = {
            "Alice": self.alice,
            "Bob": self.bob,
            "Adversary": self.adversary,
            "Generators": {name: {"fixed": room.generator_fixed, "fix_count": room.fix_count} for name, room in self.rooms.items() if room.has_generator},
            "exit gate": False
        }

    # saves updated state values only, doesn't actually progress game state
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
            return True

        if self.state["exit gate"]:
            if self.alice.current_room.name == "room 7" or self.bob.current_room.name == "room 7":
                print("Game Over: Players win!")
                self.game_over = True
                return True

        return False

    # get room number from selected action
    def extract_room(self, action_string):
        room_name = re.search(r"move to (.+)", action_string).group(1)
        return room_name

    def print_readable_state(self):
        print("\nCurrent State:")
        print("--------------")
        print(f"Alice is in {self.state['Alice'].current_room}")
        print(f"Bob is in {self.state['Bob'].current_room}")
        print(f"Adversary is in {self.state['Adversary'].current_room}")
        print("Generators:")
        for room, info in self.state['Generators'].items():
            print(f"  - {room}: {info['fix_count']}/2 fixed")
        print(f"Exit Gate: {'Open' if self.state['exit gate'] else 'Closed'}")
        print("--------------\n")

    # handle main gameplay logic and flow
    def play(self):
        self.alice_llm_agent = LLMAgent(player_id=0)
        self.bob_llm_agent = LLMAgent(player_id=1)
        while not self.game_over:
            # Update state
            # self.update_state()
            # print("Current State:", self.state)

            current_state = self.state.copy()
            
            # Adversary's turn
            self.adversary.move_greedily(current_state)
            
            # Alice's turn
            killer_info = ''
            killer_info += 'Currently, we have information that the killer will certainly move to the room where ' + self.adversary.target_name + ' is. '

            alice_action_string = self.alice_llm_agent.get_next_move(current_state, killer_info)


            
            # Bob's turn
            bob_action_string = self.bob_llm_agent.get_next_move(current_state, killer_info)
                                                                #  'Now in this turn, Alice is going to ' \
                                                                #  + alice_action_string \
                                                                #  + '. ' + killer_info[:-2] + ' and target at ' + self.adversary.target_name + '. ')
            
            
            
            # Reset generators
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

            
            
            if 'move' in alice_action_string:
                target_room = self.extract_room(alice_action_string)
                self.alice.move(self.rooms[target_room])
            elif "fix" in alice_action_string:
                self.alice.fix_generator()
            else:
                # Wait in the same room 
                pass 
            
            if 'move' in bob_action_string:
                target_room = self.extract_room(bob_action_string)
                self.bob.move(self.rooms[target_room])
            elif "fix" in bob_action_string:
                self.bob.fix_generator()
            else:
                # Wait in the same room 
                pass 
            
            # Check game over conditions
            if self.check_game_over():
                break

            # Update state
            self.update_state()
            self.print_readable_state()

            time.sleep(2.)

