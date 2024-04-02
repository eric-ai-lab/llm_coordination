from llm_coordination_agents.collab_capture_action_manager import Environment, Agent, Thief, GreedyAgent
import time 
import numpy as np 
import argparse

parser = argparse.ArgumentParser(description='Run Collab Escape benchmark with a specific model.')
parser.add_argument('model_name', type=str, help='The name of the model to benchmark')
args = parser.parse_args()

model_name = args.model_name
print(f'Benchmarking model: {model_name}')

def game_loop_greedy_agents():
    environment = Environment()
    environment.doors_open[1] = True
    environment.doors_open[3] = False

    alice = GreedyAgent(1, 0, "Alice", environment)
    bob = GreedyAgent(6, 1, "Bob", environment)
    thief = Thief(2, "Thief", environment)

    start_time = time.time()
    num_turns = 0
    MAX_TURNS = 30

    while num_turns < MAX_TURNS:
        alice.decide_next_action(thief.current_room)
        bob.decide_next_action(thief.current_room)
        thief.plan_move_away_from_agents(alice.next_room, bob.next_room)

        alice.execute_move()
        bob.execute_move()
        thief.execute_move()

        print(f"Alice is in Room {alice.current_room}, Bob is in Room {bob.current_room}, Thief is in Room {thief.current_room}.")
        num_turns += 1

        if (thief.current_room == alice.current_room) or \
           (thief.current_room == bob.current_room):
            print("The thief has been caught.")
            break

        if num_turns >= MAX_TURNS:
            print("The thief could not be caught in time.")
            break

    print(f"Game took {time.time() - start_time} seconds and {num_turns} turns.")
    return num_turns

def game_loop(model):
    environment = Environment()
    environment.doors_open[1] = False 
    environment.doors_open[3] = True 

    
    #  = {1: True, 3: True}
    alice = Agent(1, 0, "Alice", environment, model)
    bob = Agent(6, 1, "Bob", environment, model)
    thief = Thief(2, "Thief", environment)
    start_time = time.time()
    num_turns = 0
    MAX_TURNS = 30
    while True:
        state_for_llm = environment.get_state_for_llm(alice, bob, thief)
        alice_action = alice.llm_agent.get_next_move(state_for_llm)
        if isinstance(alice_action, int):
            alice.plan_move(alice_action)
        elif alice_action.startswith("Press"):
            alice.plan_press_button()

        bob_action = bob.llm_agent.get_next_move(state_for_llm)
        if isinstance(bob_action, int):
            bob.plan_move(bob_action)
        elif bob_action.startswith("Press"):
            bob.plan_press_button()

        thief.plan_move_away_from_agents(alice.next_room, bob.next_room)

        alice.execute_move()
        bob.execute_move()
        thief.execute_move()

        print(f"Alice is in Room {alice.current_room}, Bob is in Room {bob.current_room}, Thief is in Room {thief.current_room}.")
        num_turns += 1
        # Check if the thief is caught
        if (thief.current_room == alice.current_room) or \
            (thief.current_room == bob.current_room) or \
            (alice.previous_room == thief.current_room and thief.previous_room == alice.current_room) or \
            (bob.previous_room == thief.current_room and thief.previous_room == bob.current_room):
            print(f"The thief has been caught. ")
            print(f"Turns to Completion: {num_turns}")
            break
        thief.previous_room = thief.current_room
        alice.previous_room = alice.current_room
        bob.previous_room = bob.current_room
        if num_turns >= MAX_TURNS:
            print(f"The thief could not be caught. ")
            print(f"Turns to Completion: {35}")
            break 
    

    print(f"GAME TOOK: {time.time() - start_time} seconds")
    return num_turns





if __name__ == '__main__':
    turn_count = []
    NUM_TRIALS = 3
    for i in range(NUM_TRIALS):
        nt = game_loop(model_name)
        turn_count.append(nt)
    
    print('Mean Turns to Capture: ', np.mean(turn_count))
    print('Standard Error: ', np.std(np.array(turn_count)) / np.sqrt(NUM_TRIALS))
    print(turn_count)
    