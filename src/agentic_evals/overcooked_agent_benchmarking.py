from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from llm_coordination_agents.overcooked_action_manager import LLMActionManager
from overcooked_ai_py.mdp.actions import Action, Direction
import time 
import numpy as np 
from tqdm import tqdm 
import argparse 

def main(layout_name, model_name):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    am = [LLMActionManager(mdp, 'player_0', layout_name, model_name), LLMActionManager(mdp, 'player_1', layout_name, model_name)]
    state = mdp.get_standard_start_state()
    print(am[0].llm_agent.model)
    # print(action)
    # game_messages[action_manager.player_id] = message 
    score = 0
    NUM_TICKS = 400
    for tick in tqdm(range(NUM_TICKS)):
        joint_action = [Action.STAY] * 2

        for i in range(2):
            action, message = am[i].get_next_move(state, '')
            joint_action[i] = action 
        # print(joint_action)
        # Apply overcooked game logic to get state transition
        prev_state = state
        state, sparse_reward, shaped_reward = mdp.get_state_transition(
            prev_state, joint_action
        )
        info = {
            'sparse_reward_by_agent': sparse_reward, 
            'shaped_reward_by_agent': shaped_reward
        }
            
        # Update score based on soup deliveries that might have occured
        curr_reward = sparse_reward
        score += curr_reward
        if tick % 50 == 0:
            print(f"Current Score: {score}")
        print("Current Tick is: ", tick)
        print(mdp.state_string(state))
        print(f"Current score is : {score}")
        time.sleep(0.5) # Delay to avoid overloading LLM API with calls 
    return score
    

parser = argparse.ArgumentParser(description='Run Overcooked benchmark with a specific model.')
parser.add_argument('model_name', type=str, help='The name of the model to benchmark')
args = parser.parse_args()

model_name = args.model_name
print(f'Benchmarking model: {model_name}')


if __name__ == '__main__':
    LAYOUTS = ['forced_coordination', 'cramped_room', 'counter_circuit_o_1order', 'asymmetric_advantages', 'coordination_ring']
    NUM_TRIALS = 3
    
    for layout_name in LAYOUTS:
        scores = []
        gpt_3_costs = []
        gpt_4_costs = []
        for idx in range(NUM_TRIALS):
            score = main(layout_name, model_name)
            scores.append(score)

        with open(f'{layout_name}.txt', 'w') as f:
            f.write("MODEL: GPT4-turbo",)
            f.write(f"MEAN SCORE: {np.mean(scores)}\n")
            f.write(f"STD ERROR: {np.std(np.array(scores)) / np.sqrt(NUM_TRIALS)}\n")
            f.write(f"SAMPLE SCORES: {scores}\n")

    
