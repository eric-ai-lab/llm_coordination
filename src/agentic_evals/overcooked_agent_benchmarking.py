from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from llm_coordination_agents.overcooked_action_manager import LLMActionManager
from overcooked_ai_py.mdp.actions import Action, Direction
import time 
import numpy as np 
from tqdm import tqdm 

def main(layout_name):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    am = [LLMActionManager(mdp, 'player_0', layout_name), LLMActionManager(mdp, 'player_1', layout_name)]
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
            print(f"Current Cost: {am[0].llm_agent.llm.gpt3_cost + am[1].llm_agent.llm.gpt3_cost}") 
            print(f"Current Cost (GPT4img): {am[0].llm_agent.llm.gpt4_cost + am[1].llm_agent.llm.gpt4_cost}") 
        print("Current Tick is: ", tick)
        print(mdp.state_string(state))
        print(f"Current score is : {score}")
        # if tick > 225:
        # time.sleep(0.5) # Delay to avoid overloading LLM API with calls 
    gpt_3_cost = am[0].llm_agent.llm.gpt3_cost + am[1].llm_agent.llm.gpt3_cost
    gpt_4_cost = am[0].llm_agent.llm.gpt4_cost + am[1].llm_agent.llm.gpt4_cost
    return score, gpt_3_cost, gpt_4_cost
    
# Change the argument to main() to test out different maps

if __name__ == '__main__':
    # LAYOUTS = ['forced_coordination', 'cramped_room', 'counter_circuit_o_1order', 'asymmetric_advantages', 'coordination_ring']
    LAYOUTS = ['no_counter_door']
    NUM_TRIALS = 1
    
    for layout_name in LAYOUTS:
        scores = []
        gpt_3_costs = []
        gpt_4_costs = []
        for idx in range(NUM_TRIALS):
            score, gpt3_cost, gpt4_cost = main(layout_name)
            scores.append(score)
            gpt_3_costs.append(gpt3_cost)
            gpt_4_costs.append(gpt4_cost) 
            print(' COSTS THIS TRIAL: ', gpt3_cost, gpt4_cost)

        with open(f'/home/saaket/llm_coordination/src/agentic_evals/{layout_name}.txt', 'w') as f:
            f.write("MODEL: GPT4-turbo",)
            f.write(f"MEAN SCORE: {np.mean(scores)}\n")
            f.write(f"STD ERROR: {np.std(np.array(scores)) / np.sqrt(NUM_TRIALS)}\n")
            f.write(f"SAMPLE SCORES: {scores}\n")
            f.write(f"TOTAL COST (GPT-35): {np.sum(gpt_3_costs)}\n")
            f.write(f"TOTAL COST (GPT-4t) (upper bound): {np.sum(gpt_4_costs)}\n")

    
