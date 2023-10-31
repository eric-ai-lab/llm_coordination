from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from llm_coordination_agents.overcooked_action_manager import LLMActionManager
from overcooked_ai_py.mdp.actions import Action, Direction
import time 

def main(layout_name):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    am = [LLMActionManager(mdp, 'player_0', layout_name), LLMActionManager(mdp, 'player_1', layout_name)]
    state = mdp.get_standard_start_state()
    # print(action)
    # game_messages[action_manager.player_id] = message 
    score = 0
    for tick in range(400):
        joint_action = [Action.STAY] * 2

        for i in range(2):
            action, message = am[i].get_next_move(state, '')
            joint_action[i] = action 
        print(joint_action)
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
        print("Current Tick is: ", tick)
        print(mdp.state_string(state))
        print(f"Current score is : {score}")
        # if tick > 225:
        time.sleep(2.5) # Delay to avoid overloading LLM API with calls 
            
# Change the argument to main() to test out different maps
main('forced_coordination')