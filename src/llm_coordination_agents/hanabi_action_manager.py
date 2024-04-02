from __future__ import print_function

import numpy as np
from hanabi_learning_environment import pyhanabi
from llm_coordination_agents.hanabi_agent import LLMAgent
import datetime 

def ai_score(fireworks):
    return np.sum(fireworks)

def run_game(game_parameters, model_name):
    game = pyhanabi.HanabiGame(game_parameters)
    print(game.parameter_string(), end="")
    state = game.new_initial_state()
    Players = [LLMAgent(0, model_name), LLMAgent(1, model_name)]
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = 'random'
    if Players[0].model_type != 'openai':
        model_name = 'Mixtral'
    else:
        model_name = Players[0].model
    game_name = f'TEST_HanabiGamePlay_{time_stamp}_score_model_{model_name}_seed_{game_parameters["seed"]}.txt'
    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            state.deal_random_card()
            continue

        observation = state.observation(state.cur_player())
        episodic_memory = [Players[0].action_history, Players[1].action_history]
        working_memory = [Players[0].working_memory, Players[1].working_memory]
        # print_encoded_observations(obs_encoder, state, game.num_players())

        move = Players[state.cur_player()-1].get_next_move(observation, episodic_memory, working_memory)

        print("Selected Move: {}".format(move))

        state.apply_move(move)

    print("")
    print("Game done. Terminal state:")
    print("")
    print(state)
    print("")

    # This score is the total number of cards placed without considering bombing 
    print("Score: {}".format(ai_score(state.fireworks())))
    print("Bombed: {}".format('YES' if state.score() == 0 else 'NO'))
