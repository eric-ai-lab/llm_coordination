from __future__ import print_function
import numpy as np
from hanabi_learning_environment import pyhanabi
from llm_coordination_agents.hanabi_agent import LLMAgent
import datetime 
from llm_coordination_agents.hanabi_action_manager import run_game
import argparse

parser = argparse.ArgumentParser(description='Run Hanabi benchmark with a specific model.')
parser.add_argument('model_name', type=str, help='The name of the model to benchmark')
args = parser.parse_args()

model_name = args.model_name
print(f'Benchmarking model: {model_name}')

SEEDS = [321, 212, 628] # - Used during self-play
# SEEDS = [45, 114, 4788] # - Used during cross-play
if __name__ == "__main__":
    # Check that the cdef and library were loaded from the standard paths.
    assert pyhanabi.cdef_loaded(), "cdef failed to load"
    assert pyhanabi.lib_loaded(), "lib failed to load"
    # Run 3 trials for each model 
    #   for s in SEEDS:
    scores = []
    for s in SEEDS:
        run_game({"players": 2, "random_start_player": False, "seed": s}, model_name)
        with open("hanabi_agent_benchmarking.txt", "a") as f:
            f.write("Seed: " + str(s) + "\n")
        print("Seed: ", s)
        print("Scores: ", scores)

    print("Average Score: ", np.mean(scores))
    with open("hanabi_agent_benchmarking_average.txt", "w") as f:
        # write all scores
        f.write("Scores: " + str(scores) + "\n")
        # write average score
        f.write("Average Score: " + str(np.mean(scores)) + "\n")

    