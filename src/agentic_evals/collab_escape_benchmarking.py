import sys 
import numpy as np

sys.path.append('..')
from collab_escape.CollabEscapeMDP import Game
from llm_coordination_agents.collab_escape_agent import LLMAgent


if __name__ == "__main__":
    results = []
    turn_counts = []
    NUM_TRIALS = 3
    for i in range(NUM_TRIALS):
        # init game
        game = Game()

        # run game, start to finish
        outcome, turns = game.play()

        results.append(outcome)
        turn_counts.append(turns)
        print(f"\n\n\n\nGAME {i} FINISHED\n\n\n\n")
    

    wins = results.count('win')
    escape_rate = (wins / NUM_TRIALS) * 100
   
    results_to_save = f'''
    Escape Rate: {escape_rate}%
    Average Turns to Escape: {np.mean(turn_counts)}
    Standard Error: {np.std(turn_counts) / np.sqrt(NUM_TRIALS)}
    Turn Counts: {turn_counts}
    '''

    # Save to a text file
    with open('collab_escape_results.txt', 'w') as file:
        file.write(results_to_save)
