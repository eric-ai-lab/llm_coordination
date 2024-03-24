import sys 
sys.path.append('..')
from collab_escape.CollabEscapeMDP import Game
from llm_coordination_agents.collab_escape_agent import LLMAgent


if __name__ == "__main__":
    game = Game()
    game.play()