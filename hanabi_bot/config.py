from pathlib import Path
import os
import sys
path = os.path.dirname(sys.modules['__main__'].__file__)
path_simple_agent = Path(path + 'agents' + 'simple_agent')

path_rainbow_agent = Path(__file__ + 'agents' + 'agent_player')

""" agent_classes provides the possible args for the clients agent instantiation.
The keys of agent_classes provide the possible values for what you can call the client.py with. For instance

 python client.py -n 0 -a simple simple rainbow_copy

 will start a hanabigame with 0 human players and 3 AI agents,
 2 of them are simple agents and one is a rainbow_copy agent"""

AGENT_CLASSES = {
    'simple': {'filepath': path_simple_agent, 'class': 'SimpleAgent'},
    'rainbow_copy': {'filepath': path_rainbow_agent, 'class': 'RainbowPlayer'},
    'rainbow': {'filepath': path_rainbow_agent, 'class': 'RainbowPlayer'}
    # You can add your own agents here, simply follow this structure:
    # 'sys argument': {'filepath': path_sexy_agent, 'class': 'MySexyAgent'},
    # just make sure they match the imports in client.py and if not, simply add a corresponding import statement there
}
""" If you have terminated the client while a game was running, the default behaviour will make the agents return to
the game when the client is restarted. However, this is not always desired for instance when the number of players
change etc, so you can run the client with -e 1 which will make them finish this particular game and then idle so you
can restart with another parametrization. Unfortunately that is the only way to handle 'hanging' games (to finish them
first) """
