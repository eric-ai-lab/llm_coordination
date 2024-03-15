#!/usr/bin/python3
import typing
from typing import Dict
import json
import ast
import enum

import utils


def hello() -> str:
    """ Sent after joining game table """
    return 'hello {}'


def gameJoin(gameID: dict) -> str:
    """ To join game table from lobby """
    # return 'tableJoin {"gameID":' + gameID + '}'
    return 'tableJoin' + " " + json.dumps(gameID)


def ready() -> str:
    """ After hello() upon joining game table """
    return 'ready {}'


def gameUnattend():
    """ This will return agents to lobby """
    return 'gameUnattend {}'


def _get_table_params(config: Dict) -> Dict:
    """ This method is called when no human player is playing and an AI agent hosts a lobby.
    In order to open a game lobby he needs to send a json encoded lobby config, for which we get the params here."""

    # if they decide to change the lobby parameters on the server code some day, we have a problem when we want
    # to let our agents play remotely on Zamiels server, lol. But lets assume that this is very unlikely to happen.
    game_config = dict()
    game_config['name'] = config['table_name']
    game_config['variant'] = config['variant']
    game_config['timed'] = 'false'
    game_config['baseTime'] = 120
    game_config['timePerTurn'] = 20
    game_config['speedrun'] = 'false'
    game_config['deckPlays'] = 'false'
    game_config['emptyClues'] = str(config['empty_clues']).lower()  # parse bool flag to str
    game_config['characterAssignments'] = 'false'
    game_config['correspondence'] = 'false'
    game_config['password'] = config['table_pw']
    game_config['alertWaiters'] = 'false'

    return game_config


def gameCreate(config: Dict):
    lobby_config = _get_table_params(config)
    return 'gameCreate ' + json.dumps(lobby_config).replace('"false"', 'false').replace(''"true"'', 'true')


def gameStart():
    return 'gameStart {}'


def dict_from_response(response: str, msg_type: str = None) -> Dict:
    assert msg_type is not None
    d = ast.literal_eval(
        response.split(msg_type.strip() + ' ')[1].replace('false', 'False').replace('list', 'List').replace('true',
                                                                                                            'True')
    )
    return d


""" ACTIONS INGAME
clue: { // Not present if the type is 1 or 2
        type: 0, // 0 is a rank clue, 1 is a color clue
        value: 1, // If a rank clue, corresponds to the number
        // If a color clue:
        // 0 is blue
        // 1 is green
        // 2 is yellow
        // 3 is red
        // 4 is purple
        // (these mappings change in the mixed variants)
},
"""
CLUE = 0
PLAY = 1
DISCARD = 2
RANK_CLUE = 0
COLOR_CLUE = 1


def _action_clue(action, agent_pos, num_players):
    """ Returns action string that can be read by GUI server.
    This method is only called from inside get_server_msg_for_pyhanabi_action"""
    action_msg = ''

    if action['action_type'] == "REVEAL_COLOR":
        # compute absolute player position from target_offset
        target = str((action['target_offset'] + agent_pos) % num_players)
        # Change color representation to GUI
        cluevalue = str(utils.convert_color(action['color']))

        action_msg = 'action {"type":' + str(CLUE) + \
                     ',"target":' + target + \
                     ',"clue":{"type":' + str(COLOR_CLUE) + \
                     ',"value":' + cluevalue + '}}'
    elif action['action_type'] == "REVEAL_RANK":
        # compute absolute player position from target_offset
        target = str((action['target_offset'] + agent_pos) % num_players)
        # Change color representation to GUI
        cluevalue = str(utils.parse_rank_server(action['rank']))

        action_msg = 'action {"type":' + str(CLUE) + \
                     ',"target":' + target + \
                     ',"clue":{"type":' + str(RANK_CLUE) + \
                     ',"value":' + cluevalue + '}}'

    return action_msg


def _action_other(action, agent_pos, abs_card_nums, hand_size):
    """ Returns action string that can be read by GUI server
    This method is only called from inside get_server_msg_for_pyhanabi_action"""

    if action['action_type'] == 'PLAY':
        card_index = action['card_index']  # need for conversion from stack to fifo
        # Get GUI target from pyhanabi target_offset
        max_idx = hand_size - 1
        target = str(abs_card_nums[agent_pos][max_idx - card_index])

        return 'action {"type":' + str(PLAY) + \
            ',"target":' + target + '}'

    # -------- Convert DISCARD ----------- #
    if action['action_type'] == 'DISCARD':
        card_index = action['card_index']  # need for conversion from stack to fifo
        # Get GUI target from pyhanabi target_offset
        max_idx = hand_size - 1
        target = str(abs_card_nums[agent_pos][max_idx - card_index])

        return 'action {"type":' + str(DISCARD) + \
               ',"target":' + target + '}'
    else:
        raise ValueError


def get_server_msg_for_pyhanabi_action(action, abs_card_nums, agent_pos, num_players, hand_size):
    """ Takes an action dictionary as gotten from pyhanabi
    converts it to action string for GUI server """
    if action["action_type"] in ["REVEAL_COLOR", "REVEAL_RANK"]:
        return _action_clue(action, agent_pos, num_players)
    elif action["action_type"] in ["PLAY", "DISCARD"]:
        return _action_other(action, agent_pos, abs_card_nums, hand_size)
