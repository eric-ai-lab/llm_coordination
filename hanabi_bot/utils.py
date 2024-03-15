from typing import Dict
import config as conf
import vectorizer
import json_to_pyhanabi
from typing import Optional, Set, List
import enum

# Just for convenience, copied from pyhanabi
class HanabiMoveType(enum.IntEnum):
    """Move types, consistent with hanabi_lib/hanabi_move.h."""
    INVALID = 0
    PLAY = 1
    DISCARD = 2
    REVEAL_COLOR = 3
    REVEAL_RANK = 4
    DEAL = 5

def get_agent_config(game_config: Dict, agent: str):
    """ Performs look-up for agent in config.AGENT_CLASSES and returns individual config. The agent config must
    always be derived from the game_config. If it cannot be computed from it, its not an OpenAI Gym compatible agent.
    New agents may be added here"""

    if agent not in conf.AGENT_CLASSES:
        raise NotImplementedError

    if agent == 'rainbow_copy' or agent == 'rainbow':
        return dict({
            'observation_size': get_observation_size(game_config),
            'num_actions': get_num_actions(game_config),
            'num_players': game_config['num_total_players'],
            'history_size': 1
        }, **game_config)
    elif agent == 'simple':
        return {
            'players': game_config['num_total_players']
        }
    else:
        raise NotImplementedError


def get_observation_size(game_config):
    """ Returns the len of the vectorized observation """
    num_players = game_config['num_total_players']  # number of players ingame
    num_colors = game_config['colors']
    num_ranks = game_config['ranks']
    hand_size = game_config['hand_size']
    max_information_tokens = game_config['life_tokens']
    max_life_tokens = game_config['info_tokens']
    max_moves = game_config['max_moves']
    variant = game_config['variant']
    env = json_to_pyhanabi.create_env_mock(
        num_players=num_players,
        num_colors=num_colors,
        num_ranks=num_ranks,
        hand_size=hand_size,
        max_information_tokens=max_information_tokens,
        max_life_tokens=max_life_tokens,
        max_moves=max_moves,
        variant=variant
    )

    vec = vectorizer.ObservationVectorizer(env)
    legal_moves_vectorizer = vectorizer.LegalMovesVectorizer(env)
    return vec.total_state_length


def get_hand_size(players: int) -> int:
    """ Returns number of cards in each players hand, depending on the number of players """
    assert 1 < players < 6
    return 4 if players > 3 else 5


def get_num_actions(game_config):
    """ total number of moves possible each turn (legal or not, depending on num_players and num_cards),
    i.e. MaxDiscardMoves + MaxPlayMoves + MaxRevealColorMoves + MaxRevealRankMoves """
    num_players = game_config["players"]
    hand_size = get_hand_size(num_players)
    num_colors = game_config['colors']
    num_ranks = game_config['ranks']


    return 2 * hand_size + (num_players - 1) * num_colors + (num_players - 1) * num_ranks


def convert_color(color: str) -> Optional[int]:
    """
    Returns format desired by server
        // 0 is blue
        // 1 is green
        // 2 is yellow
        // 3 is red
        // 4 is purple
    """
    if color is None: return -1
    # if color == 'B': return 0
    # if color == 'G': return 1
    # if color == 'Y': return 2
    # if color == 'R': return 3
    # if color == 'W': return 4
    if color == 'R': return 0
    if color == 'Y': return 1
    if color == 'G': return 2
    if color == 'B': return 3
    if color == 'W': return 4
    return -1


def convert_suit(suit: int) -> Optional[str]:

    """
    Returns format desired by pyhanabi
    // 0 is blue
    // 1 is green
    // 2 is yellow
    // 3 is red
    // 4 is purple
    returns None if suit is None or -1
    """
    if suit == -1: return None
    # if suit == 0: return 'B'
    # if suit == 1: return 'G'
    # if suit == 2: return 'Y'
    # if suit == 3: return 'R'
    # if suit == 4: return 'W'
    if suit == 0: return 'R'
    if suit == 1: return 'Y'
    if suit == 2: return 'G'
    if suit == 3: return 'B'
    if suit == 4: return 'W'
    return None


def sort_colors(colors: Set) -> List:
    """ Sorts list, s.t. colors are in order RYGWB """
    result = list()
    for i in range(len(colors)):
        if 'R' in colors:
            colors.remove('R')
            result.append('R')
        if 'Y' in colors:
            colors.remove('Y')
            result.append('Y')
        if 'G' in colors:
            colors.remove('G')
            result.append('G')
        if 'W' in colors:
            colors.remove('W')
            result.append('W')
        if 'B' in colors:
            colors.remove('B')
            result.append('B')

    return result


def parse_rank_server(rank):
    """ Returns rank as expected by the gui """
    if int(rank) > -1:
            rank += 1

    return str(rank)


def get_target_from_offset(offset, agent_pos, num_players):

    """ Returns player index as desired by server. That means offset comes from an agents computation """
    # make up for the fact, that we changed the order of the agents, s.t. self always is at first position

    return str((offset + agent_pos) % num_players)
