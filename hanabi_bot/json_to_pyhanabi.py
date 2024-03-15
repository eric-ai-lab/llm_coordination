"""
    # ------------------------------------------------- # ''
    # ------------------ MOCK Classes ----------------  # ''
    # ------------------------------------------------- # ''
"""
# These are used to wrap the low level interface implemented in pyhanabi.py
# to be compatible with GUI server

""" 
SERVER ACTION EXAMPLES [CAN BE USED FOR TESTS]

############   DRAW   ##############
{"type":"draw","who":1,"rank":-1,"suit":-1,"order":11}
############   CLUE   ##############
{"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
############   PLAY   ##############
{"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
#  {type: "discard", failed: true, which: {index: 1, suit: 2, rank: 2, order: 8}} is also possible
############   DISCARD   ##############
{"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}

"""
import enum


"""
    # ------------------------------------------------- # ''
    # --------------------- Utils --------------------  # ''
    # ------------------------------------------------- # ''
"""


def get_move_type(move):
    """
    Input: 'type' value as sent by GUI server, e.g.
    {'type': 'play', 'which': {'index': 1, 'suit': 1, 'rank': 1, 'order': 9}}

    Output: Move types, consistent with hanabi_lib/hanabi_move.h.
        INVALID = 0
        PLAY = 1
        DISCARD = 2
        REVEAL_COLOR = 3
        REVEAL_RANK = 4
        DEAL = 5
    """
    if move['type'] == 'play':
        return HanabiMoveType.PLAY
    elif move['type'] == 'discard' and move['failed'] is False:  # when failed is True, discard comes from play
        return HanabiMoveType.DISCARD
    elif move['type'] == 'discard' and move['failed']:
        return HanabiMoveType.PLAY
    elif move['type'] == 'clue':
        if move['clue']['type'] == 1:  # rank clue
            return HanabiMoveType.REVEAL_RANK
        elif move['clue']['type'] == 0:  # color clue
            return HanabiMoveType.REVEAL_COLOR
    elif move['type'] == 'draw':
        return HanabiMoveType.DEAL

    return HanabiMoveType.INVALID


def get_move_card_index(move, deepcopy_card_nums, num_players):
    """
    Input: Move dictionary sent by server, e.g. one out of
        ############   DRAW   ##############
        {"type":"draw","who":1,"rank":-1,"suit":-1,"order":11}
        ############   CLUE   ##############
        {"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
        ############   PLAY   ##############
        {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
        #  {type: "discard", failed: true, which: {index: 1, suit: 2, rank: 2, order: 8}} is also possible
        ############   DISCARD   ##############
        {"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}
    Output: 0-based card index for PLAY and DISCARD moves, -1 otherwise."""
    card_index = -1
    if move['type'] == 'play' or move['type'] == 'discard':
        # abs_card_num ranges from 0 to |decksize|
        abs_card_num = move['which']['order']
        # get target player index
        print("inside get move card index")
        pid = move['which']['index']
        # get index of card with number abs_card_num in hand of player pid
        card_index = deepcopy_card_nums[pid].index(abs_card_num)
        # flip because hands are inverted in pyhanabi
        max_idx = num_players - 1
        card_index = max_idx - card_index
    return card_index


def get_target_offset(giver, target, num_players):
    """ Computes target_offset for this direction: gui -> pyhanabi """

    """
    Input: giver and target as set by GUI server
    e.g. The First player hints the second, then giver=0, target=1
    Note that when player 2 goes first then players[giver] equals 2.
    :param giver:  Absolute 0-indexed position at table
    :param target: Absolute 0-indexed position at table
    :param num_players: Number of players at the table
    :return: pyhanabi target offset
    
    (Returns target player offset for REVEAL_XYZ moves.)
    """
    if target is None or target == -1:
        return target
    else:
        # accounts for absolute player positions
        return target - giver + int(target < giver) * num_players


def suit_to_color(suit, move_type):

        """
        Returns format desired by agent
        // 0 is blue
        // 1 is green
        // 2 is yellow
        // 3 is red
        // 4 is purple
        returns None if suit is None or -1
        """
        
        if move_type == 'REVEAL' or "PLAY" or "DISCARD" or "DEAL":
            if suit == -1: return None
            if suit == 0: return 'Red' # 'B'
            if suit == 1: return 'Yellow' # 'G'
            if suit == 2: return 'Green' # 'Y'
            if suit == 3: return 'Blue' # 'R'
            if suit == 4: return 'White'
            else:
                return None

        # if move_type == "PLAY" or "DISCARD" or "DEAL":
        #     if suit == -1: return suit
        #     elif suit == 0: return 4  # 'B'
        #     elif suit == 1: return 2  # 'G'
        #     elif suit == 2: return 1  # 'Y'
        #     elif suit == 3: return 0  # 'R'
        #     elif suit == 4: return 3  # 'W'
        #     return -1 


def parse_rank_pyhanabi(rank):
    """ Subtracts 1 from rank values to account for 1-indexed ranks in GUI """
    if int(rank) > -1:
        rank -= 1
    return int(rank)


def get_move_rank(move):
    """
    Input: Move dictionary sent by server, e.g. one out of
        ############   DRAW   ##############
        {"type":"draw","who":1,"rank":-1,"suit":-1,"order":11}
        ############   CLUE   ##############
        {"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
        ############   PLAY   ##############
        {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
        #  {type: "discard", failed: true, which: {index: 1, suit: 2, rank: 2, order: 8}} is also possible
        ############   DISCARD   ##############
        {"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}

    Returns 0-based rank index for REVEAL_RANK and DEAL moves. We have to subtract 1 as the server uses
    1-indexed ranks, None for colorclues, and -1 otherwise
    """
    rank = -1
    # for REVEAL_RANK moves
    if move['type'] == 'clue':
        rankclue = bool(move['clue']['type'])  # 1 means rank clue, 0 means color clue
        if rankclue:
            rank = parse_rank_pyhanabi(move['clue']['value'])
        else:
            rank = -1
    # for DEAL moves
    if move['type'] == 'draw':
        rank = parse_rank_pyhanabi(move['rank'])

    # for PLAY moves
    if move['type'] == 'play' or move['type'] == 'discard':
        rank = parse_rank_pyhanabi(rank=move['which']['rank'])

    return rank


def get_move_color(move):
    """Returns 0-based color index for REVEAL_COLOR and DEAL moves."""

    """
       Input: Move dictionary sent by server, e.g. one out of
           ############   DRAW   ##############
           {"type":"draw","who":1,"rank":-1,"suit":-1,"order":11}
           ############   CLUE   ##############
           {"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
           ############   PLAY   ##############
           {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
           #  {type: "discard", failed: true, which: {index: 1, suit: 2, rank: 2, order: 8}} is also possible
           ############   DISCARD   ##############
           {"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}
        Output: 0-based color index for REVEAL_COLOR and DEAL moves
    """

    # R,Y,G,W,B map onto 0,1,2,3,4 in pyhanabi
    # 0, 1, 2, 3, 4 map onto B, G, Y, R, W on server
    color = None

    # for REVEAL_COLOR moves
    if move['type'] == 'clue':
        colorclue = not bool(move['clue']['type'])  # 0 means colo clue, 1 means rank clue
        if colorclue:
            suit = move['clue']['value']
            # map number to color
            color = suit_to_color(suit, move_type="REVEAL")

    elif move['type'] == 'play' or move['type'] == 'discard':
        suit = int(move['which']['suit'])
        color = suit_to_color(suit, move_type=move['type'].upper())

    return color

"""
    # ------------------------------------------------- # ''
    # ------------------- HanabiEnv ------------------  # ''
    # ------------------------------------------------- # ''
"""


class EnvMock:
    def __init__(self, num_players, num_colors, num_ranks, hand_size, max_information_tokens, max_life_tokens, max_moves, variant):
        self.num_players = num_players
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
        self.max_information_tokens = max_information_tokens
        self.max_life_tokens = max_life_tokens
        self.max_moves = max_moves
        self.variant = variant

    def num_cards(self, color, rank, variant):
        """ Input: Color string in "RYGWB" and rank in [0,4]
        Output: How often deck contains card with given color and rank, i.e. 1-cards will be return 3"""
        if rank == 0:
            return 3
        elif rank < 4:
            return 2
        elif rank == 4:
            return 1
        else:
            return 0


def create_env_mock(num_players, num_colors, num_ranks, hand_size, max_information_tokens, max_life_tokens, max_moves, variant):
    num_players = num_players
    num_colors = num_colors
    num_ranks = num_ranks
    hand_size = hand_size
    max_information_tokens = max_information_tokens
    max_life_tokens = max_life_tokens
    max_moves = max_moves
    variant = "Hanabi-Full"

    return EnvMock(
        num_players=num_players,
        num_colors=num_colors,
        num_ranks=num_ranks,
        hand_size=hand_size,
        max_information_tokens=max_information_tokens,
        max_life_tokens=max_life_tokens,
        max_moves=max_moves,
        variant=variant
    )


"""
    # ------------------------------------------------- # ''
    # --------------- HanabiHistoryItem --------------  # ''
    # ------------------------------------------------- # ''
"""


class HanabiHistoryItemMock:
    """ Returns object that immitates pyhanabi.HanabiMove instance """

    # We only need move, we could implement the rest on demand
    def __init__(self, move, player, scored, information_token, color, rank, card_info_revealed,
                 card_info_newly_revealed, deal_to_player):
        """A move that has been made within a game, along with the side-effects.
          For example, a play move simply selects a card index between 0-5, but after
          making the move, there is an associated color and rank for the selected card,
          a possibility that the card was successfully added to the fireworks, and an
          information token added if the firework stack was completed.
          Python wrapper of C++ HanabiHistoryItem class.
        """
        self._move = move
        self._player = player
        self._scored = scored
        self._information_token = information_token
        self._color = color
        self._rank = rank
        self._card_info_revealed = card_info_revealed
        self._card_info_newly_revealed = card_info_newly_revealed
        self._deal_to_player = deal_to_player

    def move(self):
        return self._move

    def player(self):
        return self._player

    def scored(self):
        """Play move succeeded in placing card on fireworks."""
        return self._scored

    def information_token(self):
        """Play/Discard move increased the number of information tokens."""
        return self._information_token

    def color(self):
        """Color index of card that was Played/Discarded."""
        raise NotImplementedError

    def rank(self):
        """Rank index of card that was Played/Discarded."""
        raise NotImplementedError

    def card_info_revealed(self):
        """Returns information about whether color/rank was revealed.
        Indices where card i color/rank matches the reveal move. E.g.,
        for Reveal player 1 color red when player 1 has R1 W1 R2 R4 __ the
        result would be [0, 2, 3].
        """
        return self._card_info_revealed

    def card_info_newly_revealed(self):
        """Returns information about whether color/rank was newly revealed.
        Indices where card i color/rank was not previously known. E.g.,
        for Reveal player 1 color red when player 1 has R1 W1 R2 R4 __ the
        result might be [2, 3].  Cards 2 and 3 were revealed to be red,
        but card 0 was previously known to be red, so nothing new was
        revealed. Card 4 is missing, so nothing was revealed about it.
        """
        raise NotImplementedError

    def deal_to_player(self):
        """player that card was dealt to for Deal moves."""
        raise NotImplementedError

    def __str__(self):

        # return str(self._move.to_dict()) + f"card_info_revealed{self._card_info_revealed}"
        obj_arr = [
        self._player,
        self._move._type,
        self._move._card_index,
        self._move._target_offset,
        self._move._color,
        self._move._rank,
        self._move._discard_move,
        self._move._play_move,
        self._move._reveal_color_move,
        self._move._reveal_rank_move,
        self._card_info_revealed
        ]
        str_arr = [
        "self._player",
        "self._move._type",
        "self._move._card_index",
        "self._move._target_offset",
        "self._move._color",
        "self._move._rank",
        "self._move._discard_move",
        "self._move._play_move",
        "self._move._reveal_color_move",
        "self._move._reveal_rank_move",
        "self._card_info_revealed"
        ]
        return str(list(zip(str_arr, obj_arr)))
    # def __repr__(self):
    #     return self.__str__(list(zip(str_arr, obj_arr)))


"""
    # ------------------------------------------------- # ''
    # ------------------- HanabiMove -----------------  # ''
    # ------------------------------------------------- # ''
"""


class HanabiMoveMock:
    """ Returns object that immitates pyhanabi.HanabiMove instance """

    def __init__(self, move_type, card_index, target_offset, color, rank, discard_move, play_move, reveal_color_move,
                 reveal_rank_move, move_dict):
        """Description of an agent move or chance event.
          Python wrapper of C++ HanabiMove class.
        """
        self._type = move_type
        self._card_index = card_index
        self._target_offset = target_offset
        self._color = color
        self._rank = rank
        self._discard_move = discard_move
        self._play_move = play_move
        self._reveal_color_move = reveal_color_move
        self._reveal_rank_move = reveal_rank_move
        self._move_dict = move_dict

    def type(self):
        """
            Move types, consistent with hanabi_lib/hanabi_move.h.
            INVALID = 0
            PLAY = 1
            DISCARD = 2
            REVEAL_COLOR = 3
            REVEAL_RANK = 4
            DEAL = 5
        """
        return self._type

    def card_index(self):
        """Returns 0-based card index for PLAY and DISCARD moves."""
        return self._card_index

    def target_offset(self):
        """Returns target player offset for REVEAL_XYZ moves."""
        return self._target_offset

    def color(self):
        """Returns 0-based color index for REVEAL_COLOR and DEAL moves."""
        return self._color

    def rank(self):
        """Returns 0-based rank index for REVEAL_RANK and DEAL moves."""
        return self._rank

    def get_discard_move(self, card_index):
        raise NotImplementedError

    def get_play_move(self, card_index):
        raise NotImplementedError

    def get_reveal_color_move(self, target_offset, color):
        """current player is 0, next player clockwise is target_offset 1, etc."""
        raise NotImplementedError

    def get_reveal_rank_move(self, target_offset, rank):
        """current player is 0, next player clockwise is target_offset 1, etc."""
        raise NotImplementedError

    def to_dict(self):
        return self._move_dict


class HanabiMoveType(enum.IntEnum):
    """Move types, consistent with hanabi_lib/hanabi_move.h."""
    INVALID = 0
    PLAY = 1
    DISCARD = 2
    REVEAL_COLOR = 3
    REVEAL_RANK = 4
    DEAL = 5


def get_pyhanabi_move_mock(dict_action, deepcopy_card_nums, num_players):

    """ dict_action looks like
    ############   DRAW   ##############
    {"type":"draw","who":1,"rank":-1,"suit":-1,"order":11}
    ############   CLUE   ##############
    {"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
    ############   PLAY   ##############
    {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
    #  {type: "discard", failed: true, which: {index: 1, suit: 2, rank: 2, order: 8}} is also possible
    ############   DISCARD   ##############
    {"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}

    """

    move_type = get_move_type(dict_action)
    card_index = get_move_card_index(dict_action, deepcopy_card_nums, num_players)
    if "target" in dict_action and "giver" in dict_action:
        target_offset = get_target_offset(dict_action['giver'], dict_action['target'], num_players)
    else:
        target_offset = -1
    color = get_move_color(dict_action)
    rank = get_move_rank(dict_action)

    discard_move = None
    play_move = None
    reveal_color_move = None
    reveal_rank_move = None

    move = HanabiMoveMock(
        move_type=move_type,
        card_index=card_index,
        target_offset=target_offset,
        color=color,
        rank=rank,
        # not implemented
        discard_move=discard_move,
        play_move=play_move,
        reveal_color_move=reveal_color_move,
        reveal_rank_move=reveal_rank_move,
        move_dict=None
    )
    return move