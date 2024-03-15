import numpy as np
from itertools import count
np.set_printoptions(threshold=np.inf)

COLOR_CHAR = ["R", "Y", "G", "W", "B"]

PLAY = 1
DISCARD = 2
REVEAL_COLOR = 3
REVEAL_RANK = 4
DEAL = 5


'''
Used to vectorize/encode player-dependent state-dicts and action dicts that are used
for trained agents to interact with the UI-environment
For more details check:
GitHub Wiki - Hanabi Env Doc - Encodings
'''

"""
UTILS
"""
def color_idx_to_char(color_idx):
  """Helper function for converting color index to a character.

  Args:
    color_idx: int, index into color char vector.

  Returns:
    color_char: str, A single character representing a color.

  Raises:
    AssertionError: If index is not in range.
  """
  assert isinstance(color_idx, int)
  if color_idx == -1:
    return None
  else:
    return COLOR_CHAR[color_idx]


def color_char_to_idx(color_char):
  r"""Helper function for converting color character to index.

  Args:
    color_char: str, Character representing a color.

  Returns:
    color_idx: int, Index into a color array \in [0, num_colors -1]

  Raises:
    ValueError: If color_char is not a valid color.
  """
  assert isinstance(color_char, str)
  try:
    return next(idx for (idx, c) in enumerate(COLOR_CHAR) if c == color_char)
  except StopIteration:
    raise ValueError("Invalid color: {}. Should be one of {}.".format(
        color_char, COLOR_CHAR))


class HandKnowledge(object):
    '''
    Used to encode implicit hints
    '''
    def __init__(self, hand_size, num_ranks, num_colors):
        self.num_ranks = num_ranks
        self.num_colors = num_colors
        self.hand = [CardKnowledge(num_ranks, num_colors) for _ in range(hand_size)]

    def sync_colors(self, card_ids, color):

        for cid in range(len(self.hand)):
            if cid in card_ids:
                for c in range(len(self.hand[cid].colors)):
                    if c != color:
                        self.hand[cid].colors[c] = None
            else:
                self.hand[cid].colors[color] = None

    def sync_ranks(self, card_ids, rank):

        for cid in range(len(self.hand)):
            if cid in card_ids:
                for r in range(len(self.hand[cid].ranks)):
                    if r != rank:
                        self.hand[cid].ranks[r] = None
            else:
                self.hand[cid].ranks[rank] = None

    def remove_card(self, card_id, deck_empty = False):
        new_hand = []
        for c_id,card in enumerate(self.hand):
            if c_id != card_id:
                new_hand.append(card)

        # NOTE: STOP REFILLING IF DECK IS EMPTY
        if not deck_empty:
            while len(new_hand) < len(self.hand):
                new_hand.append(CardKnowledge(self.num_ranks, self.num_colors))

        self.hand = new_hand

        #print("\n=========================")
        #print("NEW HAND AFTER DISCARDING/PLAYING CARD")
        #print(new_hand)
        #print("==========================\n")

'''
Used to encode implicit hints
'''

class CardKnowledge(object):
    def __init__(self, num_ranks, num_colors):
        self.colors = [c for c in range(num_colors)]
        self.ranks = [r for r in range(num_ranks)]

    def color_plausible(self, color):
        # print("\n INSIDE COLOR PLAUSIBLE FUNCTION")
        # print(self.colors[color])
        # print("\n")
        return self.colors[color] != None

    def rank_plausible(self, rank):
        # print("\n INSIDE RANK PLAUSIBLE FUNCTION")
        # print(self.ranks[rank])
        # print("\n")
        return self.ranks[rank] != None

    def remove_rank(self, rank):
        if rank in self.ranks:
            self.ranks[rank] = None

    def remove_color(self, color):
        if color in self.colors:
            self.colors[color] = None

class ObservationVectorizer(object):
    _ids = count(0)
    @property
    def knowledge(self):
        return self.__class__.knowledge
    @knowledge.setter
    def knowledge(self, player_knowledge):
        self.__class__.knowledge = player_knowledge

    def __init__(self, env):
        '''
        Encoding Order =
         HandEncoding
        +BoardEncoding
        +DiscardEncoding
        +LastAcionEncoding
        +CardKnowledgeEncoding
        '''
        self.id = next(self._ids)
        self.env = env
        self.obs = None
        self.num_players = self.env.num_players
        self.num_colors = self.env.num_colors
        self.num_ranks = self.env.num_ranks
        self.hand_size = self.env.hand_size
        self.max_info_tokens = self.env.max_information_tokens
        self.max_life_tokens = self.env.max_life_tokens
        self.max_moves = self.env.max_moves
        self.bits_per_card = self.num_colors * self.num_ranks
        self.max_deck_size = 0
        self.variant = self.env.variant
        # start of the vectorized observation
        self.offset = None

        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                self.max_deck_size += self.env.num_cards(color, rank, self.variant)
        """ Bit lengths """
        # Compute total state length
        self.hands_bit_length = (self.num_players - 1) * self.hand_size * self.bits_per_card + self.num_players


        self.board_bit_length = self.max_deck_size - self.num_players * \
                                self.hand_size + self.num_colors * self.num_ranks \
                                + self.max_info_tokens + self.max_life_tokens


        self.discard_pile_bit_length = self.max_deck_size


        self.last_action_bit_length = self.num_players + 4 + self.num_players + \
                                      self.num_colors + self.num_ranks \
                                      + self.hand_size + self.hand_size + self.bits_per_card + 2

        self.card_knowledge_bit_length = self.num_players * self.hand_size * \
                                         (self.bits_per_card + self.num_colors + self.num_ranks)

        self.total_state_length = self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length \
                                  + self.last_action_bit_length + self.card_knowledge_bit_length
        self.obs_vec = np.zeros(self.total_state_length)
        if self.id == 0:

            self.player_knowledge = [HandKnowledge(
                self.hand_size, self.num_ranks, self.num_colors) for _ in range(self.num_players)
            ]

            self.knowledge = self.player_knowledge

        else:
            self.player_knowledge = self.knowledge

        self.last_player_action = None

    def get_vector_length(self):
        return self.total_state_length

    def vectorize_observation(self, obs):

        self.obs_vec = np.zeros(self.total_state_length)
        self.obs = obs

        if obs["last_moves"] != []:

            if obs["last_moves"][0].move().type() != DEAL:
                self.last_player_action = obs["last_moves"][0]

            elif len(obs["last_moves"]) >= 2:
                self.last_player_action = obs["last_moves"][1]

            else:
                 self.last_player_action = None

        self.encode_hands(obs)
        #print("OFFSET END ENCODE HANDS", self.offset)
        offset_encode_hands = self.offset
        #print("SUM END ENCODE HANDS", sum(self.obs_vec[:self.offset]))

        self.encode_board(obs)
        #print("OFFSET END ENCODE BOARDS", self.offset)
        offset_encode_boards = self.offset
        #print("SUM END ENCODE BOARDS", sum(self.obs_vec[offset_encode_hands:self.offset]))
        self.encode_discards(obs)
        #print("OFFSET END ENCODE DISCARDS", self.offset)
        offset_encode_discards = self.offset
        #print("SUM END ENCODE DISCARDS", sum(self.obs_vec[offset_encode_boards:self.offset]))
        self.encode_last_action()
        #print("OFFSET END ENCODE LAST ACTION", self.offset)
        offset_encode_last_action = self.offset
        #print("SUM END ENCODE LAST_ACTION", sum(self.obs_vec[offset_encode_discards:self.offset]))
        self.encode_card_knowledge(obs)
        #print("OFFSET END ENCODE CARD KNOWLEDGE", self.offset)

        #print("SUM END ENCODE CARD_KNOWLEDGE", sum(self.obs_vec[offset_encode_last_action:self.offset]))

        self.knowledge = self.player_knowledge

        return self.obs_vec

    '''Encodes cards in all other player's hands (excluding our unknown hand),
     and whether the hand is missing a card for all players (when deck is empty.)
     Each card in a hand is encoded with a one-hot representation using
     <num_colors> * <num_ranks> bits (25 bits in a standard game) per card.
     Returns the number of entries written to the encoding.'''
    def encode_hands(self, obs):
        self.offset = 0
        # don't use own hand
        hands = obs["observed_hands"]
        for player_hand in hands:
            if player_hand[0]["color"] is not None:
                num_cards = 0
                for card in player_hand:
                    rank = card["rank"]
                    color = color_char_to_idx(card["color"])
                    card_index = color * self.num_ranks + rank

                    self.obs_vec[self.offset + card_index] = 1
                    num_cards += 1
                    self.offset += self.bits_per_card

                '''
                A player's hand can have fewer cards than the initial hand size.
                Leave the bits for the absent cards empty (adjust the offset to skip
                bits for the missing cards).
                '''

                if num_cards < self.hand_size:
                    self.offset += (self.hand_size - num_cards) * self.bits_per_card

        # For each player, set a bit if their hand is missing a card
        i = 0
        for i, player_hand in enumerate(hands):
            if len(player_hand) < self.hand_size:
                self.obs_vec[self.offset + i] = 1
        self.offset += self.num_players


        assert self.offset - self.hands_bit_length == 0
        return True
    '''
    Encode the board, including:
       - remaining deck size
         (max_deck_size - num_players * hand_size bits; thermometer)
       - state of the fireworks (<num_ranks> bits per color; one-hot)
       - information tokens remaining (max_information_tokens bits; thermometer)
       - life tokens remaining (max_life_tokens bits; thermometer)
     We note several features use a thermometer representation instead of one-hot.
     For example, life tokens could be: 000 (0), 100 (1), 110 (2), 111 (3).
     Returns the number of entries written to the encoding.
    '''
    def encode_board(self, obs):
        # encode the deck size:
        for i in range(obs["deck_size"]):
            self.obs_vec[self.offset + i] = 1
        self.offset += self.max_deck_size - self.hand_size * self.num_players

        # encode fireworks
        fireworks = obs["fireworks"]
        for c in range(len(fireworks)):
            color = color_idx_to_char(c)
            # print(fireworks[color])
            if fireworks[color] > 0:
                self.obs_vec[self.offset + fireworks[color] - 1] = 1
            self.offset += self.num_ranks

        # encode info tokens
        info_tokens = obs["information_tokens"]
        for t in range(info_tokens):
            self.obs_vec[self.offset + t] = 1
        self.offset += self.max_info_tokens

        # encode life tokens

        life_tokens = obs["life_tokens"]
        #print(f"BAD lifetokens = {life_tokens}")
        for l in range(life_tokens):
            self.obs_vec[self.offset + l] = 1
        self.offset += self.max_life_tokens

        assert self.offset - (self.hands_bit_length + self.board_bit_length) == 0
        return True

    '''
    Encode the discard pile. (max_deck_size bits)
    Encoding is in color-major ordering, as in kColorStr ("RYGWB"), with each
    color and rank using a thermometer to represent the number of cards
    discarded. For example, in a standard game, there are 3 cards of lowest rank
    (1), 1 card of highest rank (5), 2 of all else. So each color would be
    ordered like so:

      LLL      H
      1100011101

    This means for this color:
      - 2 cards of the lowest rank have been discarded
      - none of the second lowest rank have been discarded
      - both of the third lowest rank have been discarded
      - one of the second highest rank have been discarded
      - the highest rank card has been discarded
    Returns the number of entries written to the encoding.
    '''

    def encode_discards(self, obs):
        discard_pile = obs['discard_pile']
        counts = np.zeros(self.num_colors * self.num_ranks)
        #print(f"GBAD discard_pile = {discard_pile}")
        #print(f"GBAD lifes = {obs['life_tokens']}")
        for card in discard_pile:
            color = color_char_to_idx(card["color"])
            rank = card["rank"]
            counts[color * self.num_ranks + rank] += 1

        for c in range(self.num_colors):
            for r in range(self.num_ranks):
                num_discarded = counts[c * self.num_ranks + r]
                for i in range(int(num_discarded)):
                    self.obs_vec[self.offset + i] = 1
                self.offset += self.env.num_cards(c, r, self.variant)

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length) == 0
        return True


    '''
    Encode the last player action (not chance's deal of cards). This encodes:
      - Acting player index, relative to ourself (<num_players> bits; one-hot)
      - The MoveType (4 bits; one-hot)
      - Target player index, relative to acting player, if a reveal move
        (<num_players> bits; one-hot)
      - Color revealed, if a reveal color move (<num_colors> bits; one-hot)
      - Rank revealed, if a reveal rank move (<num_ranks> bits; one-hot)
      - Reveal outcome (<hand_size> bits; each bit is 1 if the card was hinted at)
      - Position played/discarded (<hand_size> bits; one-hot)
      - Card played/discarded (<num_colors> * <num_ranks> bits; one-hot)
    Returns the number of entries written to the encoding.
    '''

    def encode_last_action(self):
        if self.last_player_action is None:
            self.offset += self.last_action_bit_length
        else:
            last_move_type = self.last_player_action.move().type()
            #print(f"player inside broken vec: {self.last_player_action.player()}")
            #print(f"action inside broken vec: "
            #      f"{self.last_player_action.player()},"
            #      f"{self.last_player_action.move().color()},"
            #      f"{self.last_player_action.move().rank()},")
            #print(f"COLORBAD: {self.last_player_action.move().color()}")
            #print(f"RANKBAD: {self.last_player_action.move().rank()}")
            '''
            player_id
            Note: no assertion here. At a terminal state, the last player could have
            been me (player id 0).
            '''

            self.obs_vec[self.offset + self.last_player_action.player()] = 1
            self.offset += self.num_players

            # encode move type
            if last_move_type == PLAY:
                self.obs_vec[self.offset] = 1
            elif last_move_type == DISCARD:
                self.obs_vec[self.offset + 1] = 1
            elif last_move_type == REVEAL_COLOR:
                self.obs_vec[self.offset + 2] = 1
            elif last_move_type == REVEAL_RANK:
                self.obs_vec[self.offset + 3] = 1
            else:
                print("ACTION UNKNOWN")
            self.offset += 4

            # encode target player (if hint action)
            if last_move_type == REVEAL_COLOR or last_move_type == REVEAL_RANK:
                observer_relative_target = (self.last_player_action.player() + self.last_player_action.move().target_offset()) % self.num_players

                self.obs_vec[self.offset + observer_relative_target] = 1

            self.offset += self.num_players

            # encode color (if hint action)
            if last_move_type == REVEAL_COLOR:
                last_move_color = self.last_player_action.move().color()

                self.obs_vec[self.offset + color_char_to_idx(last_move_color)] = 1

            self.offset += self.num_colors

            # encode rank (if hint action)
            if last_move_type == REVEAL_RANK:
                last_move_rank = self.last_player_action.move().rank()
                self.obs_vec[self.offset + last_move_rank] = 1

            self.offset += self.num_ranks

            # If multiple positions where selected
            if last_move_type == REVEAL_COLOR or last_move_type == REVEAL_RANK:
                positions = self.last_player_action.card_info_revealed()
                for pos in positions:
                    self.obs_vec[self.offset + pos] = 1

            self.offset += self.hand_size

            # encode position (if play or discard action)
            if last_move_type == PLAY or last_move_type == DISCARD:

                card_index = self.last_player_action.move().card_index()
                #print(f"BAD card_index={card_index}")
                self.obs_vec[self.offset + card_index] = 1

            self.offset += self.hand_size


            # encode card (if play or discard action)
            if last_move_type == PLAY or last_move_type == DISCARD:
                card_index_hgame = self.last_player_action.move().color() * self.num_ranks + \
                                   self.last_player_action.move().rank()
                # print(self.offset + card_index_hgame)
                self.obs_vec[self.offset + card_index_hgame] = 1

            self.offset += self.bits_per_card

            if last_move_type == PLAY:
                if self.last_player_action.scored():
                    self.obs_vec[self.offset] = 1

                # IF INFO TOKEN WAS ADDED
                if self.last_player_action.information_token():
                    self.obs_vec[self.offset + 1] = 1

            self.offset += 2

        assert self.offset - (
                self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length) == 0
        return True


    '''
     Encode the common card knowledge.
     For each card/position in each player's hand, including the observing player,
     encode the possible cards that could be in that position and whether the
     color and rank were directly revealed by a Reveal action. Possible card
     values are in color-major order, using <num_colors> * <num_ranks> bits per
     card. For example, if you knew nothing about a card, and a player revealed
     that is was green, the knowledge would be encoded as follows.
     R    Y    G    W    B
     0000000000111110000000000   Only green cards are possible.
     0    0    1    0    0       Card was revealed to be green.
     00000                       Card rank was not revealed.

     Similarly, if the player revealed that one of your other cards was green, you
     would know that this card could not be green, resulting in:
     R    Y    G    W    B
     1111111111000001111111111   Any card that is not green is possible.
     0    0    0    0    0       Card color was not revealed.
     00000                       Card rank was not revealed.
     Uses <num_players> * <hand_size> *
     (<num_colors> * <num_ranks> + <num_colors> + <num_ranks>) bits.
     Returns the number of entries written to the encoding.
    '''

    def encode_card_knowledge(self, obs):

        card_knowledge_list = obs["card_knowledge"]
        current_pid = obs["current_player"]
        action = self.last_player_action

        if action:  # comparison is equal to 'if action != []'
            type_action = self.last_player_action.move().type()

            if type_action in [REVEAL_COLOR, REVEAL_RANK]:
                player_hand_to_sync = (
                    action.player() +
                    action.move().target_offset() +
                    current_pid
                ) % self.num_players
                card_pos_to_sync = self.last_player_action.card_info_revealed()

                if type_action == REVEAL_COLOR:
                    color_to_sync = color_char_to_idx(self.last_player_action.move().color())
                    self.player_knowledge[player_hand_to_sync].sync_colors(card_pos_to_sync, color_to_sync)

                elif type_action == REVEAL_RANK:
                    rank_to_sync = self.last_player_action.move().rank()
                    self.player_knowledge[player_hand_to_sync].sync_ranks(card_pos_to_sync, rank_to_sync)

            elif type_action in [PLAY, DISCARD]:

                player_hand_to_sync = (action.player() + current_pid) % self.num_players
                card_id = action.move().card_index()

                self.player_knowledge[player_hand_to_sync].remove_card(card_id)

        for pid, player_card_knowledge in enumerate(card_knowledge_list):
            num_cards = 0
            rel_player_pos = (current_pid + pid) % self.num_players

            for card_id, card in enumerate(player_card_knowledge):
                for color in range(self.num_colors):

                    if self.player_knowledge[rel_player_pos].hand[card_id].color_plausible(color):
                        for rank in range(self.num_ranks):

                            if self.player_knowledge[rel_player_pos].hand[card_id].rank_plausible(rank):
                                card_index = color * self.num_ranks + rank
                                self.obs_vec[self.offset + card_index] = 1

                self.offset += self.bits_per_card

                # Encode explicitly revealed colors and ranks
                if card["color"] is not None:
                    color = color_char_to_idx(card["color"])
                    self.obs_vec[self.offset + color] = 1

                self.offset += self.num_colors

                if card["rank"] is not None:
                    rank = card["rank"]
                    self.obs_vec[self.offset + rank] = 1

                self.offset += self.num_ranks
                num_cards += 1

            if num_cards < self.hand_size:
                self.offset += (self.hand_size - num_cards) * (self.bits_per_card + self.num_colors + self.num_ranks)

        # print(self.offset)
        assert self.offset - (
                    self.hands_bit_length +
                    self.board_bit_length +
                    self.discard_pile_bit_length +
                    self.last_action_bit_length +
                    self.card_knowledge_bit_length) == 0

        return True

class LegalMovesVectorizer(object):
    '''
    // Uid mapping.  h=hand_size, p=num_player_knowledgeplayers, c=colors, r=ranks
    // 0, h-1: discard
    // h, 2h-1: play
    // 2h, 2h+(p-1)c-1: color hint
    // 2h+(p-1)c, 2h+(p-1)c+(p-1)r-1: rank hint
    '''
    def __init__(self, env):
        self.env = env
        self.num_players = self.env.num_players
        self.num_ranks = self.env.num_ranks
        self.num_colors = self.env.num_colors
        self.hand_size = self.env.hand_size
        self.max_reveal_color_moves = (self.num_players - 1) * self.num_colors
        self.num_moves = self.env.max_moves

    def get_legal_moves_as_int(self, legal_moves):
        legal_moves_as_int = [-np.Inf for _ in range(self.num_moves)]
        tmp_legal_moves_as_int = [self.get_move_uid(move) for move in legal_moves]

        for move in tmp_legal_moves_as_int:
            legal_moves_as_int[move] = 0.0

        return [self.get_move_uid(move) for move in legal_moves]

    def get_legal_moves_as_int_formated(self,legal_moves_as_int):

        new_legal_moves = np.full(self.num_moves, -float('inf'))

        if legal_moves_as_int:
            new_legal_moves[legal_moves_as_int] = 0
        return new_legal_moves

    def get_move_uid(self, move):
        if move["action_type"] == "DISCARD":
            card_index = move["card_index"]
            return card_index

        elif move["action_type"] == "PLAY":
            card_index = move["card_index"]
            return self.hand_size + card_index

        elif move["action_type"] == "REVEAL_COLOR":
            target_offset = move["target_offset"]
            color = color_char_to_idx(move["color"])
            return self.hand_size + self.hand_size + (target_offset-1) * self.num_colors + color

        elif move["action_type"] == "REVEAL_RANK":
            rank = move["rank"]
            target_offset = move["target_offset"]
            return self.hand_size + self.hand_size + self.max_reveal_color_moves + (target_offset-1) * self.num_ranks + rank
        else:
            print("\n==================")
            print("MOVE IS NOT POSSIBLE")
            print("===================\n")

            # print(move)
            return -2
