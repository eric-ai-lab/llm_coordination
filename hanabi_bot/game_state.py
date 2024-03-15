from constants import MAX_CLUE_NUM


# This is just a reference. For a fully-fledged bot, the game state would need
# to be more specific. (For example, a card object should contain the positive
# and negative clues that are "on" the card.)
class GameState:
    replaying_past_actions = True
    clue_tokens = MAX_CLUE_NUM
    player_names = []
    our_player_index = -1
    hands = []  # An array containing card objects (dictionaries).
    play_stacks = []
    discard_pile = []
    turn = -1
    current_player_index = -1
