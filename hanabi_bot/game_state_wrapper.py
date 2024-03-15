import ast
from typing import Optional, List, Set, Dict
import copy
import os
import sys
# rel_path = os.path.join(os.environ['PYTHONPATH'])
# sys.path.append(rel_path)
import json_to_pyhanabi, utils
import commands_websocket as cmd


class GameStateWrapper:

    def __init__(self, game_config):
        """
        # ################################################ #
        # -------------------- CONFIG -------------------- #
        # ################################################ #
        """
        self.agent_name = game_config['username']  # used to identify absolute position on table
        self.num_players = game_config['num_total_players']  # number of players ingame
        self.max_life_tokens = game_config['life_tokens']
        self.max_info_tokens = game_config['info_tokens']
        self.max_deck_size = game_config['deck_size']
        self.deck_size = self.max_deck_size
        self.life_tokens = self.max_life_tokens
        self.information_tokens = self.max_info_tokens

        self.players = None  # list of names of players currently ingame
        self.player_position = None  # agents absolute position at table
        self.agents_turn = False  # flag that is True whenever its our turn
        self.hand_size = 4 if self.num_players > 3 else 5  # deal 5 cards when playing with 2 or 3 ppl


        """
        # ################################################ #
        # ------- Observed Cards and Card knowledge ------ #
        # ################################################ #
        """
        """ New cards are prepended, that means agent 1s inital draw looks like [4,3,2,1] """
        # list of all players hands as _seen_ by calling agent [excluding clues]
        self.observed_hands = list()  # is refreshed in self.update() on each notify message

        # list of clues given
        self.clues = list()  # is refreshed in self.update() on each notify message

        # unfortunately, server references clue cards not by index but by an id between 0 and deck size,
        # so we need to store card_numbers to map the card ids to indices
        self.card_numbers = list()

        """
        # ################################################ #
        # ----------------- GAME STATS ------------------- #
        # ################################################ #
        """
        # is refreshed in self.update() on each notify message
        self.fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}

        # list of discarded cards as returned by self.card(suit, rank)
        self.discard_pile = list()

        # actually not contained in the returned dict of the
        # rl_env.HanabiEnvobservation._extract_from_dict()-method, but we need a history so we add this here.
        # Similarly, it could be added by appending obs_dict['last_moves'] = observation.last_moves() in said method.
        self.last_moves = list()
        self.variant = game_config['variant']
        self.num_colors = game_config['colors']
        self.num_ranks = game_config['ranks']
        self.max_moves = game_config['max_moves']
        self.order = 0  # number of the next card on the deck, incremented when card is drawn/dealt
        """
        # ################################################ #
        # -------------- USE PYHANABI MOCKS -------------- #
        # ################################################ #
        """
        
        self.env = json_to_pyhanabi.create_env_mock(
            num_players=self.num_players,
            num_colors=self.num_colors,
            num_ranks=self.num_ranks,
            hand_size=self.hand_size,
            max_information_tokens=self.max_info_tokens,
            max_life_tokens=self.max_life_tokens,
            max_moves=self.max_moves,
            variant=self.variant
        )

        self.caller_is_admin = False  # flag is used to determine one RL agent that keeps track of human players
        # environment observations in order to be able to keep the vectorized observations synchronized
        if self.agent_name == "bot1":
            self.caller_is_admin = True  # admin is the first instance of the client class
        self.idx_human_player = -1  # used to determine whenever human is target of a card hint and thus when

    def __str__(self):
        return f"Current State is: {self.observed_hands}\n{self.fireworks}\n{self.information_tokens}\n{self.life_tokens}\n{self.discard_pile}"


    def reset(self):
        self.observed_hands = list()
        self.clues = list()
        self.card_numbers = list()
        self.fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}
        self.information_tokens = self.max_info_tokens
        self.life_tokens = self.max_life_tokens
        self.discard_pile = list()
        self.last_moves = list()
        self.player_position = None
        self.agents_turn = False
        self.deck_size = self.max_deck_size
        self.order = 0
        return
    
    def init_players_from_game_history(self, game_history):
        self.reset()
        

    def init_players(self, player_dict):
        """ Sets self.players to a list of the players currently ingame and creates empty hands """

        self.reset()
        # player_dict = ast.literal_eval(notify_msg.split('init ')[1].replace(
        #         'false', 'False').replace(
        #         'list', 'List').replace(
        #         'true', 'True'))
        self.players = player_dict['names']
        self.num_players = len(self.players)
        self.observed_hands = [list() for _ in range(self.num_players)]
        self.card_numbers = [list() for _ in range(self.num_players)]
        self.clues = [list() for _ in range(self.num_players)]
        self.initial_draw_counter = 10
        # determine table position of our agent
        self.player_position = self.players.index(self.agent_name)
        """ Following code has been added afterwards in order to cope with the fact that the vectorizer is stateful 
         and therefore needs to be synchronized by being called with the observation of the human players"""
        # determine if there are human players and if not, get their index
        # as of now we hardcode it assuming there are only rainbow agents and simple agents
        for p_name in self.players:
            if ('bot1' not in p_name) and ('RainbowPlayer' not in p_name):  # Assume it is a human player
                self.idx_human_player = self.players.index(p_name)
                break  # assume there is at most one human player
        return
    # TODO: Get deal cards back into the picture 
    def deal_cards(self, card_list):
        """ Initializes self.hand_list from server message 'notifyList [{"type":"draw","who":0,"rank":4,"suit":1,
        "order":0},...'"""

        # list of dictionaries storing the draws
        # card_list = ast.literal_eval(notify_msg.split('notifyList ')[1].replace(
        #     'false', 'False').replace(
        #     'list', 'List').replace(
        #     'true', 'True'))

        for d in card_list:
            if d['type'] == 'draw':  # add card to hand of player with id d['who'] from left to right
                self.draw_card(d)
                # the new card has no clues on it when drawn
                # also it doesnt matter whether we append or insert here, as it is ambigous
                self.clues[d['who']].append({'color': "RYGBW", 'rank': "01234"})

            if d['type'] == 'turn':
                # notifyList message also contains info on who goes first
                if d['who'] == self.player_position:
                    self.agents_turn = True
        return

    def draw_card(self, d):
        """ Adds card to players hand and updates deck size. Then updates card references and clues."""
        # prepend drawn card to players hand, i.e. oldest card has highest index
        self.observed_hands[d['who']].insert(0, self.card(d['suit'], d['rank']))

        # decrease deck size counter
        self.deck_size -= 1
        # increase card count
        self.order += 1

        # unfortunately, server references clued cards by absolute number and not its index on the hand
        # so we store the number too, to map it onto indices for playing and discarding
        self.card_numbers[d['who']].insert(0, d['order'])

    def discard(self, d, true_discard):
        """
        Synchronizes references between handcards and clues.
        Need to reference by card_number, as we cannot access the card by rank and suit, if we discard own card,
        because its values are not known to us at the time of discarding
        """
        # discarding player ID
        pid = d['which']['index']

        # Remove card number reference
        idx_card = self.card_numbers[pid].index(d['which']['order'])
        del self.card_numbers[pid][idx_card]

        # Remove card from hand
        del self.observed_hands[pid][idx_card]

        # Remove card from clues
        del self.clues[pid][idx_card]
        self.clues[pid].insert(0, {'color': "RYGBW", 'rank': "01234"})

        # Update discard pile only if not PLAY move
        if true_discard:
            self.discard_pile.append(self.card(d['which']['suit'], d['which']['rank']))


        return

    def update_state(self, d):
        """
        This is the main event loop of the game

        Updates Game State after server sends an action-notification.
        Notify message can contain 'turn', 'draw', 'play', 'clue', 'discard' as "type"-values
        Computes last_moves including recursive
        """

        'Create dictionary from server message that contains actions'
        # d = ast.literal_eval(
        #     notify_msg.split('notify ')[1].replace(
        #         'false', 'False').replace(
        #         'list', 'List').replace(
        #         'true', 'True'))

        tmp_deepcopy = copy.deepcopy(self.card_numbers)  # safe these for pyhanabi mock objects (i.e. last_moves)
        scored = False
        information_token = False
        deal_to_player = -1
        card_info_revealed = list()
        discarded = False

        # DISCARD
        if d['type'] == 'discard':
            true_discard = False
            # only recover info token when not discarding through failed play
            if 'failed' in d and d['failed'] is False:
                self.information_tokens += 1
                true_discard = True
                # will be used in HanabiHistoryItemMock
                information_token = True
            if 'failed' in d and d['failed']:
                discarded = True
                true_discard = True
                d['type'] = 'play'

            self.discard(d, true_discard=true_discard)

        # DRAW - if player with pid draws a card, it is prepended to hand_list[pid]
        if d['type'] == 'draw':
            self.draw_card(d)
            deal_to_player = d['who']
            if self.initial_draw_counter > 0:
                self.clues[d['who']].insert(0, {'color': "RYGBW", 'rank': "01234"})
                self.initial_draw_counter -= 1
        # PLAY - remove played card from players hand and update fireworks/life tokens
        if d['type'] == 'play':
            # server names some plays "discard" and hence we can come from the discard-if
            if not discarded:
                # remove card
                self.discard(d, true_discard=False)

            # update fireworks and life tokens eventually
            c = self.card(d['which']['suit'], d['which']['rank'])
            scored, information_token = self.play(c)
            if 'failed' not in d and not scored:
                self.discard_pile.append(self.card(d['which']['suit'], d['which']['rank']))
        # CLUE - change players card_knowledge and remove an info-token
        if d['type'] == 'clue':
            card_info_revealed = self.update_clues(d)
            self.information_tokens -= 1  # validity has previously been checked by the server so were good with that

        # Set current player flag
        if d['type'] == 'turn':
            if d['who'] == self.player_position:
                self.agents_turn = True
            else:
                self.agents_turn = False

        # Print move and add it to last_moves()
        if d['type'] in ['play', 'draw', 'clue', 'discard']:
            #  {type: "discard", failed: true, which: {index: 1, suit: 2, rank: 2, order: 8}} is also possible
            self.append_to_last_moves(d, tmp_deepcopy, scored, information_token, deal_to_player, card_info_revealed)

        # On end of game, do something later if necessary (resetting happens on init so no need here)
        if d['type'] == 'turn' and d['who'] == -1:
            pass

        return

    # CHANGE from NONE to RYGBW and add the option to remove colors/ranks from cards which are definitely not that 
    def update_clues(self, dict_clue):

        card_info_revealed = list()
        clue = dict_clue['clue']
        target = dict_clue['target']
        touched_cards = dict_clue['list']
        
        all_cards = [i for i in range(10)]
        for c in touched_cards:
            idx_c = self.card_numbers[target].index(c)
            # reverse order to match with pyhanabi encoding
            max_idx = self.hand_size - 1
            # card_info_revealed.append(idx_c)
            card_info_revealed.append(max_idx - idx_c)
            if clue['type'] == 1:
                old_color_clue = self.clues[target][idx_c]['color']  # keep old clue value
                new_rank_clue = self.card(old_color_clue, clue['value'])['rank']  # pass rank to card() function for conversion
                # update old+new clue
                self.clues[target][idx_c] = {'color': old_color_clue, 'rank': str(new_rank_clue)}
            else:
                old_rank_clue = self.clues[target][idx_c]['rank']  # keep old clue value
                clued_card_color = self.card(clue['value'], -1)['color']  # pass color to card() function for conversion
                # update old+new clue
                self.clues[target][idx_c] = {'color': clued_card_color, 'rank': old_rank_clue}
        
        # Everytime a touched card is clues with a color/rank, it also implies that the untouched cards are *not* that color/rank
        not_touched_cards = [c for c in self.card_numbers[target] if c not in touched_cards]
        for c in not_touched_cards:
            idx_c = self.card_numbers[target].index(c)
            # reverse order to match with pyhanabi encoding
            max_idx = self.hand_size - 1
            if clue['type'] == 1:
                old_color_unclue = self.clues[target][idx_c]['color']
                old_rank_unclue = self.clues[target][idx_c]['rank']
                new_rank_unclue = str(self.card(old_color_unclue, clue['value'])['rank'])
                self.clues[target][idx_c] = {'color': old_color_unclue, 'rank': old_rank_unclue.replace(new_rank_unclue, '')}
            
            else:
                old_color_unclue = self.clues[target][idx_c]['color']
                old_rank_unclue = self.clues[target][idx_c]['rank']
                new_color_unclue = str(self.card(clue['value'], -1)['color'])
                self.clues[target][idx_c] = {'color': old_color_unclue.replace(new_color_unclue, ''), 'rank': old_rank_unclue}




        return card_info_revealed

    def play(self, card):
        scored = False
        information_token = False
        # on success, update fireworks
        if self.fireworks[card['color']] == card['rank']:
            self.fireworks[card['color']] += 1
            # completing a firework restores one info token
            if card['rank'] == 4:
                self.information_tokens += 1
                information_token = True
            # will be used in HanabiHistoryItemMock
            scored = True
        # on fail, remove a life token
        else:
            self.life_tokens -= 1
        return scored, information_token

    # scored, information_token are bool
    def append_to_last_moves(self, dict_action, deepcopy_card_nums, scored, information_token, deal_to_player, card_info_revealed):
        """
        Mocks HanabiHistoryItems as gotten from pyhanabi. As these objects provide callables, we have to create these
        here.
        Input dict_action looks like
        ############   DRAW   ##############
        {"type":"draw","who":1,"rank":-1,"suit":-1,"order":11}
        ############   CLUE   ##############
        {"type":"clue","clue":{"type":0,"value":3},"giver":0,"list":[5,8,9],"target":1,"turn":0}
        ############   PLAY   ##############
        {"type":"play","which":{"index":1,"suit":1,"rank":1,"order":11}}
        ############   DISCARD   ##############
        {"type":"discard","failed":false,"which":{"index":1,"suit":0,"rank":4,"order":7}}
        """

        move = json_to_pyhanabi.get_pyhanabi_move_mock(dict_action, deepcopy_card_nums, self.num_players)

        def get_player(dict_action):
            player = None
            type = dict_action['type']
            if 'clue' in type:
                player = dict_action['giver']
            elif 'play' in type or 'discard' in type:
                player = dict_action['index']
            else:
                player = -1
            return player

        player = 3  # get_player(dict_action)
        scored = scored  # boolean, True if firework increased
        information_token = information_token  # boolean, True if info_token gained on discard or play
        card_info_revealed = card_info_revealed
        deal_to_player = deal_to_player

        history_item_mock = json_to_pyhanabi.HanabiHistoryItemMock(
            move=move,
            player=player,
            scored=scored,
            information_token=information_token,
            color=None,
            rank=None,
            card_info_revealed=card_info_revealed,
            card_info_newly_revealed=None,
            deal_to_player=deal_to_player
        )

        self.last_moves.insert(0, history_item_mock)
        return

    def get_observed_hands(self) -> List:
        """ Converts internal hand_list to pyhanabi observed_hands. This includes
          - Shifting players observed_hands list, s.t. calling agent sits at index 0
           i.e. [[hands_pid_2], [hands_pid_0], [hands_pid_3]] becomes [[hands_pid_3], [hands_pid_2], [hands_pid_0]]
           when agent with pid 3 is calling this function
          - Reversing each hand from ([newest,..., oldest] to [oldest,...,newest])
         """
        # moves self.cur_player hand to the front
        hand_list = copy.deepcopy(self.observed_hands)
        # shift observed_hands, s.t. calling agent sits at index 0
        n = self.player_position
        hand_list = hand_list[n:] + hand_list[:n]
        # return each hand reversed, s.t. it matches the pyhanabi format
        return [list(reversed(hand)) for hand in hand_list]

    def get_card_knowledge(self):
        """ Converts internal clue_list to pyhanabi card_knowledge. This includes
          - Shifting players card_knowledge list, s.t. calling agent sits at index 0
           i.e. [[hands_pid_2], [hands_pid_0], [hands_pid_3]] becomes [[hands_pid_3], [hands_pid_2], [hands_pid_0]]
           when agent with pid 3 is calling this function
          - Reversing each hand from ([newest,..., oldest] to [oldest,...,newest])
         """
        card_knowledge = copy.deepcopy(self.clues)

        # sort, s.t. agents cards are at index 0
        n = self.player_position
        card_knowledge = card_knowledge[n:] + card_knowledge[:n]
        # return each hand reversed, s.t. it matches the pyhanabi format

        return [list(reversed(clues)) for clues in card_knowledge]
    def get_agent_observation(self):
        """ Returns state as perceived by the calling agent """

        observation = {
            'current_player': self.players.index(self.agent_name),
            'current_player_offset': 0,
            'life_tokens': self.life_tokens,
            'information_tokens': self.information_tokens,
            'num_players': self.num_players,
            'deck_size': self.deck_size,
            'fireworks': self.fireworks,
            'legal_moves': self.get_legal_moves(),
            'observed_hands': self.get_observed_hands(),  # moves own hand to front
            'discard_pile': self.discard_pile,
            'card_knowledge': self.get_card_knowledge(),
            'last_moves': self.last_moves  # actually not contained in the returned dict of the
            # rl_env.HanabiEnvobservation._extract_from_dict method, but we need a history so we add this here.
            # Similarly, it can be added by appending obs_dict['last_moves'] = observation.last_moves() in said method.
        }

        return observation

    @staticmethod
    def card(suit: int, rank: int):
        # """ Returns card format desired by agent. Rank values of None and -1 will be passed through."""

        if rank is not None:
            if rank > -1:  # return rank = -1 for an own unclued card
                rank -= 1  # server cards are not 0-indexed
        return {'color': utils.convert_suit(suit), 'rank': rank}

    def get_legal_moves(self):
        """ Computes observation['legal_moves'] or observation.legal_moves(), depending on use_pyhanabi_mock"""
        # order is 2. discard 1. play 3. reveal_color reveal rank and RYGBW for color
        legal_moves = []


        # play
        for i in range(self.hand_size):
            legal_moves.append({'action_type': 'PLAY', 'card_index': i})

        # discard if possible
        if self.information_tokens < self.max_info_tokens:
            for i in range(self.hand_size):
                legal_moves.append({'action_type': 'DISCARD', 'card_index': i})

        # clue if info token available
        if self.information_tokens > 0:
            hand_list = self.get_observed_hands()

            # append colors
            for i in range(1, self.num_players):

                colors = set()
                for card in hand_list[i]:
                    colors.add(card['color'])

                colors = utils.sort_colors(colors)
                for c in colors:
                    legal_moves.append({'action_type': 'REVEAL_COLOR', 'target_offset': i, 'color': c})

            # append ranks
            for i in range(1, self.num_players):
                ranks = set()
                for card in hand_list[i]:
                    ranks.add(card['rank'])
                for r in ranks:
                    legal_moves.append({'action_type': 'REVEAL_RANK', 'target_offset': i, 'rank': r+1})

        return legal_moves

    def parse_action_to_msg(self, action: dict) -> str:
        """ Takes an action dictionary as gotten from pyhanabi
        converts it to action string for GUI server """
        # one of ['REVEAL_COLOR', 'REVEAL_RANK', 'PLAY', 'DISCARD']

        assert isinstance(action, dict)  # make sure we did not get an int action
        assert 'action_type' in action  # make sure we have an action dictionary
        # if one of the above assertions failed, it is probably due to the agent sending an
        # integer version of the acction. It should be parsed to a dictionary by means of the
        # environment in which the agent operates on, so we will not do it here (for now)

        abs_card_nums = copy.deepcopy(self.card_numbers)
        agent_pos = self.player_position
        num_players = self.num_players
        hand_size = self.hand_size

        # TODO: fix this 
        return cmd.get_server_msg_for_pyhanabi_action(
            action=action,
            abs_card_nums=abs_card_nums,
            agent_pos=agent_pos,
            num_players=num_players,
            hand_size=hand_size
        )
