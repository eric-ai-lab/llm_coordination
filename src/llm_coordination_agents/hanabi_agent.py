from __future__ import print_function

import numpy as np
import openai 
import re 
import os
from openai import OpenAI, AzureOpenAI
import itertools
import pandas as pd 
import datetime 
from fuzzywuzzy import process

def add_to_dict_list(dictionary, key, item):
    if key not in dictionary:
        dictionary[key] = [item]
    else:
        dictionary[key].append(item)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class LLMAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.player_names = ['Alice', 'Bob']
        self.partner_pronoun = 'his' if self.player_id == 0 else 'her'
        # self.model = 'gpt-4-0613'
        # self.model= 'gpt-4-1106-preview'	
        # self.model = 'gpt-4-turbo-preview'
        # self.model = 'gpt-3.5-turbo-1106'
        # self.model = 'gpt-35-turbo'
        # self.model_type = 'openai'
        self.model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        self.model_type = 'mistral'
        if self.model_type == 'openai':
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version="2023-05-15"
            )
        else:
            self.client = OpenAI(
                    api_key="EMPTY",
                    base_url="http://localhost:8000/v1",
                )
        self.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.action_regex = r"Action:\s*(.*)"
        self.color_map = {
            0: 'R',
            1: 'Y',
            2: 'G',
            3: 'W',
            4: 'B'
        }
        self.llm_system_prompt = "You are a helpful assistant."
        self.rules = '''
        - The game uses a 50-card deck, divided into five colours (red (R), green (G), blue (B), yellow (Y), white (W)). Each color has cards of ranks 1 to 5. Each color has with three 1's, two 2's, two 3's, two 4's, one 5.
        - Players have to create stacks of each color. Each color stack starts with a Rank 1 card and goes up one by one in ascending order up to Rank 5.  (e.g. Red Stack should go from R1 -> R2 -> R3 -> R4 -> R5). A card can only be played if it is the next in the incremental sequence for its color stack.
        - Players can only see the other's hand, not their own.
        - Players have plausible knowledge of their cards based on previously provided hints by the other player
        - They can either play a card, give a reveal, or discard a card.
        ***Actions:***
                1. Reveal (Clue): Spend a reveal token to reveal cards with a particular color or rank. Revealing a color reveals all cards of that color in partner's hand. Revealing a rank reveals all cards with that rank in partner's hand. The game starts with 8 reveal tokens. If no token left, no more reveals can be given. 
                2. Discard: Discard a card to regain a reveal token and draw a new card. 
                3. Play a Card: If a card played follows sequence in its color stack, it succeeds. Success of rank 5 card in any stack gives an additional reveal token. Failure discards the card, and loses a life. Playing a card you are unsure about is risky as it costs a life and you have only 3 lives. Before playing a card make sure that it's the next card in the sequence for that stack.
        ***The game ends when:***
                - All five stacks are completed. 25 Points. 
                - Three lives have been lost. 0 Points no matter how many cards have been placed in the stack. 
                - After the last card from the deck is drawn and each player has had a final turn. Sum total of the top card ranks of each color stack. 
        '''
        self.conventions_odd = '''
        1. **Card Layout:**
            - Cards are added to the right; the oldest card is on the left.
            - Positions are referenced from left to right.
        2. **Clues:**
            - Two types of clues: Play Clue (play the card) and Save Clue (save for later).
            - If a Play Clue or Save Clue can't be given, players must discard.
        3. **Play Clue:**
            - A play clue is revealing a card or cards in partners hand that are immediately playable on the stack by indicating their rank or color.
        4. **Save Clue**
            - A save clue is used to save rank 5 cards, unique rank 2 cards and critical cards (only one of the kind left) 
        5. **Do Not Repeat Known Information**
            - If a player already knows the color of their card, do not repeat the color in a clue. If a player already knows the rank of their card, do not repeat the rank in a clue.
        5. **Prioritize Play Clues over Save Clues:**
            - Prefer giving Play Clues if both are viable options.
        6. **Discard Without Fear:**
            - Discard confidently, as saving important cards is a team responsibility.
        7. **Play with Fear:**
            - You should take risks and play a card even though you are not completely sure when you have 2 or 3 lives left. However when you have only 1 life left you should play a card only when you are sure that is goes next on the stack. '''
        
        self.conventions = '''
        1. **Card Layout:**
            - Cards are added to the right; the oldest card is on the left.
            - Positions are referenced from left to right.
        2. **Clues:**
            - Two types of clues: Play Clue (play the card) and Save Clue (save for later).
            - If a Play Clue or Save Clue can't be given, players must discard.
        3. **Play Clue:**
            - A play clue is revealing a card or cards in partners hand that are immediately playable on the stack by indicating their rank or color.
        4. **Save Clue**
            - A save clue is used to save rank 5 cards, unique rank 2 cards and critical cards (only one of the kind left) 
        5. **Prioritize Play Clues over Save Clues:**
            - Prefer giving Play Clues if both are viable options.
        6. **Discard Without Fear:**
            - Discard confidently, as saving important cards is a team responsibility.
        7. **Play with Fear:**
            - You can take risks and play a card even though you are not completely sure when you have 2 or 3 lives left. However when you have only 1 life left you should play a card only when you are sure that is goes next on the stack. '''
        
        self.h_list_conventions = '''
        1. **Card Layout:**
            - Cards are added to the right; the oldest card is on the left.
            - Positions are referenced from left to right.
        2. **Chop Card:**
            - The left-most unclued card is the "chop" card. An unclued card is one whose color/rank can be anything (RYGWB12345). 
            - The default discard action is to discard the chop card.
        3. **Clues:**
            - Two types of clues: Play Clue (play the card) and Save Clue (save for later).
            - If a valid Play Clue or Save Clue can't be given, players must discard.
        4. **Play Clue:**
            - Indicates a card is playable.
            - Can be delayed if it's playable after other cards are played.
        5. **Save Clue:**
            - Indicates to save a card, typically given to chop cards.
            - Focus on saving 5's, unique 2's, and critical cards.
            - Number clues (2 or 5) are used for saving 2's and 5's.
            - Critical cards are the last unplayed card of a suit and value.
        7. **Clue Interpretation Algorithm:**
            - When a clue happens and you need to figure out what it means, always ask yourself the following questions: 1. What card is focused? 2. Is it a Play Clue or a Save Clue or could it still be either? - Clues that do not concern chop cards are always Play Clues. - Chop clues can be either a Play Clue or a Save Clue. 3. What is the identity of the card?
        8. **Good Touch Principle:**
            - Only clue cards that will eventually be played.
        9. **Save Principle:**
            - Prevent discarding of critical and unique playable cards.
            - Most important principle, can override others if necessary.
        10. **Minimum Clue Value Principle:**
            - Restricts clue giving to Play Clues for playable cards and Save Clues for critical cards.
        11. **Early Game:**
            - Focus on using all available Play and Save Clues before the first discard.
        12. **Check Team Chops:**
            - Prioritize checking your partner’s chop cards at the start of your turn.
        13. **Prioritize Play Clues over Save Clues:**
            - Prefer giving Play Clues if both are viable options.
        14. **Discard Without Fear:**
            - Discard confidently, as saving important cards is a team responsibility.
        15. **Play with Fear:**
            - Do not play a card unless you know for sure that it goes next on the stack when you only have 1 life token left. '''

        self.base_prompt = f'''The card game Hanabi has the following rules:
        {self.rules}
        I am {self.player_names[self.player_id]}, playing the card game Hanabi with {self.player_names[1 - self.player_id]}. 
        At each time step I will provide you with the relevant information of the game. I will also provide you with the legal action, help me select the best next action. Remember I am playing as {self.player_names[self.player_id]}. Format your response as Explanation: <brief explanation for selecting the move>\nAction:<selected move>. Do not say anything else. Got it?'''
        
        ###
        self.verifier_base_prompt = f'''The card game Hanabi has the following rules:
        {self.rules}          
        I am {self.player_names[self.player_id]}, playing the card game Hanabi with {self.player_names[1-self.player_id]}.'''
        
        # self.verifier_base_prompt = f'''The card game Hanabi has the following rules:
        # {self.rules}          
        # I am {self.player_names[self.player_id]}, playing the card game Hanabi with {self.player_names[1-self.player_id]}. We have agreed to follow these conventions: 
        # {self.conventions}.
        # You are an action verification agent for games. I will provide you with an action and you need to check whether the action satisfies the criteria: 1. Rule Following: It follows to the rules of the game. 2. Convention Following: It adheres to the mentioned conventions. 3. Safety: It won't lead to the game ending immediately. Think about the action, the current state of the stack and the available lives and reveal tokens. End you response with "Verification: Okay" if selected action follows ***all three*** criteria and "Verification: Not Okay" otherwise. Restrict your response to 4-5 sentences.'''
        ###
        
        ###
        # self.epistemologist_base_prompt = f'''The card game Hanabi has the following rules:
        # {self.rules}
        # I am {self.player_names[self.player_id]}, playing the card game Hanabi with {self.player_names[1-self.player_id]}. We have agreed to follow these conventions: {self.conventions}.
        # You will assist me with understanding my partner.  You will be provided with the rules of a game, conventions, my partner's selected action and my latest state information after my partner took their action. You will respond with a summary of my partner’s beliefs, intentions, and optional implicit communications based on the action they took and the resulting updated state. You will also summarize your partner's knowledge to suggest what new information they need to play a card. Restrict your response to 4-5 sentences.'''
        # self.epistemologist_base_prompt = f'''The card game Hanabi has the following rules:
        # # {self.rules}
        # # I am {self.player_names[self.player_id]}, playing the card game Hanabi with {self.player_names[1-self.player_id]}. We have agreed to follow these conventions: {self.conventions}.'''
        
        self.epistemologist_base_prompt = f'''The card game Hanabi has the following rules:
        {self.rules}
        I am {self.player_names[self.player_id]}, playing the card game Hanabi with {self.player_names[1-self.player_id]}. 
        You are a Theory of Mind inference agent for our game. You will be provided with my partner's selected action and my latest state information after my partner took their action. You will provide me with two things: 1.  An explanation for my partner’s previous action along with their intention and implicit communication. 2. What is the best information for me to give my partner based on their knowledge? 
        Format your response as:
        Partner Action Explanation:<1 sentence explanation of partner action>
        Clue Suggestion:<What information (specify rank or color) should I reveal to my partner based on their knowledge>.
        '''
        ##
        self.verifier_system_prompt = '''You are an action verification agent for games. I will provide you with an action and you need to check whether the action satisfies the criteria: 1. Rule Following: It follows to the rules of the game. 2. Safety: It won't lead to the game ending immediately. Think about the action, the current state of the stack and the available lives and reveal tokens. End you response with "Verification: Okay" if selected action follows ***both*** criteria and "Verification: Not Okay" otherwise. Restrict your response to 4-5 sentences.'''

        self.assistant_response_initial = f'''Got it!'''

        if self.model_type == 'openai':
            self.base_message = [
                        {"role": "system", "content": self.llm_system_prompt},
                        {"role": "user", "content": self.base_prompt},
                        {"role": "assistant", "content": self.assistant_response_initial},
                    ]
            self.verifier_base_message = [
                {"role": "system", "content": self.verifier_system_prompt}, 
                {"role": "user", "content": self.verifier_base_prompt},
                {"role": "assistant", "content": self.assistant_response_initial},
            ]
            self.epistemologist_message = [
                # {"role": "system", "content": self.epistemologist_system_prompt}, 
                {"role": "user", "content": self.epistemologist_base_prompt},
                {"role": "assistant", "content": self.assistant_response_initial},
            ]
        else:
            self.base_message = [
                        {"role": "user", "content": self.llm_system_prompt + self.base_prompt},
                        {"role": "assistant", "content": self.assistant_response_initial},
                    ]
            self.verifier_base_message = [
                {"role": "user", "content": self.verifier_system_prompt + self.verifier_base_prompt},
                {"role": "assistant", "content": self.assistant_response_initial},
            ]
            self.epistemologist_message = [
                # {"role": "system", "content": self.epistemologist_system_prompt}, 
                {"role": "user", "content": self.epistemologist_base_prompt},
                {"role": "assistant", "content": self.assistant_response_initial},
            ]
            
        self.c_map = {"R": "Red", "Y": "Yellow", "G": "Green", "W": "White", "B": "Blue"}
        self.working_memory = {}
        self.prev_working_memory = {}
        # self.working_memory_dict = {}
        self.action_history = []
        self.log_csv_dict = {} 
        self.log_dir = f'/home/saaket/llm_coordination/logs/hanabi'
        self.traj_dir = f'/home/saaket/llm_coordination/logs/hanabi'
        self.my_card_uncertainty = []
        self.partner_card_uncertainty = []
        self.prev_state_description = ''
        for i in range(5):
            self.my_card_uncertainty.append(0)
            self.partner_card_uncertainty.append(0)
        # self.player_actions = list(np.load(f'{self.traj_dir}/{self.player_names[self.player_id]}_2023-12-23_03-07-12.npy'))
        self.player_actions = []
        self.partner_action_inference_string = ''
        
    def _get_card_knowledge(self, observation):
        description = f"My cards based on my knowledge: \n"
        my_cards = [] 
        c_map = {"R": "Red", "Y": "Yellow", "G": "Green", "W": "White", "B": "Blue"}

        for i, card in enumerate(observation.card_knowledge()[0]):
            plausible_colors = ['R', 'Y', 'G', 'W', 'B']
            plausible_ranks = [1,2,3,4,5]
            card_string = card.__str__()

            plausible_cards = card_string.split("|")[-1]

            plausible_colors = [pc for pc in plausible_colors if str(pc) in plausible_cards]

            plausible_ranks = [pc for pc in plausible_ranks if str(pc) in plausible_cards]
            

            sure_cards = card_string.split("|")[0]
            if sure_cards[0] in plausible_colors:
                plausible_colors = [sure_cards[0]]
            if sure_cards[1] in plausible_ranks:
                plausible_ranks = [sure_cards[1]]            
            # card_combinations = [''.join(map(str, combo)) for combo in itertools.product(plausible_colors, plausible_ranks)]
            # self.my_card_uncertainty[i] = len(card_combinations)
            # if len(card_combinations) == 1:
            #     description += f"Card {i} is: {card_combinations}\n"
            # else:
            #     description += f"Card {i} could be: {card_combinations}\n"
            description += f"Card {i} could be: ["
            for color in plausible_colors:
                description += f"{c_map[color]}, "
            description = description[:-2]
            description += '] ['
            
            
            for rank in plausible_ranks:
                description += f"{rank}, "
            description = description[:-2]
            description += ']\n'
            description = description.replace("'", "")
        add_to_dict_list(self.log_csv_dict, 'My Card Knowledge', description)

        return description

    def _get_partner_cards(self, observation):
        description = f"I can see {self.player_names[1-self.player_id]}'s Cards are: \n"
        c_map = {"R": "Red", "Y": "Yellow", "G": "Green", "W": "White", "B": "Blue"}
        hands = observation.observed_hands()
        partner_cards = []
        for i, card in enumerate(hands[-1]):
            card_color = self.color_map[card._color]
            card_rank = card._rank + 1
            description += f"[Card {i}: {c_map[card_color]} {card_rank}]\n"
            # description += f"Rank: {card_rank}>\n"
            # partner_cards.append(f"{card_color}{card_rank}")
        # description += str(partner_cards)
        add_to_dict_list(self.log_csv_dict, 'Partner Cards', description)
        description = description.replace("'", "")
        return description
    
    # Delineating epistemic knowledge 
    def _infer_partner_knowledge(self, observation):
        description = f"{self.player_names[1-self.player_id]}'s Knowledge about {self.partner_pronoun} cards: \n"
        c_map = {"R": "Red", "Y": "Yellow", "G": "Green", "W": "White", "B": "Blue"}
        for i, card in enumerate(observation.card_knowledge()[1]):
            plausible_colors = ['R', 'Y', 'G', 'W', 'B']
            plausible_ranks = [1,2,3,4,5]
            card_string = card.__str__()

            plausible_cards = card_string.split("|")[-1]

            plausible_colors = [pc for pc in plausible_colors if str(pc) in plausible_cards]

            plausible_ranks = [pc for pc in plausible_ranks if str(pc) in plausible_cards]
            

            sure_cards = card_string.split("|")[0]
            if sure_cards[0] in plausible_colors:
                plausible_colors = [sure_cards[0]]
            if sure_cards[1] in plausible_ranks:
                plausible_ranks = [sure_cards[1]]            
            # card_combinations = [''.join(map(str, combo)) for combo in itertools.product(plausible_colors, plausible_ranks)]
            # self.partner_card_uncertainty[i] = len(card_combinations)
            # if len(card_combinations) == 1:
            #     description += f"{self.player_names[1-self.player_id]} knows {self.partner_pronoun} Card {i} is: {card_combinations}\n"
            # else:
            #     description += f"{self.player_names[1-self.player_id]} believes {self.partner_pronoun} Card {i} could be:  {card_combinations}\n"
            description += f"{self.player_names[1-self.player_id]} believes {self.partner_pronoun} Card {i} could be: ["
            for color in plausible_colors:
                description += f"{c_map[color]}, "
            description = description[:-2]
            description += '] ['
            for rank in plausible_ranks:
                description += f"{rank}, "
            description = description[:-2]
            description += ']\n'
            description = description.replace("'", "")
        add_to_dict_list(self.log_csv_dict, 'Partner Card Knowledge', description)

        return description

    def _get_legal_moves(self, observation):
        self.transformed = []
        moves = observation.legal_moves()
        cmap = {"R": "Red", "Y": "Yellow", "G": "Green", "W": "White", "B": "Blue"} 
        for enum, move in enumerate(moves):
            move = move.__str__()
            move = move.replace('(', '').replace(')', '')
            parts = move.split(' ')
            action = parts[0]
            ord = chr(enum + 65)

            if action == 'Discard' or action == 'Play':
                self.transformed.append(f"{ord}. {action} My Card {parts[1]}.")
            elif action == 'Reveal':
                player = parts[2]
                attribute = parts[3]
                value = parts[4]
                if attribute == 'color':
                    self.transformed.append(f"{ord}. Reveal {self.player_names[1-self.player_id]}'s {cmap[value]} color cards.")
                elif attribute == 'rank':
                    self.transformed.append(f"{ord}. Reveal {self.player_names[1-self.player_id]}'s rank {value} cards.")
        description = 'Available Legal Actions: \n'
        for tm in self.transformed:
            description += tm
            description += '\n'
        add_to_dict_list(self.log_csv_dict, 'Available Actions', description.replace('#', ''))
        return description

    def _get_current_stack(self, observation):
        description = 'Current Stacks: '
        colors = 'RYGWB'
        color_names = ['Red', 'Yellow', 'Green', 'White', 'Blue']
        stack_state = []
        for i, firework in enumerate(observation.fireworks()):
            # stack_state.append(f'{colors[i]}{firework}')
            description += f"{color_names[i]} - {color_names[i]} {firework} "
        # description += str(stack_state)
        description += '\n'
        add_to_dict_list(self.log_csv_dict, 'Stack', str(stack_state))
        return description
    
    def _add_soft_constraints(self, observation):
        description = f'The next cards that can be played on each stack are: \n'
        colors = 'RYGWB'
        color_names = ['Red', 'Yellow', 'Green', 'White', 'Blue']
        stack_state = []
        for i, firework in enumerate(observation.fireworks()):
            # stack_state.append(f'{colors[i]}{firework+1}')
            if firework != 5:
                description += f"Only {color_names[i]} {firework+1} can be played on {color_names[i]} stack \n"
            else:
                description += f"{color_names[i]} stack is complete. "
        # description += str(stack_state)
        add_to_dict_list(self.log_csv_dict, 'Soft Constraints', description)
        description += '\n'
        return description


    def _get_previous_selected_actions(self): 
        if len(self.action_history)>0:
            add_to_dict_list(self.log_csv_dict, 'Prev Actions', f'My Action History: {", ".join([ac for ac in self.action_history])}')
            return f'My Action History: {", ".join([ac for ac in self.action_history])}\n'
        else:
            add_to_dict_list(self.log_csv_dict, 'Prev Actions', f'My Action History: ')
            return ''
        
    def _clarify_chop_card(self):    
        description = ''
        # My Chop Card: 
        my_chop_card = len(self.my_card_uncertainty)-1
        my_chop_card_value = self.my_card_uncertainty[-1]
        for i in range(len(self.my_card_uncertainty)-1, -1, -1):
            if self.my_card_uncertainty[i] > my_chop_card_value:
                my_chop_card_value = self.my_card_uncertainty[i]
                my_chop_card = i
        
        partner_chop_card = len(self.partner_card_uncertainty) - 1
        partner_chop_card_value = self.partner_card_uncertainty[-1]
        for i in range(len(self.partner_card_uncertainty) - 1, -1, -1):
            if self.partner_card_uncertainty[i] > partner_chop_card_value:
                partner_chop_card_value = self.partner_card_uncertainty[i]
                partner_chop_card = i
        description += f'My Chop Card is: Card {my_chop_card}.\n'
        description += f"{self.player_names[1-self.player_id]}'s Chop Card is: Card {partner_chop_card}.\n"
        return description
    
    ## TODO: Complete this 
    # def _infer_partner_action(self, partner_action, description):
    #     partner_action_inference_description = ''
    #     if partner_action is not None:
    #         partner_move_string = self.convert_pyhanabi_move(partner_action)
    #         epistemic_message = self.epistemologist_message + [{"role": "user", "content": f"***{self.player_names[1-self.player_id]}'s selected action***: {partner_move_string}\n\nMy current state information: {description}. Note that I have updated my knowledge of my cards based on partner's action. Think step by step about partners action. Think about the action. Think about what it implies. If I should give a clue next, think about what clue I can give my partner."}]
    #         epistemic_response_string = self.llm_inference(epistemic_message)
    #         partner_action_inference_description = f"Interpretation of {self.player_names[self.player_id-1]}'s Last Action: {epistemic_response_string}. \n"
    #         self.partner_action_inference_string = partner_action_inference_description
    #         add_to_dict_list(self.log_csv_dict, 'Epistemic Information', partner_action_inference_description)
    #     return partner_action_inference_description
    def _infer_partner_action(self, episodic_memory, working_memory, description):
        partner_action_inference_description = ''
        if len(episodic_memory[1-self.player_id])>0:
            # Extract partner action and the state they used to make their decision 
            partners_most_recent_action = episodic_memory[1-self.player_id][-1]
            partners_most_recent_working_memory = working_memory[1-self.player_id] # Only take the part of working memory that was observable to both agents
            
            # description = self.working_memory['turn']
            # description += partners_most_recent_working_memory['stack']
            # if self.prev_working_memory != {}:
            #     description += self.prev_working_memory['card_knowledge']
            #     description += self.prev_working_memory['partner_cards']
            # else:
            #     description += self.working_memory['card_knowledge']
            #     description += self.working_memory['partner_cards']

            # description += f'{self.player_names[1-self.player_id]} had the following beliefs about {self.partner_pronoun} cards before taking the action: ' 
            # description += partners_most_recent_working_memory['card_knowledge'].replace('My cards based on my knowledge: ', '')

            # description += partners_most_recent_working_memory['reveal_tokens']
            # description += partners_most_recent_working_memory['lives']            
            # # description += self.working_memory['deck_size']
            # description += partners_most_recent_working_memory['discard_pile']
            
            # # Construct message for partner interpreter 
            # epistemic_message = self.epistemologist_message + [{"role": "user", "content": f"***{self.player_names[1-self.player_id]}'s selected action***: {partners_most_recent_action}\n\nState information used to take this decision: {description}. Think step by step. Think about the action. Think about what it implies. If I should give a clue next, think about what clue I can give."}]
            epistemic_message = self.epistemologist_message + [{"role": "user", "content": f"***{self.player_names[1-self.player_id]}'s selected action***: {partners_most_recent_action}\n\nMy current state information: {description}. Note that I have updated my knowledge of my cards based on partner's action. Think step by step about partners action. Think about the action. Think about what it implies. If I should give a clue next, think about what clue I can give my partner."}]
            
            # print(f'''{bcolors.OKGREEN}EPISTEMIC INPUT: {f"***{self.player_names[1-self.player_id]}'s selected action***: {partners_most_recent_action}\n\nMy current state information: {description}. Note that I have updated my knowledge of my cards based on partner's action. Think step by step about partners action. Think about the action. Think about what it implies. If I should give a clue next, think about what clue I can give my partner."}{bcolors.ENDC}''')
            print(f'''{bcolors.OKGREEN}EPISTEMIC INPUT: ***{self.player_names[1-self.player_id]}'s selected action***: {partners_most_recent_action}\n\nMy current state information: {description}. Note that I have updated my knowledge of my cards based on partner's action. Think step by step about partners action. Think about the action. Think about what it implies. If I should give a clue next, think about what clue I can give my partner.{bcolors.ENDC}''')
            
            epistemic_response_string = self.llm_inference(epistemic_message)
            

            partner_action_inference_description = f"Interpretation of {self.player_names[self.player_id-1]}'s Last Action: {epistemic_response_string}. \n You can use the clue suggestion if giving a hint (reveal) is the next best possible move and ignore it otherwise."
            
            print(f'''{bcolors.OKGREEN}EPISTEMIC INFERENCE: {partner_action_inference_description}{bcolors.ENDC}''')
            self.partner_action_inference_string = partner_action_inference_description
            
        add_to_dict_list(self.log_csv_dict, 'Epistemic Information', partner_action_inference_description)
        return partner_action_inference_description
        

    def _observation_to_description(self, observation, episodic_memory, working_memory_hub):
        self.working_memory['turn'] = f'It is currently My ({self.player_names[self.player_id]}) turn. '
        description = self.working_memory['turn']

        self.working_memory['stack'] = self._get_current_stack(observation)
        description += self.working_memory['stack']
        
        self.working_memory['card_knowledge'] = self._get_card_knowledge(observation)
        description += self.working_memory['card_knowledge']

        self.working_memory['partner_cards'] = self._get_partner_cards(observation)
        description += self.working_memory['partner_cards']

        self.working_memory['partner_card_knowledge'] = self._infer_partner_knowledge(observation)
        description += self.working_memory['partner_card_knowledge']

        self.working_memory['reveal_tokens'] = f"Remaining Reveal Tokens: {observation.information_tokens()}\n"
        description += self.working_memory['reveal_tokens']

        self.working_memory['lives'] = f"Remaining Lives: {observation.life_tokens()}\n"
        description += self.working_memory['lives']

        self.working_memory['deck_size'] = f"Deck Size: {observation.deck_size()}\n"
        description += self.working_memory['deck_size']

        self.working_memory['discard_pile'] = f"The discard pile is: {observation.discard_pile()}\n"
        description += self.working_memory['discard_pile']

        # description += f'\nInformation: We have {observation.information_tokens()} reveal tokens, {observation.life_tokens()} life tokens. The discard pile consists {observation.discard_pile()} and the deck size is {observation.deck_size()}.\n'
        add_to_dict_list(self.log_csv_dict, 'Board Information', f"Information: We have {observation.information_tokens()} reveal tokens, {observation.life_tokens()} life tokens. The discard pile consists {observation.discard_pile()} and the deck size is {observation.deck_size()}.")
        
        
        self.working_memory['previous_selected_actions'] = self._get_previous_selected_actions()
        description += self.working_memory['previous_selected_actions']
        
        self.working_memory['soft_constraints'] = self._add_soft_constraints(observation)
        description += self.working_memory['soft_constraints']

        
        
        self.working_memory['legal_moves'] = self._get_legal_moves(observation)
        

        temp_description = description + self.working_memory['legal_moves']

        self.working_memory['partner_interpretation'] = self._infer_partner_action(episodic_memory, working_memory_hub,temp_description)

        description += self.working_memory['partner_interpretation']
        description += self.working_memory['legal_moves']
        
        return description
        
    def llm_inference(self, message):
        response = self.client.chat.completions.create(
                messages=message,
                model=self.model,
                temperature=0.6,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
        response_string = response.choices[0].message.content
        return response_string

    def find_best_match(self, action_string):
        match = re.search(self.action_regex, action_string.strip())
        if match:
            selected_match = match.group(1).strip().lower()

            # Sometimes model repeats Action: withing the action string
            if 'action:' in selected_match.lower():
                updated_action_regex = r"action:\s*(.*)"
                match = re.search(updated_action_regex, selected_match.strip())
                if match:
                    selected_match = match.group(1).strip().lower()
            ####
                
            selected_move, score = process.extractOne(selected_match, self.transformed)
        else:
            selected_move = np.random.choice(self.transformed)
        return selected_move

    
    def get_next_move(self, observation, episodic_memory, working_memory):

       ###### GENERATION ###### 
        print("Current Player is: ", self.player_names[self.player_id])
        # This description is for the action generator 
        generator_description = self._observation_to_description(observation, episodic_memory, working_memory)
        print(generator_description)
        
        # self.working_memory = generator_description.replace(self.partner_action_inference_string, '')
        
        self.generator_message = self.base_message + [{"role": "user", "content": generator_description}]
        
        add_to_dict_list(self.log_csv_dict, 'State', generator_description)
        add_to_dict_list(self.log_csv_dict, 'Message', self.generator_message)
        
        
        if len(self.player_actions) > 0:
            selected_move = self.player_actions.pop(0)
        else:
            action_string = self.llm_inference(self.generator_message)
            print(f'''{bcolors.WARNING}LLM RESPONSE: {action_string}{bcolors.ENDC}''')
            selected_move = self.find_best_match(action_string)
        ###### GENERATION ######
                
            #### VERIFICATION ####
            verification_response_string = ''
            verifier_responses = []
            verifier_description = f"State: {generator_description.replace(self.partner_action_inference_string, '')}\n\n My Solution: {selected_move}. Think step by step. Think about rules, think about conventions, and think about safety. " # https://arxiv.org/pdf/2401.04925.pdf
            print(f'''{bcolors.WARNING}VERIFIER INPUT: {verifier_description}{bcolors.ENDC}''')
            self.verifier_message = self.verifier_base_message + [{"role": "user", "content": verifier_description}]
            verification_response_string = self.llm_inference(self.verifier_message)
            verifier_responses.append(verification_response_string)
            print(f'''{bcolors.OKCYAN}VERIFICATION RESPONSE: {verification_response_string}{bcolors.ENDC}''')
            counter = 0 
            while 'verification: okay' not in verification_response_string.lower():   
                counter += 1
                self.generator_message.append({"role": "assistant", "content": action_string})
                updated_generator_message = f"Your selected action: {selected_move} is not appropriate. {verification_response_string}. Please choose another action. List of Available Actions:\n"
                for tm in self.transformed:
                    if tm.lower() != selected_move.lower():
                        updated_generator_message += tm 
                        updated_generator_message += '\n'

                self.generator_message.append({"role": "user", "content": updated_generator_message})
                
                action_string = self.llm_inference(self.generator_message)
                print(f"{bcolors.WARNING}LLM CORRECTED RESPONSE: {action_string}{bcolors.ENDC}") 
                selected_move = self.find_best_match(action_string)
                # Two Step Verification Only 
                # if counter == 2:
                #     break 
                
                # self.verifier_message.append({"role": "assistant", "content": verification_response_string})
                # self.verifier_message.append({"role": "user", "content": f"New Solution: {selected_move}. "})
                self.verifier_message[-1]["content"] = f"State: {generator_description.replace(self.partner_action_inference_string, '')}\n\n My Solution: {selected_move}. Think step by step. Think about rules, think about conventions, and think about safety. "
                print(self.verifier_message)
                verification_response_string = self.llm_inference(self.verifier_message)
                verifier_responses.append(verification_response_string) 
                print(f'''{bcolors.OKCYAN}VERIFICATION RESPONSE: {verification_response_string}{bcolors.ENDC}''')
                
            add_to_dict_list(self.log_csv_dict, 'VERIFICATION Response', ' ***** '.join(verifier_responses)) 


        add_to_dict_list(self.log_csv_dict, 'Generator Response', action_string) 
        add_to_dict_list(self.log_csv_dict, 'Selected Action', selected_move)
        self.action_history.append(selected_move.title())
        selected_move_idx = 0
        # for move in self.transformed:
        #     if selected_move == move.lower():
        #         selected_move_idx = self.transformed.index(move)
        #         break 
        final_selection, score = process.extractOne(selected_move, self.transformed)
        selected_move_idx = self.transformed.index(final_selection)
        
        # print('Hanabi Selected Move: ', observation.legal_moves()[selected_move_idx]) 
        add_to_dict_list(self.log_csv_dict, 'Selected Action in Hanabi Space', observation.legal_moves()[selected_move_idx])

        df = pd.DataFrame(self.log_csv_dict)
        
        if self.model_type == 'openai':
            df.to_csv(f"{self.log_dir}/{self.player_names[self.player_id]}_{self.model}_{self.time_stamp}.csv")
            np.save(f'{self.traj_dir}/{self.player_names[self.player_id]}_{self.model}_{self.time_stamp}.npy', self.action_history)
        else:
            df.to_csv(f"{self.log_dir}/{self.player_names[self.player_id]}_Mixtral_{self.time_stamp}.csv")
            np.save(f'{self.traj_dir}/{self.player_names[self.player_id]}_Mixtral_{self.time_stamp}.npy', self.action_history)

        self.prev_working_memory = self.working_memory.copy()
        return observation.legal_moves()[selected_move_idx]


class LLMAgentHanabiLive(LLMAgent):
    def _get_card_knowledge(self, observation):
        description = f"My cards based on my knowledge: \n"
        my_cards = [] 
        for i, card in enumerate(observation["card_knowledge"][0]):
            plausible_colors = card["color"]
            plausible_ranks = card["rank"]
            description += f"Card {i} could be: ["
            for color in plausible_colors:
                description += f"{self.c_map[color]}, "
            description = description[:-2]
            description += '] ['
            
            
            for rank in plausible_ranks:
                description += f"{int(rank) + 1}, "
            description = description[:-2]
            description += ']\n'
            description = description.replace("'", "")
        add_to_dict_list(self.log_csv_dict, 'My Card Knowledge', description)

        return description

    def _get_partner_cards(self, observation):
        description = f"I can see {self.player_names[1-self.player_id]}'s Cards are: \n"
        hands = observation["observed_hands"]
        partner_cards = []
        for i, card in enumerate(hands[-1]):
            card_color = card["color"]
            card_rank = int(card["rank"]) + 1
            description += f"[Card {i}: {self.c_map[card_color]} {card_rank}]\n"
        add_to_dict_list(self.log_csv_dict, 'Partner Cards', description)
        description = description.replace("'", "") 
        return description
    
    # Infer partner knowledge 
    def _infer_partner_knowledge(self, observation):
        description = f"{self.player_names[1-self.player_id]}'s Knowledge about {self.partner_pronoun} cards: \n"
        for i, card in enumerate(observation["card_knowledge"][1]):
            plausible_colors = card["color"]
            plausible_ranks = card["rank"]
            
            description += f"{self.player_names[1-self.player_id]} believes {self.partner_pronoun} Card {i} could be: ["
            for color in plausible_colors:
                description += f"{self.c_map[color]}, "
            description = description[:-2]
            description += '] ['
            for rank in plausible_ranks:
                description += f"{int(rank) + 1}, "
            description = description[:-2]
            description += ']\n'
            description = description.replace("'", "")

        add_to_dict_list(self.log_csv_dict, 'Partner Card Knowledge', description)

        return description
    
    def convert_pyhanabi_move(self, move):
        if move['action_type'] == 'DISCARD':
            return f"Discard My Card {move['card_index']}."
        elif move['action_type'] == 'PLAY':
            return f"Play My Card {move['card_index']}."
        elif move['action_type'] == 'REVEAL_COLOR':
            return f"Reveal {self.player_names[1-self.player_id]}'s {self.c_map[move['color']]} color cards."
        elif move['action_type'] == 'REVEAL_RANK':
            return f"Reveal {self.player_names[1-self.player_id]}'s rank {move['rank']+1} cards."
        
    def convert_pyhanabi_partner_move(self, move):
        if move['action_type'] == 'DISCARD':
            return f"discarded {move['color']} {move['rank'] + 1} card."
        elif move['action_type'] == 'PLAY':
            return f"played {move['color']} {move['rank'] + 1} card."
        elif move['action_type'] == 'REVEAL_COLOR':
            return f"revealed my {self.c_map[move['color']]} color cards."
        elif move['action_type'] == 'REVEAL_RANK':
            return f"revealed my rank {move['rank']+1} cards." 

    def _get_legal_moves(self, observation):
        self.transformed = []
        moves = observation["legal_moves"]
        
        for idx, move in enumerate(moves):
            self.transformed.append(f"{chr(65+idx)}. {self.convert_pyhanabi_move(move)}")

        description = 'Available Legal Actions: \n'
        for tm in self.transformed:
            description += tm
            description += '\n'
        add_to_dict_list(self.log_csv_dict, 'Available Actions', description.replace('#', ''))
        return description

    def _get_current_stack(self, observation):
        description = 'Current Stacks: '
        colors = 'RYGWB'
        color_names = ['Red', 'Yellow', 'Green', 'White', 'Blue']
        stack_state = []
        for i, stack_color in enumerate(observation['fireworks']):
            description += f"{color_names[i]} - {color_names[i]} {observation['fireworks'][stack_color]} "
        description += '\n'
        add_to_dict_list(self.log_csv_dict, 'Stack', str(stack_state))
        return description
    
    def _add_soft_constraints(self, observation):
        description = f'The next playable cards for each stack are: '
        colors = 'RYGWB'
        color_names = ['Red', 'Yellow', 'Green', 'White', 'Blue']
        stack_state = []
        for i, firework in enumerate(colors):
            if observation['fireworks'][colors[i]] != 4:
                description += f"{color_names[i]} - {color_names[i]} {observation['fireworks'][colors[i]]+1} "
            else:
                description += f"{color_names[i]} stack is complete. "
        
        # description += str(stack_state)
        add_to_dict_list(self.log_csv_dict, 'Soft Constraints', description)
        description += '\n'
        return description

    def _get_previous_selected_actions(self): 
        if len(self.action_history)>0:
            add_to_dict_list(self.log_csv_dict, 'Prev Actions', f'My Action History: {", ".join([ac for ac in self.action_history])}')
            return f'My Action History: {", ".join([ac for ac in self.action_history])}\n'
        else:
            add_to_dict_list(self.log_csv_dict, 'Prev Actions', f'My Action History: ')
            return ''
    
    ## TODO: Complete this 
    def _infer_partner_action(self, partner_action, description):
        partner_action_inference_description = ''
        if partner_action is not None:
            print("PARTNER's MOVE: ", partner_action)
            partner_move_string = self.convert_pyhanabi_partner_move(partner_action)
            print("PARTNER's MOVE: ", partner_move_string)
            epistemic_message = self.epistemologist_message + [{"role": "user", "content": f"***{self.player_names[1-self.player_id]}'s selected action***: {self.player_names[1-self.player_id]} {partner_move_string}\n\nMy current state information: {description}. Note that I have updated my knowledge of my cards based on partner's action. Think step by step about partners action. Think about the action. Think about what it implies. If I should give a clue next, think about what clue I can give my partner."}]
            epistemic_response_string = self.llm_inference(epistemic_message)
            partner_action_inference_description = f"Interpretation of {self.player_names[self.player_id-1]}'s Last Action: {epistemic_response_string}. \n"
            self.partner_action_inference_string = partner_action_inference_description
        return partner_action_inference_description

    def _get_discard_pile(self, observation):
        cards = ''
        for card in observation['discard_pile']:
            cards += f'{self.c_map[card["color"]]} {int(card["rank"])+1}, '
        description = f"{cards}\n"
        return description
        
    def _observation_to_description(self, observation, partner_action):
        self.working_memory['turn'] = f'It is currently My ({self.player_names[self.player_id]}) turn. '
        description = self.working_memory['turn']

        self.working_memory['stack'] = self._get_current_stack(observation)
        description += self.working_memory['stack']
        
        self.working_memory['card_knowledge'] = self._get_card_knowledge(observation)
        description += self.working_memory['card_knowledge']

        self.working_memory['partner_cards'] = self._get_partner_cards(observation)
        description += self.working_memory['partner_cards']

        self.working_memory['partner_card_knowledge'] = self._infer_partner_knowledge(observation)
        description += self.working_memory['partner_card_knowledge']

        self.working_memory['reveal_tokens'] = f"Remaining Reveal Tokens: {observation['information_tokens']}\n"
        description += self.working_memory['reveal_tokens']

        self.working_memory['lives'] = f"Remaining Lives: {observation['life_tokens']}\n"
        description += self.working_memory['lives']

        self.working_memory['deck_size'] = f"Deck Size: {observation['deck_size']}\n"
        description += self.working_memory['deck_size']

        self.working_memory['discard_pile'] = f"The discard pile is: {self._get_discard_pile(observation)}\n"
        description += self.working_memory['discard_pile']

        # description += f'\nInformation: We have {observation.information_tokens()} reveal tokens, {observation.life_tokens()} life tokens. The discard pile consists {observation.discard_pile()} and the deck size is {observation.deck_size()}.\n'
        add_to_dict_list(self.log_csv_dict, 'Board Information', f"Information: We have {self.working_memory['reveal_tokens']} reveal tokens, {self.working_memory['lives']} lives. The discard pile consists {self.working_memory['discard_pile']} and the deck size is {self.working_memory['deck_size']}.")
        
        self.working_memory['previous_selected_actions'] = self._get_previous_selected_actions()
        description += self.working_memory['previous_selected_actions']
        
        self.working_memory['soft_constraints'] = self._add_soft_constraints(observation)
        description += self.working_memory['soft_constraints']

        self.working_memory['partner_interpretation'] = self._infer_partner_action(partner_action, description)
        description += self.working_memory['partner_interpretation']
        
        self.working_memory['legal_moves'] = self._get_legal_moves(observation)
        description += self.working_memory['legal_moves']
        
        return description

    def get_next_move(self, observation, partner_action):
        
        # Print current player 
        print("Current Player is: ", self.player_names[self.player_id])

        # This description is for the action generator 
        generator_description = self._observation_to_description(observation, partner_action)
        print(generator_description)
        
        # prepare generator message 
        self.generator_message = self.base_message + [{"role": "user", "content": generator_description}]
        
        # log
        add_to_dict_list(self.log_csv_dict, 'State', generator_description)
        add_to_dict_list(self.log_csv_dict, 'Message', self.generator_message)
        
        
        if len(self.player_actions) > 0:
            selected_move = self.player_actions.pop(0)
        else:
            # Generate initial player action 
            action_string = self.llm_inference(self.generator_message)
            print(f'''{bcolors.WARNING}LLM RESPONSE: {action_string}{bcolors.ENDC}''')
            selected_move = self.find_best_match(action_string)
        
                
            # Initial verification 
            verification_response_string = ''
            verifier_responses = []
            verifier_description = f"State: {generator_description.replace(self.partner_action_inference_string, '')}\n\n My Solution: {selected_move}. Think step by step. Think about rules, think about conventions, and think about safety. " # https://arxiv.org/pdf/2401.04925.pdf
            print(f'''{bcolors.WARNING}VERIFIER INPUT: {verifier_description}{bcolors.ENDC}''')
            self.verifier_message = self.verifier_base_message + [{"role": "user", "content": verifier_description}]
            verification_response_string = self.llm_inference(self.verifier_message)
            verifier_responses.append(verification_response_string)
            print(f'''{bcolors.OKCYAN}VERIFICATION RESPONSE: {verification_response_string}{bcolors.ENDC}''')
            counter = 0 

            # Repeat generation and verification, till verification: okay
            while 'verification: okay' not in verification_response_string.lower():   
                counter += 1
                
                # Add verifier feedback to generator messages 
                self.generator_message.append({"role": "assistant", "content": action_string})
                updated_generator_message = f"Your selected action: {selected_move} is not appropriate. {verification_response_string}. Please choose another action. List of Available Actions:\n"
                
                # Remove the action that was rejected from the list of available actions 
                for tm in self.transformed:
                    if tm.lower() != selected_move.lower():
                        updated_generator_message += tm 
                        updated_generator_message += '\n'
                self.generator_message.append({"role": "user", "content": updated_generator_message})
                
                # Generate a new action 
                action_string = self.llm_inference(self.generator_message)
                print(f"{bcolors.WARNING}LLM CORRECTED RESPONSE: {action_string}{bcolors.ENDC}") 
                selected_move = self.find_best_match(action_string)
                
                # Verify new action 
                self.verifier_message[-1]["content"] = f"State: {generator_description.replace(self.partner_action_inference_string, '')}\n\n My Solution: {selected_move}. Think step by step. Think about rules, think about conventions, and think about safety. "
                print(self.verifier_message)
                verification_response_string = self.llm_inference(self.verifier_message)
                verifier_responses.append(verification_response_string) 
                print(f'''{bcolors.OKCYAN}VERIFICATION RESPONSE: {verification_response_string}{bcolors.ENDC}''')
                
            add_to_dict_list(self.log_csv_dict, 'VERIFICATION Response', ' ***** '.join(verifier_responses)) 


        add_to_dict_list(self.log_csv_dict, 'Generator Response', action_string) 
        add_to_dict_list(self.log_csv_dict, 'Selected Action', selected_move)
        self.action_history.append(selected_move.title())
        selected_move_idx = 0
        
        final_selected_move, score = process.extractOne(selected_move, self.transformed)
        # for move in self.transformed:
        #     if selected_move == move.lower():
        #         selected_move_idx = self.transformed.index(move)
        #         break 
        selected_move_idx = self.transformed.index(final_selected_move)
        
        # print('Hanabi Selected Move: ', observation.legal_moves()[selected_move_idx]) 
        add_to_dict_list(self.log_csv_dict, 'Selected Action in Hanabi Space', observation["legal_moves"][selected_move_idx])

        self.prev_working_memory = self.working_memory.copy()
        return observation["legal_moves"][selected_move_idx]