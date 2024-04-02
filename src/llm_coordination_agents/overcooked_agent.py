from openai import OpenAI, AzureOpenAI
import csv
from tqdm import tqdm 
import numpy as np  
import re 
import sys 
import json 
import logging
import sys
import pandas as pd 
import datetime
import random 

from overcooked_ai_py.mdp.actions import LLMActionSet
logging.basicConfig(filename='debug.log', level=logging.DEBUG)
logging.debug('Initiated Logger...')
import time 
import os.path
from fuzzywuzzy import process

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

set_global_seed(42)

# When Using Locally hosted model: 
# openai.api_key = "EMPTY"  
# openai.api_base = "http://localhost:8002/v1" # Replace with URL of locally hosted model. Tested with https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md

# When using openai API
# openai.api_key = os.environ['API_KEY']
# openai.organization = os.environ['ORGANIZATION']


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

def add_to_dict_list(dictionary, key, item):
    if key not in dictionary:
        dictionary[key] = [item]
    else:
        dictionary[key].append(item)
        
EnvDescriptions= {
    'cramped_room': 'Environment Details: The environment is rectangular with 2 onions dispensers (o0, o1), cooker (c0), plate dispenser (p0) and delivery area (d0). Additionally there are kitchen counters (k0 to k8) which can be used to temporarily store onions and plates while you do something else. Objects on counters can be picked up later and should be considered as they may be closer than items in dispensers.',
    
    'no_counter_door': 'Environment Details: The environment is cramped with little space to move aroundrectangular with 2 onions dispensers (o0, o1), cooker (c0) and plate dispenser (p0). The delivery area (d0) is inaccessible behind closed gates and can be accessed by opening one of the gates (g0, g1). A gate can only be opened by a player if they are not carrying an object. Once the door is opened it will only stay open for a brief time and then close on its own',

    'soup_passing':"Environment Details: The environment is divided into 2 partitions. Alice is in the left partition with access to onion dispenser (o0), plate dispenser (p1), cooker (c0), delivery area (d0) and kitchen counters (k0, k1, k8, k11, k12). Bob is in the right partition with access to onion dispenser (o1), plate dispenser (p0), cooker (c1) and kitchen counters (k3, k4, k6, k10, k14, k15). Only Alice has access to the delivery area (d0). There are shared counters (s0, s1, s2, s3) which can be used to pass onions, plates or cooked soup from one player to the other. Note that the objects on the shared counters can be accessed by both players. Objects on counters can be picked up later and should be considered as they may be closer than items in dispensers. ",

    'soup_passing_door':"Environment Details: The environment is divided into 2 partitions. Alice is in the left partition with access to onion dispenser o0, plate dispenser p1, cooker c0, and delivery area d0. Bob is in the right partition with access to onion dispenser o1, plate dispenser p0, and cooker c1.  The two partitions are connected by gate g0 which can be opened if a player is holding nothing. Opening the gate will allow players to move freely between partitions.  The gate will close after enough time automatically. ",

    'asymmetric_advantages' : 'Environment Details: There are two partitions in the current environment. Bob is in the left partition with access to onion dispenser o0, delivery area d0, plate dispenser p0 and kitchen counters k0, k1, k2, k3, k4, k11, k12, k16, k18, k20, k21, k22, k23. Alice is in the right partition and has access to onion dispenser o1, delivery area d1, plate dispenser p1 and kitchen counters k6, k7, k8, k9, k10, k14, k15, k17, k19, k25, k26, k27, k28. Both have access to both cookers c0 and c1 which are on the partition line. Kitchen counters (k0 to k28) which can be used to temporarily store onions and plates while you do something else. Objects on counters can be picked up later and should be considered as they may be closer than items in dispensers.',

    'forced_coordination' : 'Environment Details: The environment is split into two partitions, one with each player. In the right partition, Alice has access to cookers (c0, c1),  delivery area (d0) and kitchen counters (k6, k8, k12). In the left partition, Bob has access to onion dispensers (o0, o1), plate dispenser (p0) and kitchen counters (k1, k10). Kitchen counters can be used to temporarily store onions and plates while you do something else. Both players have access to shared counters (s0, s1, s2) which can be used to transfer onions and plates to the other player depending on the situation. Note that the objects on the shared counters can be accessed by both players. ',


    'coordination_ring': 'Environment Details: The environment is narrow and circular with room for only one player to walk along a path or access places. It features onion dispensers (o0, o1), plate dispenser (p0), cookers (c0, c1), and a delivery area (d0). Additionally there are kitchen counters (k0 to k10) which should only be used to temporarily store onions and plates while you do something else. Objects on counters can be picked up later and should be considered as they may be closer than items in dispensers',

    'counter_circuit_o_1order': 'Environment Details: The environment is circular with two onion dispensers (o0, o1), plate dispenser (p0), cookers (c0, c1) and delivery area d0. There are also the shared counters (s0, s1, s2, s3) which can be used to pass objects from one player to the other. Additionally there are kitchen counters (k0 to k15) which can be used to temporarily store onions and plates while you do something else. ',

    'bottleneck': 'Environment Details: There is 1 onion dispenser (o0), plate dispenser (p0), cookers (c0, c1) and delivery area (d0). Additionally there are kitchen counters (k0 to k17) which can be used to temporarily store onions and plates while you do something else. ', 

    'large_room': 'Environment Details: There are 2 onions dispensers (o0, o1), cooker (c0), plate dispenser (p0) and delivery area (d0). Additionally there are kitchen counters (k0 to k19) which can be used to temporarily store onions and plates while you do something else. ', 
    
    'centre_objects': 'Environment Details: There is 1 onion dispenser (o0), cooker (c0), plate dispenser (p0) and delivery area (d0). Additionally there are kitchen counters (k0 to k24) which can be used to temporarily store onions and plates while you do something else. ', 

    'centre_pots': 'Environment Details: There are 2 onion dispenser (o0, 1), 2 cookers (c0, c1), 2 plate dispensers (p0, 1) and 2 delivery areas (d0, d1). Additionally there are kitchen counters (k0 to k14) which can be used to temporarily store onions and plates while you do something else. ', 
}

EnvDescriptions_SingleAgentAblation = {
    'cramped_room': 'The environment is rectangular with 2 onions dispensers (o0, o1), cooker (c0), plate dispenser (p0) and delivery area (d0). Additionally there are kitchen counters (k0 to k8) which can be used to temporarily store onions and plates while you do something else. Objects on counters can be picked up later and should be considered as they may be closer than items in dispensers.',

    'forced_coordination' : 'The environment is split into two partitions, one with each player. In the right partition, Alice has access to cookers (c0, c1),  delivery area (d0) and kitchen counters (k6, k8, k12). In the left partition, Bob has access to onion dispensers (o0, o1), plate dispenser (p0) and kitchen counters (k1, k10). Kitchen counters can be used to temporarily store onions and plates while you do something else. Both players have access to shared counters (s0, s1, s2) which can be used to transfer onions and plates to the other player depending on the situation. Note that the objects on the shared counters can be accessed by both players. ',
}

class LLMManager:
    def __init__(self, model_name, model_type, cache_dir, temperature=0.6, do_sample=True, max_new_tokens=1000, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, api_server=True):
        self.model_name = model_name 
        self.model_type = model_type
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_server = api_server
        self.device = 'cuda'
        self.gpt_3_base_cost = 0.00075
        self.gpt_4_base_cost = 0.011
        self.gpt3_cost = 0
        self.gpt4_cost = 0
        if self.model_type == 'openai':
            self.akey = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.org = os.getenv("AZURE_OPENAI_API_KEY")
            # self.client = OpenAI(api_key = self.akey, organization = self.org)
            self.client = AzureOpenAI(
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                api_version="2023-05-15"
            )
        else:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8000/v1"
            )
        self.inference_fn = self.run_openai_inference                    

    def run_openai_inference(self, messages):
        api_call_start = time.time()
        completion = self.client.chat.completions.create(
            messages = messages,
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )
        print(f"{bcolors.FAIL}LLM INFERENCE TIME: {time.time() - api_call_start}{bcolors.ENDC}")
        # print("INFERENCE STRING: ", completion.choices[0].message.content)
        total_tokens = completion.usage.total_tokens
        gpt_3_cost =  self.gpt_3_base_cost * (total_tokens / 1000)
        gpt_4_cost = self.gpt_4_base_cost * (total_tokens / 1000)
        self.gpt3_cost += gpt_3_cost
        self.gpt4_cost += gpt_4_cost
        print(f"COST SO FAR GPT-4: {self.gpt4_cost} USD")
        return completion.choices[0].message.content



class LLMAgent:
    def __init__(self, player_id, layout_name):
        self.player_id = player_id
        self.layout_name = layout_name
        self.player_names = ['Alice', 'Bob']
        

        # Controls 
        self.DEBUG = True     
        self.enable_cache = False # True    
        self.write_to_cache = False 
        self.save_trajectory = True # True 
        # Enable kitchen counters only for GPT-4, other models cannot handle the complexity
        self.enable_kitchen_counters = True   
        self.explicit_help = False 
        self.single_agent_ablation = True  
        self.log_replay = pd.read_csv(f'/home/saaket/llm_coordination/src/agentic_evals/game_logs/ai/forced_coordination/forced_coordination_ai_gpt-4-0125_player_{self.player_id}_2024-03-28_16-39-54.csv')
        self.replay_actions = list(self.log_replay['selected_action'])
        self.model = 'gpt-4-0125'
        self.model_name = 'gpt-4-0125'
        # self.model = 'gpt-35-turbo'
        # self.model_name = 'gpt-35-turbo'
        self.model_type = 'openai'
        # self.model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        # self.model_type = 'mistral'
        # self.model = 'mixtral'

        self.llm = LLMManager(model_name=self.model_name, model_type=self.model_type, cache_dir=os.getenv('HF_HOME'))

        self.experiment_type = 'ai'
        
        ### LOGGING ###
        self.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.trial_note = f'{self.layout_name}_{self.experiment_type}_{self.model}'        
        self.trajectory_dir = f'game_trajectories/{self.experiment_type}/{self.layout_name}/'
        self.log_dir = f'game_logs/{self.experiment_type}/{self.layout_name}/'
        if not os.path.isdir(self.trajectory_dir):
            os.makedirs(self.trajectory_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.trajectory_path = self.trajectory_dir + f'/{self.trial_note}_player_{self.player_id}_{self.time_stamp}.npy'
        ####
        ### LOGGING ###
        
        self.action_set = LLMActionSet[self.layout_name]
        self.num_api_calls = 0 
        self.player_actions = []

        # Set other player ID 
        if int(self.player_id) == 0:
            self.other_player_id = 1
        else:
            self.other_player_id = 0


        self.llm_system_prompt = "You are a friendly chat assistant who is correct and brief at all times."
        self.partner_action_inference_string = ''

        # if self.explicit_help:
        #     self.base_prompt = f'''In the game Overcooked, I am {self.player_names[self.player_id]}, my teammate is {self.player_names[self.player_id]}. 
        #     {EnvDescriptions[self.layout_name]} 
        #     We must coordinate to make onion soups with 3 onions each. Once a soup is cooked it needs to be placed on a plate and delivered. I can only carry one item at a time. My goal is to maximize the number of deliveries. I want to be efficient and prepare for the next soup while the current soup is cooking. I'll provide my action history, current state, teammate's status, and my possible actions. I want to prefer helping the other player with their cooking and delivery if the situation arises. Help me select the best action from the list. Format your response as: Explanation:<Brief explanation for next action including a prediction of {self.player_names[self.player_id]}'s next action>. Action: <action>. Only select one action. Do not say anything else. Got it?'''
        # else:
        #     self.base_prompt = f'''In the game Overcooked, I am {self.player_names[self.player_id]}, my teammate is {self.player_names[self.player_id]}. 
        #     {EnvDescriptions[self.layout_name]} 
        #     We must coordinate to make onion soups with 3 onions each. Once a soup is cooked it needs to be placed on a plate and delivered. I can only carry one item at a time. My goal is to maximize the number of deliveries. I want to be efficient and prepare for the next soup while the current soup is cooking. I'll provide my action history, current state, teammate's status, and my possible actions. Help me select the best action from the list. Format your response as: Explanation:<Brief explanation for next action including a prediction of {self.player_names[self.player_id]}'s next action>. Action: <action>. Only select one action. Do not say anything else. Got it?'''
        
        self.rules = f'''Players must coordinate to make onion soups with 3 onions each. Once a soup is cooked it needs to be placed on a plate and delivered. Players can only carry one item at a time. A soup can only be loaded onto plate by a player if they are holding a plate. The goal is to maximize the number of deliveries.'''

        self.conventions_explicit_help = f'''
        1. We will try to maximize the number of deliveries. 
        2. We will try to be efficient and prepare for the next soup while the current soup is cooking. 
        3. If we are in the same section, we will minimize our movement to avoid getting in each other's way.
        4. We will prefer helping the other player with their cooking and delivery if the situation arises.'''

        # self.conventions= f'''
        # 1. We will try to maximize the number of deliveries. 
        # 2. We will try to be efficient and prepare for the next soup while the current soup is cooking. 
        # '''
        # self.conventions = f'''1. We want to be efficient and prepare for the next soup while the current soup is cooking. 
        # '''

        self.conventions = f'''1. We want to be efficient and prepare for the next soup while the current soup is cooking. 
        2. We want to prefer helping our partner with their cooking and delivery if the situation arises
        '''



        # self.conventions = f'''1. We want to be efficient and prepare for the next soup while the current soup is cooking. '''
        # self.conventions = f'''
        #     1. If we are in the same section of workspace, we will minimize our movement to avoid getting in each other's way.  
        #     2. We will start preparing the next soup while the one before is being cooked/delivered.
        #     3. We will make optimal use of all available cookers, and counters.
        #     4. We will take actions that ensures as many soups are delivered as possible 
        # '''

        self.pi_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. {EnvDescriptions[self.layout_name]} 
        Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}. I'll provide my action history, current state, teammate's status, and my possible actions. Help me understand my partner's intentions and needs. describe what my partner intends to do or needs in one sentence only. Do not say anything else.'''

        self.verifier_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. {EnvDescriptions[self.layout_name]} 
        Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}.'''

        self.verifier_system_prompt = 'You are an action verification agent for Overcooked. I will provide you with my inventory, location information, and state information for me and my partner and my selected action. You need to check whether the action satisfies the criteria: 1. Rule Following: It follows to the rules of the game. 2. Convention Following: It adheres to the mentioned conventions 3. Safety: The selected action does not lead to the game being stuck. Your response should be Reasoning:<Brief Reasoning for Verification> followed by "Verification: Okay" if selected action follows **all three** criteria and "Verification: Not Okay" otherwise. Do not say anything else. Got it?'
        

        # With COT
        # self.base_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. {EnvDescriptions[self.layout_name]} 
        # Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}. I'll provide my action history, current state, teammate's status, and my possible actions. Help me select the best action from the list. Format your response as: Explanation:<Brief explanation for my next action>. Action: <action>. Only select one action. Do not say anything else. Got it?'''

        # Without COT
        self.base_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. {EnvDescriptions[self.layout_name]}
        Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}. I'll provide my action history, current state, teammate's status, and my possible actions. Help me select the best action from the list. Format your response as: Action: <action>. Only select one action. Do not say anything else. Got it?'''

        # self.base_prompt = f'''I am {self.player_names[self.player_id]}. I am playing the game Overcooked with my partner {self.player_names[self.other_player_id]}. Overcooked has the following rules: {self.rules}. We have agreed to follow the following conventions: {self.conventions}. I'll provide my action history, current state, teammate's status, and my possible actions. Help me select the best action from the list. Format your response as: Action: <action>. Only select one action. Do not say anything else. Got it?'''
        self.assistant_response_initial = f'''Got it!'''

        self.action_regex = r"Action:\s*(.*)"

        if self.model_type == 'openai':
            self.message = [
                        {"role": "system", "content": self.llm_system_prompt},
                        {"role": "user", "content": self.base_prompt},
                        {"role": "assistant", "content": self.assistant_response_initial},
                    ]
            self.pi_message = [
                        {"role": "system", "content": self.llm_system_prompt},
                        {"role": "user", "content": self.pi_prompt},
                        {"role": "assistant", "content": self.assistant_response_initial},
            ]
            self.verifier_base_message = [
                        {"role": "system", "content": self.verifier_system_prompt},
                        {"role": "user", "content": self.base_prompt},
                        {"role": "assistant", "content": self.assistant_response_initial},
            ]
        else:
            self.message = [
                        {"role": "user", "content": self.base_prompt},
                        {"role": "assistant", "content": self.assistant_response_initial},
                    ]

        self.summary_so_far = ''
        self.tokens_used = 0 
        if self.enable_cache:  
            if os.path.isfile(self.cache_path):
                self.cache = json.load(open(self.cache_path, 'r'))
            else:
                self.cache = {}
        else:
            self.cache = {}
        
        self.all_actions = []
        for key, value in self.action_set.items():
            if isinstance(value, list):
                self.all_actions.extend(value)
        # print(self.all_actions)
        self.log_csv_dict = {}
        self.action_history = []
    
    def _get_available_actions(self, state_for_llm, message):
        # Available Action Constraints
        available_actions = []
        # Check what player is holding 
        if state_for_llm[self.player_id]['held_object'] == "nothing":
            for idx, d in enumerate(state_for_llm['distances']['onion_dispenser']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['onion_dispenser'][idx])

            for idx, d in enumerate(state_for_llm['distances']['plate_dispenser']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['plate'][idx])
            
            if self.enable_kitchen_counters:
            
                for idx, d in enumerate(state_for_llm['distances']['kitchen_counter']):
                    if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'onion':
                        available_actions.append(self.action_set['kitchen_counter_pick_onion'][idx])
                    if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'plate':
                        available_actions.append(self.action_set['kitchen_counter_pick_plate'][idx])
                    if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'soup in plate':
                        available_actions.append(self.action_set['kitchen_counter_pick_soup'][idx])


            if 'storage_counter_pick_onion' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'onion':
                            available_actions.append(self.action_set['storage_counter_pick_onion'][idx])
                        elif state_for_llm['storage_counter_objects'][idx] == 'plate':
                            available_actions.append(self.action_set['storage_counter_pick_plate'][idx])
                        elif state_for_llm['storage_counter_objects'][idx] == 'soup in plate':
                            available_actions.append(self.action_set['storage_counter_pick_soup'][idx])
            
            for idx, d in enumerate(state_for_llm['distances']['gate']):
                if d[0] not in ['infinite']:
                    if state_for_llm['gate_status'][idx] == 'closed':
                        available_actions.append(self.action_set['gate'][idx])

            # Add turn on cooker to instruction instead of internal mechanism
            # for idx, d in enumerate(state_for_llm['distances']['cooker']):
            #     if d[0] != 'infinite':
            #         available_actions.append(self.action_set['cooking_status'][idx])

        elif state_for_llm[self.player_id]['held_object'] == 'onion':
            for idx, d in enumerate(state_for_llm['distances']['cooker']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['cooker'][idx])

            if self.enable_kitchen_counters:
                if len(self.empty_kitchen_counters)>0:
                    
                    kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
                    k_action = self.action_set['kitchen_counter_place_onion'][kidx]
                    
                    available_actions.append(k_action)

            if 'storage_counter_place_onion' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            available_actions.append(self.action_set['storage_counter_place_onion'][idx])

        elif state_for_llm[self.player_id]['held_object'] == 'plate':
            for idx, d in enumerate(state_for_llm['distances']['cooker']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['cooked_soup'][idx])
            
            if self.enable_kitchen_counters:
                if len(self.empty_kitchen_counters)>0:
                    kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
                    k_action = self.action_set['kitchen_counter_place_plate'][kidx]
                    
                    available_actions.append(k_action)

            if 'storage_counter_place_plate' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            available_actions.append(self.action_set['storage_counter_place_plate'][idx])

        elif state_for_llm[self.player_id]['held_object'] == 'soup in plate':
            for idx, d in enumerate(state_for_llm['distances']['delivery_zone']):
                if d[0] not in ['infinite']:
                    available_actions.append(self.action_set['delivery_area'][idx])
            if self.enable_kitchen_counters:
                if len(self.empty_kitchen_counters)>0:
                    kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
                    k_action = self.action_set['kitchen_counter_place_plate'][kidx]
                    
                    available_actions.append(k_action)

            if 'storage_counter_place_soup' in self.action_set:
                for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
                    if d[0] not in ['infinite']:
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            available_actions.append(self.action_set['storage_counter_place_soup'][idx])
        return available_actions + self.action_set['wait'] + self.action_set['collision_avoidance']

    # def _get_available_actions(self, state_for_llm, message):
    #     # Available Action Constraints
    #     available_actions = []

    #     #### IF Player is holding NOTHING ####
    #     if state_for_llm[self.player_id]['held_object'] == "nothing":
        
    #         for idx, d in enumerate(state_for_llm['distances']['onion_dispenser']):
    #             if d[0] not in ['infinite']:
    #                 available_actions.append(f'pick up onion from o{idx}')

    #         for idx, d in enumerate(state_for_llm['distances']['plate_dispenser']):
    #             if d[0] not in ['infinite']:
    #                 available_actions.append(f'pick up plate from p{idx}')
            
    #         if self.enable_kitchen_counters:
    #             for idx, d in enumerate(state_for_llm['distances']['kitchen_counter']):
    #                 if d[0] not in ['infinite']:
    #                     if state_for_llm['kitchen_counter_objects'][idx] == 'onion':
    #                         available_actions.append(f'pick up onion from k{idx}')
    #                     if state_for_llm['kitchen_counter_objects'][idx] == 'plate':
    #                         available_actions.append(f'pick up plate from k{idx}')
    #                     if state_for_llm['kitchen_counter_objects'][idx] == 'soup in plate':
    #                         available_actions.append(f'pick up cooked soup from k{idx}')

    #         # If layout contains shared counter
    #         if 'storage_counter_pick_onion' in self.action_set:
    #             for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
    #                 if d[0] not in ['infinite']:
    #                     if state_for_llm['storage_counter_objects'][idx] == 'onion':
    #                         available_actions.append(f'pick up onion from s{idx}')
    #                     elif state_for_llm['storage_counter_objects'][idx] == 'plate':
    #                         available_actions.append(f'pick up plate from s{idx}')
    #                     elif state_for_llm['storage_counter_objects'][idx] == 'soup in plate':
    #                         available_actions.append(f'pick up cooked soup from s{idx}')
            
    #         for idx, d in enumerate(state_for_llm['distances']['gate']):
    #             if d[0] not in ['infinite']:
    #                 if state_for_llm['gate_status'][idx] == 'closed':
    #                     available_actions.append(f'open gate g{idx}')
    #     #### END Player is holding NOTHING ####

    #     #### IF Player is holding Onion ####
    #     elif state_for_llm[self.player_id]['held_object'] == 'onion':
    #         for idx, d in enumerate(state_for_llm['distances']['cooker']):
    #             if d[0] not in ['infinite']:
    #                 available_actions.append(f'place onion in c{idx}')

    #         if self.enable_kitchen_counters:
    #             if len(self.empty_kitchen_counters)>0:
                    
    #                 kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
    #                 available_actions.append(f'place onion in k{kidx}')

    #         if 'storage_counter_place_onion' in self.action_set:
    #             for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
    #                 if d[0] not in 'infinite':
    #                     if state_for_llm['storage_counter_objects'][idx] == 'empty':
    #                         available_actions.append(f'place onion on s{idx}')
    #     #### END Player is holding Onion ####

    #     #### IF Player is holding Plate ####
    #     elif state_for_llm[self.player_id]['held_object'] == 'plate':
    #         for idx, d in enumerate(state_for_llm['distances']['cooker']):
    #             if d[0] not in 'infinite':
    #                 available_actions.append(f'load cooked soup on plate from c{idx}')
            
    #         if self.enable_kitchen_counters:
    #             if len(self.empty_kitchen_counters)>0:
    #                 kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
    #                 available_actions.append(f'place plate on k{kidx}')

    #         if 'storage_counter_place_plate' in self.action_set:
    #             for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
    #                 if d[0] not in ['infinite']:
    #                     if state_for_llm['storage_counter_objects'][idx] == 'empty':
    #                         available_actions.append(f'place plate on s{idx}')
    #     #### END Player is holding Plate ####

    #     #### If Player is holding Soup ####
    #     elif state_for_llm[self.player_id]['held_object'] == 'soup in plate':
    #         for idx, d in enumerate(state_for_llm['distances']['delivery_zone']):
    #             if d[0] not in ['infinite']:
    #                 available_actions.append(f'place soup for delivery in d{idx}')

    #         if self.enable_kitchen_counters:
    #             if len(self.empty_kitchen_counters)>0:
    #                 kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
    #                 # k_action = self.action_set['kitchen_counter_place_plate'][kidx]
                    
    #                 available_actions.append(f'place cooked soup on k{idx}')

    #         if 'storage_counter_place_soup' in self.action_set:
    #             for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
    #                 if d[0] not in ['infinite']:
    #                     if state_for_llm['storage_counter_objects'][idx] == 'empty':
    #                         available_actions.append(f'place cooked soup on s{idx}')

    #     #### END Player is holding Soup ####
    #     available_actions.append('wait.')
    #     available_actions.append('move away.')

    #     return self.add_ordinals_to_actions(available_actions)

    # def add_ordinals_to_actions(self, available_actions):
    #     for i in range(len(available_actions)):
    #         available_actions[i] = f'{available_actions[i]}'
    #     return available_actions

    
    # def _get_available_actions(self, state_for_llm, message):
    #     # Available Action Constraints
    #     available_actions = []
    #     # Check what player is holding 
    #     if state_for_llm[self.player_id]['held_object'] == "nothing":
    #         for idx, d in enumerate(state_for_llm['distances']['onion_dispenser']):
    #             if d[0] not in ['infinite']:
    #                 available_actions.append(self.action_set['onion_dispenser'][idx])

    #         for idx, d in enumerate(state_for_llm['distances']['plate_dispenser']):
    #             if d[0] not in ['infinite']:
    #                 available_actions.append(self.action_set['plate'][idx])
            
    #         if self.enable_kitchen_counters:
            
    #             for idx, d in enumerate(state_for_llm['distances']['kitchen_counter']):
    #                 if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'onion':
    #                     available_actions.append(self.action_set['kitchen_counter_pick_onion'][idx])
    #                 if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'plate':
    #                     available_actions.append(self.action_set['kitchen_counter_pick_plate'][idx])
    #                 if d[0] not in ['infinite'] and state_for_llm['kitchen_counter_objects'][idx] == 'soup in plate':
    #                     available_actions.append(self.action_set['kitchen_counter_pick_soup'][idx])


    #         if 'storage_counter_pick_onion' in self.action_set:
    #             for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
    #                 if d[0] not in ['infinite']:
    #                     if state_for_llm['storage_counter_objects'][idx] == 'onion':
    #                         available_actions.append(self.action_set['storage_counter_pick_onion'][idx])
    #                     elif state_for_llm['storage_counter_objects'][idx] == 'plate':
    #                         available_actions.append(self.action_set['storage_counter_pick_plate'][idx])
    #                     elif state_for_llm['storage_counter_objects'][idx] == 'soup in plate':
    #                         available_actions.append(self.action_set['storage_counter_pick_soup'][idx])
            
    #         for idx, d in enumerate(state_for_llm['distances']['gate']):
    #             if d[0] not in ['infinite']:
    #                 if state_for_llm['gate_status'][idx] == 'closed':
    #                     available_actions.append(self.action_set['gate'][idx])

    #         # Add turn on cooker to instruction instead of internal mechanism
    #         # for idx, d in enumerate(state_for_llm['distances']['cooker']):
    #         #     if d[0] != 'infinite':
    #         #         available_actions.append(self.action_set['cooking_status'][idx])

    #     elif state_for_llm[self.player_id]['held_object'] == 'onion':
    #         for idx, d in enumerate(state_for_llm['distances']['cooker']):
    #             if d[0] not in ['infinite']:
    #                 available_actions.append(self.action_set['cooker'][idx])

    #         if self.enable_kitchen_counters:
    #             if len(self.empty_kitchen_counters)>0:
                    
    #                 kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
    #                 k_action = self.action_set['kitchen_counter_place_onion'][kidx]
                    
    #                 available_actions.append(k_action)

    #         if 'storage_counter_place_onion' in self.action_set:
    #             for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
    #                 if d[0] not in ['infinite']:
    #                     if state_for_llm['storage_counter_objects'][idx] == 'empty':
    #                         available_actions.append(self.action_set['storage_counter_place_onion'][idx])

    #     elif state_for_llm[self.player_id]['held_object'] == 'plate':
    #         for idx, d in enumerate(state_for_llm['distances']['cooker']):
    #             if d[0] not in ['infinite']:
    #                 available_actions.append(self.action_set['cooked_soup'][idx])
            
    #         if self.enable_kitchen_counters:
    #             if len(self.empty_kitchen_counters)>0:
    #                 kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
    #                 k_action = self.action_set['kitchen_counter_place_plate'][kidx]
                    
    #                 available_actions.append(k_action)

    #         if 'storage_counter_place_plate' in self.action_set:
    #             for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
    #                 if d[0] not in ['infinite']:
    #                     if state_for_llm['storage_counter_objects'][idx] == 'empty':
    #                         available_actions.append(self.action_set['storage_counter_place_plate'][idx])

    #     elif state_for_llm[self.player_id]['held_object'] == 'soup in plate':
    #         for idx, d in enumerate(state_for_llm['distances']['delivery_zone']):
    #             if d[0] not in ['infinite']:
    #                 available_actions.append(self.action_set['delivery_area'][idx])
    #         if self.enable_kitchen_counters:
    #             if len(self.empty_kitchen_counters)>0:
    #                 kidx = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                    
    #                 k_action = self.action_set['kitchen_counter_place_plate'][kidx]
                    
    #                 available_actions.append(k_action)

    #         if 'storage_counter_place_soup' in self.action_set:
    #             for idx, d in enumerate(state_for_llm['distances']['storage_counter']):
    #                 if d[0] not in ['infinite']:
    #                     if state_for_llm['storage_counter_objects'][idx] == 'empty':
    #                         available_actions.append(self.action_set['storage_counter_place_soup'][idx])
    #     return available_actions + self.action_set['wait'] + self.action_set['collision_avoidance']

    def _correct_dish_to_plate(self, state_for_llm):
        if state_for_llm[self.player_id ]['held_object'] == 'dish':
            state_for_llm[self.player_id ]['held_object'] = 'plate'
        
        if state_for_llm[self.other_player_id]['held_object'] == 'dish':
            state_for_llm[self.player_id ]['held_object'] = 'plate'
        return state_for_llm
    
    def _add_history(self):
        description = f'''action history: {', '.join(self.action_history[-5:])}.\n'''
        add_to_dict_list(self.log_csv_dict, f"action_history", self.action_history)
        return description

    def _add_held_object_info(self, state_for_llm):

        description = f'''<Inventory>: I am holding {state_for_llm[self.player_id ]['held_object']}. {self.player_names[self.other_player_id]} is holding {state_for_llm[self.other_player_id ]['held_object']}. '''
        if self.single_agent_ablation:
            description = f'''<Inventory>: I am holding {state_for_llm[self.player_id ]['held_object']}. '''
        
        add_to_dict_list(self.log_csv_dict, f"player_held_object", state_for_llm[self.player_id ]['held_object'])
        add_to_dict_list(self.log_csv_dict, f"other_player_held_object", state_for_llm[self.other_player_id ]['held_object'])
        return description
    
    def _add_kitchen_facility_info_single_agent_ablation(self, state_for_llm):
        # TODO: Add kitchen counter distances to both Bob and Alice's
        self.empty_kitchen_counters = []
        self.empty_kitchen_counter_distances = []
        description = f"<My location information:> "
        for obj_type in ['onion_dispenser', 'plate_dispenser', 'delivery_zone', 'cooker', 'storage_counter', 'gate']:
            for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                if d[0] == 'infinite':
                    description += f"{obj_type[0]}{idx} is inaccessible. "
                elif 'blocked' in d[0]:
                    description += f"{obj_type[0]}{idx} is {d[0]}"
                else:
                    description += f"{obj_type[0]}{idx} is {d[0]} units away. "

                add_to_dict_list(self.log_csv_dict, f"{obj_type[0]}{idx}_distance_from_{self.player_names[self.player_id]}", str(d[0]))

            
        description += f"\n<Environment Details>: "
        for obj_type in ['cooker', 'storage_counter', 'kitchen_counter', 'gate']:
            for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                if obj_type == 'cooker':
                        description += f"c{idx} contains {state_for_llm['num_onions_in_pot'][idx]} out of 3 onions. "
                        add_to_dict_list(self.log_csv_dict, f"c{idx}_num_onions", state_for_llm['num_onions_in_pot'][idx])
                        description += f"c{idx} is {state_for_llm['cooker_status'][idx]}. "
                        add_to_dict_list(self.log_csv_dict, f"c{idx}_cooker_status", state_for_llm['cooker_status'][idx])
                        description += f"soup in c{idx} is {state_for_llm['soup_in_cooker_status'][idx]}. "
                        # if state_for_llm['soup_in_cooker_status'][idx] == 'still cooking':
                        #     description += f"soup in c{idx} needs {state_for_llm['soup_in_cooker_remaining_time'][idx]} timesteps to cook. "
                        add_to_dict_list(self.log_csv_dict, f"c{idx}_soup_in_cooker_status", state_for_llm['soup_in_cooker_status'][idx])
                if self.enable_kitchen_counters:
                    if obj_type == 'kitchen_counter':
                        if state_for_llm['kitchen_counter_objects'][idx] != 'empty':
                            if d[0] == 'infinite':
                                description += f'k{idx} is inaccessible. '
                            elif 'blocked' in d[0]:
                                description += f"k{idx} is {d[0]} " 
                            else:
                                description += f"k{idx} is {d[0]} units away. "
                                description += f"k{idx} contains {state_for_llm['kitchen_counter_objects'][idx]}. " 
                            self.empty_kitchen_counter_distances.append(float('inf'))
                        else:
                            if d[0] in ['infinite'] or 'blocked' in d[0]:
                                self.empty_kitchen_counter_distances.append(float('inf'))
                            else:
                                self.empty_kitchen_counter_distances.append(int(d[0]))
                                self.empty_kitchen_counters.append(f'k{idx}')
                
                if obj_type == 'gate':
                    if d[0] not in ['infinite']:
                        description += f"g{idx} is {state_for_llm['gate_status'][idx]}. "
                        if state_for_llm['gate_status'][idx] == 'open':
                            description += f"g{idx} will stay open for {10 - state_for_llm['gate_open_time'][idx]} timesteps. "

                if self.layout_name in ['forced_coordination', 'counter_circuit_o_1order', 'soup_passing']:   
                    if obj_type == 'storage_counter':
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            description += f"s{idx} is empty. "
                        else:
                            description += f"s{idx} contains {state_for_llm['storage_counter_objects'][idx]}. "
                        add_to_dict_list(self.log_csv_dict, f"s{idx} object", state_for_llm['storage_counter_objects'][idx]) 

                # When there are no kitchen counters:
        if self.enable_kitchen_counters:
            if len(self.empty_kitchen_counter_distances) > 0:
                closest_kitchen_counter = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                distance_to_closest_kitchen_counter = min(self.empty_kitchen_counter_distances)
                # print('Number of kitchen counters: ', len(self.empty_kitchen_counter_distances))
                if distance_to_closest_kitchen_counter != float('inf'):
                    description += f'Closest empty kitchen counter k{closest_kitchen_counter} is {distance_to_closest_kitchen_counter} units away. '

        return description


    def _add_kitchen_facility_info(self, state_for_llm):
        # TODO: Add kitchen counter distances to both Bob and Alice's
        self.empty_kitchen_counters = []
        self.empty_kitchen_counter_distances = []
        description = f"<My location information:> "
        for obj_type in ['onion_dispenser', 'plate_dispenser', 'delivery_zone', 'cooker', 'storage_counter', 'gate']:
            for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                if d[0] == 'infinite':
                    description += f"{obj_type[0]}{idx} is inaccessible. "
                elif 'blocked' in d[0]:
                    description += f"{obj_type[0]}{idx} is {d[0]} by {self.player_names[self.other_player_id]}. "
                else:
                    description += f"{obj_type[0]}{idx} is {d[0]} units away. "

                add_to_dict_list(self.log_csv_dict, f"{obj_type[0]}{idx}_distance_from_{self.player_names[self.player_id]}", str(d[0]))
    
        if not self.single_agent_ablation:
            description += f"\n<{self.player_names[self.other_player_id]}'s location information>: "
            for obj_type in ['onion_dispenser', 'plate_dispenser', 'delivery_zone', 'cooker', 'storage_counter', 'gate']:
                for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                    if d[1] == 'infinite':
                        description += f"{obj_type[0]}{idx} is inaccessible. "
                    elif 'blocked' in d[1]:
                        description += f"{obj_type[0]}{idx} is {d[0]} by {self.player_names[self.player_id]}. "  
                    else:
                        description += f"{obj_type[0]}{idx} is {d[1]} units away. "
                    
                    add_to_dict_list(self.log_csv_dict, f"{obj_type[0]}{idx}_distance_from_{self.player_names[self.other_player_id]}", str(d[1]))
            
        description += f"\n<Environment Details>: "
        for obj_type in ['cooker', 'storage_counter', 'kitchen_counter', 'gate']:
            for idx, d in enumerate(state_for_llm['distances'][obj_type]):
                if obj_type == 'cooker':
                        description += f"c{idx} contains {state_for_llm['num_onions_in_pot'][idx]} out of 3 onions. "
                        add_to_dict_list(self.log_csv_dict, f"c{idx}_num_onions", state_for_llm['num_onions_in_pot'][idx])
                        description += f"c{idx} is {state_for_llm['cooker_status'][idx]}. "
                        add_to_dict_list(self.log_csv_dict, f"c{idx}_cooker_status", state_for_llm['cooker_status'][idx])
                        description += f"soup in c{idx} is {state_for_llm['soup_in_cooker_status'][idx]}. "
                        # if state_for_llm['soup_in_cooker_status'][idx] == 'still cooking':
                        #     description += f"soup in c{idx} needs {state_for_llm['soup_in_cooker_remaining_time'][idx]} timesteps to cook. "
                        add_to_dict_list(self.log_csv_dict, f"c{idx}_soup_in_cooker_status", state_for_llm['soup_in_cooker_status'][idx])
                if self.enable_kitchen_counters:
                    if obj_type == 'kitchen_counter':
                        if state_for_llm['kitchen_counter_objects'][idx] != 'empty':
                            if d[0] == 'infinite':
                                description += f'k{idx} is inaccessible. '
                            elif 'blocked' in d[0]:
                                description += f"k{idx} is {d[0]} by {self.player_names[int(self.other_player_id)]}. " 
                            else:
                                description += f"k{idx} is {d[0]} units away. "
                                description += f"k{idx} contains {state_for_llm['kitchen_counter_objects'][idx]}. " 
                            self.empty_kitchen_counter_distances.append(float('inf'))
                        else:
                            if d[0] in ['infinite'] or 'blocked' in d[0]:
                                self.empty_kitchen_counter_distances.append(float('inf'))
                            else:
                                self.empty_kitchen_counter_distances.append(int(d[0]))
                                self.empty_kitchen_counters.append(f'k{idx}')
                
                if obj_type == 'gate':
                    if d[0] not in ['infinite']:
                        description += f"g{idx} is {state_for_llm['gate_status'][idx]}. "
                        if state_for_llm['gate_status'][idx] == 'open':
                            description += f"g{idx} will stay open for {10 - state_for_llm['gate_open_time'][idx]} timesteps. "

                if self.layout_name in ['forced_coordination', 'counter_circuit_o_1order', 'soup_passing']:   
                    if obj_type == 'storage_counter':
                        if state_for_llm['storage_counter_objects'][idx] == 'empty':
                            description += f"s{idx} is empty. "
                        else:
                            description += f"s{idx} contains {state_for_llm['storage_counter_objects'][idx]}. "
                        add_to_dict_list(self.log_csv_dict, f"s{idx} object", state_for_llm['storage_counter_objects'][idx]) 

                # When there are no kitchen counters:
        if self.enable_kitchen_counters:
            if len(self.empty_kitchen_counter_distances) > 0:
                closest_kitchen_counter = self.empty_kitchen_counter_distances.index(min(self.empty_kitchen_counter_distances))
                distance_to_closest_kitchen_counter = min(self.empty_kitchen_counter_distances)
                # print('Number of kitchen counters: ', len(self.empty_kitchen_counter_distances))
                if distance_to_closest_kitchen_counter != float('inf'):
                    description += f'Closest empty kitchen counter k{closest_kitchen_counter} is {distance_to_closest_kitchen_counter} units away. '

        return description
    
    def _parse_and_add_dialog(self, other_player_message):
        description = ''
        if other_player_message != '':
            description += f' {self.player_names[self.other_player_id]} says: {other_player_message}'
            
        add_to_dict_list(self.log_csv_dict, f"other_player_message", other_player_message)
        
        return description

    def infer_partner_state(self, description):
        partner_inference_message = self.pi_message + [{"role": "user", "content": f"{description}"}]
        epistemic_response_string = self.llm.inference_fn(partner_inference_message)
        print(f"{bcolors.OKGREEN}PARTNER INFERENCE: {epistemic_response_string}{bcolors.ENDC}")
        return epistemic_response_string

    def _state_to_description(self, state_for_llm, other_player_message):
        print('STATE FOR LLM: ', state_for_llm)
        state_for_llm = self._correct_dish_to_plate(state_for_llm)
        description = self._add_history() 
        # Add state information in natural language 
        description += self._add_held_object_info(state_for_llm)
        if not self.single_agent_ablation:
            description += self._add_kitchen_facility_info(state_for_llm)
        else:
            description += self._add_kitchen_facility_info_single_agent_ablation(state_for_llm)


        # get available actions based on current state and add the information to the description
        self.available_actions_list = self._get_available_actions(state_for_llm, None)
        # self.partner_inference_string = self.infer_partner_state(description)
        # description += self.partner_inference_string
        available_actions = ', '.join(self.available_actions_list)
        description += f"\nAvailable Actions:\n[{available_actions}]"
        add_to_dict_list(self.log_csv_dict, f"available_actions", " | ".join(self.available_actions_list))

        return description

    # def run_llm_inference(self, messages):
    #     completion = self.client.chat.completions.create(
    #         messages = messages,
    #         model=self.model_name,
    #         temperature=self.temperature,
    #         top_p=self.top_p,
    #         frequency_penalty=self.frequency_penalty,
    #         presence_penalty=self.presence_penalty
    #     )
    #     total_tokens = completion.usage.total_tokens
    #     if '4' in self.model:
    #         cur_cost =  0.015 * (total_tokens / 1000)
    #     else:
    #         cur_cost = 0.00075 * (total_tokens / 1000)
    #     self.cost += cur_cost 
    #     print(f"COST SO FAR for {self.player_names[self.player_id]}: {self.cost} USD")
    #     return completion.choices[0].message.content


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
            for action in self.available_actions_list:
                if selected_match.lower() in action.lower():
                    return action 
            selected_move, score = process.extractOne(selected_match, self.available_actions_list)
        else:
            selected_move = np.random.choice(self.available_actions_list)
        return selected_move
    
    def get_player_action(self, state_for_llm, other_player_message):
        state_description = self._state_to_description(state_for_llm, other_player_message)
        response_string = ''
        message = ''
        print(f"{bcolors.HEADER}CURRENT PLAYER: {self.player_names[self.player_id]}{bcolors.ENDC}")
        print(f"{bcolors.FAIL}{state_description}{bcolors.ENDC}")
        
        if self.DEBUG:
            selected_action = 'wait.'
            selected_action = self.replay_actions.pop(0)
            return selected_action, '' 
        elif len(self.player_actions) > 0:
            selected_action = self.player_actions.pop(0)
            message = ''
            add_to_dict_list(self.log_csv_dict, 'full_state_description', state_description)
            add_to_dict_list(self.log_csv_dict, 'selected_action', selected_action)
            add_to_dict_list(self.log_csv_dict, 'llm_response', 'LLM NOT USED') 

            self.action_history.append(selected_action)
            if self.save_trajectory:
                np.save(self.trajectory_path,self.action_history)
        else:
            try: 
                # cache which contains the selected action as value 
                state_only_desc = state_description[state_description.find('state:'):]
                if self.enable_cache and state_only_desc in self.cache:
                    action, message = self.cache[state_only_desc]
                else:
                    if len(self.available_actions_list) > 1:
                        # Trick to avoid too many calls to the API in Forced Coordination 
                        if set(self.available_actions_list) == set(['AAA. wait.', 'BBB. move away.']):
                            action = 'A. wait.'
                        else:
                            messages = self.message + [{"role": "user", "content": state_description}]
                            response_string = self.llm.inference_fn(messages=messages)
                            print(f'''{bcolors.WARNING}LLM RESPONSE: {response_string}{bcolors.ENDC}''')
                            action = self.find_best_match(response_string)


                            ### VERIFICATION ###
                            # verification_response_string = ''
                            # verifier_responses = []
                            # verifier_description = f"State: {state_description.replace(self.partner_action_inference_string, '')}\n\n My Solution: {action}. Think step by step. Think about safety, think about rules, think about conventions. "
                            # print(f'''{bcolors.WARNING}VERIFIER INPUT: {verifier_description}{bcolors.ENDC}''')
                            # self.verifier_message = self.verifier_base_message + [{"role": "user", "content": verifier_description}]
                            # verification_response_string = self.llm.inference_fn(self.verifier_message)
                            # self.num_api_calls += 1
                            # verifier_responses.append(verification_response_string)
                            # print(f'''{bcolors.OKCYAN}VERIFICATION RESPONSE: {verification_response_string}{bcolors.ENDC}''')
                            # counter = 0 
                            # while 'verification: okay' not in verification_response_string.lower(): 
                            #     if action in self.available_actions_list:  
                            #         self.available_actions_list.remove(action)
                            #     counter += 1
                            #     self.generator_message.append({"role": "assistant", "content": response_string})
                            #     updated_generator_message = f"Your selected action: {action} is not appropriate. {verification_response_string}. Please choose another action. List of Available Actions:\n{self.available_actions_list}"


                            #     messages.append({"role": "user", "content": updated_generator_message})
                                
                            #     response_string = self.llm.inference_f(messages)
                            #     self.num_api_calls += 1
                            #     print(f"{bcolors.WARNING}LLM CORRECTED RESPONSE: {response_string}{bcolors.ENDC}") 
                            #     action = self.find_best_match(response_string)
    
                            #     self.verifier_message[-1]["content"] = f"State: {state_description.replace(self.partner_action_inference_string, '')}\n\n My Solution: {action}. Think step by step. Think about safety, think about rules, think about conventions. "

                            #     verification_response_string = self.llm.inference_f(self.verifier_message)
                            #     self.num_api_calls += 1
                            #     verifier_responses.append(verification_response_string) 
                            #     print(f'''{bcolors.OKCYAN}VERIFICATION RESPONSE: {verification_response_string}{bcolors.ENDC}''')
                                
                            # add_to_dict_list(self.log_csv_dict, 'VERIFICATION Response', ' ***** '.join(verifier_responses)) 

                            # verification_string = self.llm.inference_fn(messages=self.verifier_message + [{"role": "user", "content": f"My selected Action: {action}"}])
                            # print('VERIFIER RESPONSE: ', verification_string)
                            # while 'Verification: Okay' not in verification_string:
                            #     messages += [{"role": "assistant", "content": response_string}] 
                            #     messages += [{"role": "user", "content": verification_string}]
                            #     response_string = self.llm.inference_fn(messages=messages)
                            #     print(f'''{bcolors.WARNING}LLM RESPONSE: {response_string}{bcolors.ENDC}''')
                            #     action = self.find_best_match(response_string)
                                
                            #     verification_string = self.llm.inference_fn(messages=self.verifier_message + [{"role": "user", "content": f"My selected Action: {action}"}])

                    else:
                        action = 'wait.'
                    self.cache[state_only_desc] = (action, message)
                    if self.write_to_cache:
                        with open(self.cache_save_path, 'w') as f:
                            json.dump(self.cache, f)
                print(f"{bcolors.OKBLUE}Number of API calls made by player {self.player_id}: {bcolors.ENDC}", self.num_api_calls)
                # if action in self.all_actions:
                if action in self.available_actions_list:
                    selected_action = action 
                else:
                    print("WARNING: LLM returned an action that is not in the defined action set. ")
                    selected_action = 'wait.'

            except Exception as e:
                selected_action = 'wait.' 
                print(f'Failed to get response from openai api for player {self.player_id} due to {e}')
                # sys.exit(0)
                time.sleep(1.)
                pass
            add_to_dict_list(self.log_csv_dict, 'full_state_description', state_description)
            add_to_dict_list(self.log_csv_dict, 'selected_action', selected_action)
            add_to_dict_list(self.log_csv_dict, 'llm_response', response_string) 
            
            df = pd.DataFrame(self.log_csv_dict)
            df.to_csv(f"game_logs/{self.experiment_type}/{self.layout_name}/{self.trial_note}_player_{self.player_id}_{self.time_stamp}.csv") 
            
            self.action_history.append(selected_action)
            if self.save_trajectory:
                np.save(self.trajectory_path,self.action_history)

        print('SELECTED ACTION: ', selected_action) 
        
        return selected_action, message 

