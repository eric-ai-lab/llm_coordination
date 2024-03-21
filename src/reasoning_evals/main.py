# Requirements 


# 1. Don't have to write a different function for each game - there should be unified inference with the excel sheet format 
# 2. Don't want separate functions for different models, the models and all its essentials should be passed as arguments wherever needed 
# 3. Save results 
# 4. Score the model after running all inference - separate fuzzy scoring algorithm 

import os 
from openai import OpenAI, AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import re 
import numpy as np 
import json 
import pandas as pd 
from tqdm import tqdm
from dataclasses import dataclass 
from fuzzywuzzy import process
import time 
from datetime import datetime


@dataclass
class EvalDataPoint:
    '''Class for managing evaluation scenarios'''
    concept: str 
    game_desc: str 
    ec_directive: str 
    tom_directive: str 
    jp_directive: str 
    state_desc: str 
    ec_question: str 
    tom_question: str 
    jp_question: str 
    jp_answer_ordinal: str 
    jp_answer_desc: str
    tom_answer_ordinal: str
    tom_answer_desc: str
    ec_answer_ordinal: str
    ec_answer_desc: str

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
        self.cost = 0
        if not api_server:
            if self.model_type == 'openai':
                self.akey = os.environ['API_KEY']
                self.org = os.environ['ORGANIZATION']
                self.client = OpenAI(api_key = self.akey, organization = self.org)
                self.inference_fn = self.run_openai_inference
            elif self.model_type == 'mistral':
                self.device = "cuda" # the device to load the model onto
                os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', cache_dir=cache_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
                self.inference_fn = self.run_mistral_inference
            elif self.model_type == 'vicuna':
                os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', cache_dir=cache_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
                self.inference_fn = self.run_vicuna_inference
        else:
            if self.model_type == 'openai':
                self.akey = os.getenv("AZURE_OPENAI_ENDPOINT")
                self.org = os.getenv("AZURE_OPENAI_API_KEY")
                # self.client = OpenAI(api_key = self.akey, organization = self.org)
                self.client = AzureOpenAI(
                    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2023-05-15"
                )
                self.inference_fn = self.run_openai_inference
            else:
                self.akey = "EMPTY"
                self.base_url = "http://localhost:8000/v1"
                self.client = OpenAI(
                    api_key=self.akey,
                    base_url=self.base_url,
                )
                self.inference_fn = self.run_openai_inference

                        
    
    def inference_fn(self):
        if 'mistral' in self.model:
            return self.run_mistral_inference
        elif 'vicuna' in self.model:
            return self.run_vicuna_inference
        

    def run_openai_inference(self, messages):
        completion = self.client.chat.completions.create(
            messages = messages,
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty
        )
        # print("INFERENCE STRING: ", completion.choices[0].message.content)
        total_tokens = completion.usage.total_tokens
        cur_cost =  0.015 * (total_tokens / 1000)
        self.cost += cur_cost 
        # print(f"COST SO FAR: {self.cost} USD")
        return completion.choices[0].message.content

        

    def run_mistral_inference(self, messages):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature=self.temperature)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded[0].split('[/INST] ')[-1]

    def run_vicuna_inference(self, messages):
        def apply_vicuna_chat_template(messages): 
            output = f'''A chat between a curious user and an artificial intelligence assistant. 
            
            '''
            participants = ["USER", "ASSISTANT"]
            for i, m in enumerate(messages):
                turn = participants[i%2]
                if turn == 'USER':
                    output += f"{turn}: {m['content']}\n"
                else:
                    output += f"{turn}: {m['content']}</s>\n"
            output += f"ASSISTANT: "
            return self.tokenizer.encode(output, return_tensors="pt")
        
        encodeds = apply_vicuna_chat_template(messages)
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True, temperature=0.7)
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded[0].split("ASSISTANT: ")[-1]


class TestLLMCoordination:
    def __init__(self, df, game_name, model, model_type, log_file='logs.json', num_trials=3):
        self.game_name = game_name 
        self.df = df 
        self.model = model
        self.model_type = model_type 
        self.llm = LLMManager(model_name=self.model, model_type=self.model_type, cache_dir='/data4/saaket/cache/hub')
        self.log_file = log_file
        self.num_trials = num_trials
        self.issues = []
        self.log = {
            'EC_ANSWERS': [],
            'TOM_ANSWERS': [],
            'JP_ANSWERS': []
        }

    def clear_logs(self):
        self.log = {
            'EC_ANSWERS': [],
            'TOM_ANSWERS': [],
            'JP_ANSWERS': []
        }

    def __len__(self):
        return len(self.df)

    def find_answer_fuzzy(self, question_string, inference_string):
        ## Need to improve this a bit more 
        selected_ordinal = 'A'
        def get_answer_ordinal(inference_string):
            match = re.search(r'([A-Z])[\.\)]', inference_string)
            if match:
                return match.group(1)
            else:
                return None
    
        def get_answer_text(inference_string):
            # print(inference_string)
            action_regex = r"Action:\s*(.*)"
            answer_regex = r"Answer:\s*(.*)"
            match = re.search(action_regex, inference_string)
            if match:
                return match.group(1)
            
            match = re.search(answer_regex, inference_string)
            if match:
                return match.group(1)
            
            return None
        
        # If there is only one instance of Action: action then we get the action string 
        answer_text = get_answer_text(inference_string)
        # extract ordinal from the answer 
        if answer_text == None:
            answer_text = inference_string

        selected_ordinal = get_answer_ordinal(answer_text)
        
        if selected_ordinal is None: 
            action_list = []
            # print("QUESTION STRING: ", question_string)
            if 'actions' in question_string.strip().lower():
                action_list = question_string.strip().lower().split('available actions:')[-1].split('\n')
            elif 'answer' in question_string.strip().lower():
                action_list = question_string.strip().lower().split('available answers:')[-1].split('\n') 
                # print(action_list)
            if '' in action_list:
                action_list = [a for a in action_list if a != '']
            if ' ' in action_list:
                action_list = [a for a in action_list if a != ' ']
            action_list = [re.sub(r'^[a-z]\.\s*', '', action).strip() for action in action_list]
            action_start = answer_text.find("Action:")
            if action_start == -1:
                action_start = answer_text.find("Answer:")

            if action_start == -1:
                answer_text = re.sub(r'^[a-z]\.\s*', '', answer_text).strip()
                best_match, score = process.extractOne(answer_text, action_list)
                selected_ordinal = chr(action_list.index(best_match) + 65)
            else:
                answer_text = answer_text[action_start + 7:].strip()
                answer_text = re.sub(r'^[a-z]\.\s*', '', answer_text).strip()
                best_match, score = process.extractOne(answer_text, action_list)
                selected_ordinal = chr(action_list.index(best_match) + 65)
        
        return selected_ordinal
    
    def run_inference(self, directive, game_desc, question):
        if self.llm.model_type == 'openai':
            messages = [
                {"role": "system", "content": directive},
                {"role": "user", "content": game_desc},
                {"role": "assistant", "content": 'Got it!'},
               
                {"role": "user", "content": question},
            ]
        elif self.model_type == 'mistral':
            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": game_desc + '\n' + directive},
                {"role": "assistant", "content": 'Got it! Please provide me with the scenario.'},
                {"role": "user", "content": question},
            ]
        elif self.model_type == 'vicuna' or self.llm.model_type == 'llama':

            messages = [
                {"role": "system", "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."},
                {"role": "user", "content": game_desc},
                {"role": "assistant", "content": 'I understand the game. Please tell me how I can help'},
                {"role": "user", "content": directive},
                {"role": "assistant", "content": 'I understand. Plase provide the scenario.'},
                {"role": "user", "content": question + ' Think step by step. '}, 

            ]
        

        inference = self.llm.inference_fn(messages)
        # print(f"INFERENCE: ", inference)
        return inference



    def run_ec_inference(self, scenario):
        # Run inference for EC
        if self.llm.model_type == 'openai':
            ec_messages = [
                {"role": "system", "content": scenario.ec_directive},
                {"role": "user", "content": scenario.game_desc},
                {"role": "assistant", "content": 'Got it!'},
                {"role": "user", "content": scenario.ec_question},
            ]
        else:
            ec_messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": scenario.game_desc + '\n' + scenario.ec_directive},
                {"role": "assistant", "content": 'Got it! Please provide me with the scenario.'},
                {"role": "user", "content": scenario.ec_question},
            ]
        return self.llm.inference_fn(ec_messages)

    def run_tom_inference(self,scenario):
        # Run inference for TOM
        if self.llm.model_type == 'openai':
            tom_messages = [
                {"role": "system", "content": scenario.tom_directive},
                {"role": "user", "content": scenario.game_desc},
                {"role": "assistant", "content": 'Got it!'},
                {"role": "user", "content": scenario.tom_question},
            ]
        else:
            tom_messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": scenario.game_desc + '\n' + scenario.tom_directive},
                {"role": "assistant", "content": 'Got it! Please provide me with the scenario.'},
                {"role": "user", "content": scenario.tom_question},
            ]
        return self.llm.inference_fn(tom_messages)
    
    def run_jp_inference(self, scenario):
        # Run inference for JP
        if self.llm.model_type == 'openai':
            jp_messages = [
                {"role": "system", "content": scenario.jp_directive},
                {"role": "user", "content": scenario.game_desc},
                {"role": "assistant", "content": 'Got it!'},
                {"role": "user", "content": scenario.jp_question},
            ]
        else:
            jp_messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": scenario.game_desc + '\n' + scenario.jp_directive},
                {"role": "assistant", "content": 'Got it! Please provide me with the scenario.'},
                {"role": "user", "content": scenario.jp_question},
            ]
        return self.llm.inference_fn(jp_messages)

    def save_logs(self, trial_num):
        # Writing the dictionary to a file in JSON format
        # time_stamp = datetime.now() 
        save_file = self.log_file.replace('.','') + f"{trial_num}.json"
      
        print(save_file)
        with open(save_file, 'w') as file:
            json.dump(self.log, file, indent=4)  # `indent` for pretty-printing
    
    def random_inference(self, question):
        # Find the number of options in the question and select a random one
        action_list = []
        if 'actions' in question.strip().lower():
            action_list = question.strip().lower().split('available actions:')[-1].split('\n')
        elif 'answer' in question.strip().lower():
            action_list = question.strip().lower().split('available answers:')[-1].split('\n')
        
        action_items = len(action_list)
        options = [chr(65 + i) for i in range(action_items)]
        return np.random.choice(options)
            
        

    
    def test_one_scenario(self, idx): 
        scenario_row = self.df.iloc[idx]
        current_game = scenario_row['Game']
        scenario = EvalDataPoint(concept=scenario_row['Concept'],
                                    game_desc=scenario_row['Game Description'],
                                    ec_directive=scenario_row['EC Directive'],
                                    tom_directive=scenario_row['TOM Directive'],
                                    jp_directive=scenario_row['JP Directive'],
                                    state_desc=scenario_row['State Description'],
                                    ec_question=scenario_row['EC Question'],
                                    tom_question=scenario_row['TOM Question'],
                                    jp_question=scenario_row['JP Question'],
                                    jp_answer_ordinal=scenario_row['JP Answer Ordinal'],
                                    jp_answer_desc=scenario_row['JP Answer'], 
                                    tom_answer_ordinal=scenario_row['TOM Answer Ordinal'],
                                    tom_answer_desc=scenario_row['TOM Answer'],
                                    ec_answer_ordinal=scenario_row['EC Answer Ordinal'],
                                    ec_answer_desc=scenario_row['EC Answer'])
        if self.llm.model_type == 'baseline':
            if self.llm.model_name == 'random':
                ec_answer_llm = self.random_inference(scenario.ec_question)
                tom_answer_llm = self.random_inference(scenario.tom_question)
                jp_answer_llm = self.random_inference(scenario.jp_question)
        
        else:        
            ec_inference = self.run_inference(scenario.ec_directive, scenario.game_desc, scenario.ec_question) 
            self.log['EC_ANSWERS'].append(ec_inference)
            try:
                ec_answer_llm = self.find_answer_fuzzy(scenario.ec_question, ec_inference)
            except:
                self.issues.append({'type': 'EC','question': scenario.ec_question, 'inference': ec_inference, 'game': current_game})
                ec_answer_llm = '2'

            tom_inference = self.run_inference(scenario.tom_directive, scenario.game_desc, scenario.tom_question)
            self.log['TOM_ANSWERS'].append(tom_inference)
            try:
                tom_answer_llm = self.find_answer_fuzzy(scenario.tom_question,tom_inference)
            except:
                self.issues.append({'type': 'TOM','question': scenario.tom_question, 'inference': tom_inference, 'game': current_game})
                tom_answer_llm = '2'
                
            jp_inference = self.run_inference(scenario.jp_directive, scenario.game_desc, scenario.jp_question)
            self.log['JP_ANSWERS'].append(jp_inference)
            try:
                jp_answer_llm = self.find_answer_fuzzy(scenario.jp_question, jp_inference)
            except:
                self.issues.append({'type': 'TOM', 'question': scenario.jp_question, 'inference': jp_inference, 'game': current_game})
                jp_answer_llm = '2'

        scores = [1 if ec_answer_llm in scenario.ec_answer_ordinal.split(',') else 0, 
                  1 if tom_answer_llm in scenario.tom_answer_ordinal.split(',') else 0, 
                  1 if jp_answer_llm in scenario.jp_answer_ordinal.split(',') else 0] 

        return np.array(scores), current_game
    
    def evaluate_llm(self):
        scores = []
        game_wise_scores = {'Overcooked': [], 'Hanabi': [], 'CollabGames': []}
        for t_num in range(self.num_trials):
            results = []
            game_wise_results = {'Overcooked': [], 'Hanabi': [], 'CollabGames': []}
            print("CONDUCTING TRIAL NUMBER: ", t_num)
            for sc in tqdm(range(len(self.df))):
                res, current_game = self.test_one_scenario(sc)
                results.append(res)
                game_wise_results[current_game].append(res)
                # time.sleep(0.5)
                # self.save_logs(t_num)
                # print(game_wise_results)
                
            
            results = np.array(results)
            game_wise_score = {game: np.mean(np.array(game_wise_results[game]), axis=0) for game in game_wise_results} 
            # print("GAME WISE SCORE: ", game_wise_score)
            score = results.mean(axis=0)
            for game in game_wise_scores:
                game_wise_scores[game].append(game_wise_score[game])
            print("GAME WISE SCORES: ", game_wise_scores)
            scores.append(score)
            self.save_logs(t_num)
            self.clear_logs()
            # temp_results = {'accuracy': np.mean(np.array(scores), axis=0), 
            #     'standard error': (np.array(scores).std(axis=0) / np.sqrt(self.num_trials)),
            #     'Overcooked accuracy': np.mean(np.array(game_wise_scores['Overcooked']), axis=0),
            #     'Overcooked standard error': np.std(np.array(game_wise_scores['Overcooked']), axis=0) / np.sqrt(self.num_trials),
            #     'Hanabi accuracy': np.mean(np.array(game_wise_scores['Hanabi']), axis=0),
            #     'Hanabi standard error': np.std(np.array(game_wise_scores['Hanabi']), axis=0) / np.sqrt(self.num_trials),
            #     'CollabGames accuracy': np.mean(np.array(game_wise_scores['CollabGames']), axis=0),
            #     'CollabGames standard error': np.std(np.array(game_wise_scores['CollabGames']), axis=0) / np.sqrt(self.num_trials)
            #     }
            # formatted_temp_results = format_results(temp_results)

            # write_results_to_file(model_nm=self.llm.model_name, timestamp=datetime.now(), model_type=self.llm.model_type, result_table=formatted_temp_results, trial_num=t_num)
            # time.sleep(1.)
        return {'accuracy': np.mean(np.array(scores), axis=0), 
                'standard error': (np.array(scores).std(axis=0) / np.sqrt(self.num_trials)),
                'Overcooked accuracy': np.mean(np.array(game_wise_scores['Overcooked']), axis=0),
                'Overcooked standard error': np.std(np.array(game_wise_scores['Overcooked']), axis=0) / np.sqrt(self.num_trials),
                'Hanabi accuracy': np.mean(np.array(game_wise_scores['Hanabi']), axis=0),
                'Hanabi standard error': np.std(np.array(game_wise_scores['Hanabi']), axis=0) / np.sqrt(self.num_trials),
                'CollabGames accuracy': np.mean(np.array(game_wise_scores['CollabGames']), axis=0),
                'CollabGames standard error': np.std(np.array(game_wise_scores['CollabGames']), axis=0) / np.sqrt(self.num_trials)
                }
    
def write_results_to_file(model_nm, timestamp, model_type, result_table, trial_num, game_name='all'):
    with open(f'gpt_4_temp_results/{model_nm}_{timestamp}_trial_{trial_num}_output.txt', 'w') as f:
        f.write('TEST FILE: ' + str(game_name) + '\n')
        f.write('MODEL: ' + str(model) + '\n')
        f.write('MODEL TYPE: ' + str(model_type) + '\n')
        f.write(f"Environment Comprehension Score: {result_table['EC_SCORE']} +/- {result_table['EC_SE']}\n")
        f.write(f"Theory of Mind Score: {result_table['TOM_SCORE']} +/- {result_table['TOM_SE']}\n")
        f.write(f"Joint Planning Score: {result_table['JP_SCORE']} +/- {result_table['JP_SE']}\n")

        # Game Wise Scores
        f.write(f"Overcooked Environment Comprehension Score: {result_table['Overcooked accuracy'][0]} +/- {result_table['Overcooked standard error'][0]}\n")
        f.write(f"Overcooked Theory of Mind Score: {result_table['Overcooked accuracy'][1]} +/- {result_table['Overcooked standard error'][1]}\n")
        f.write(f"Overcooked Joint Planning Score: {result_table['Overcooked accuracy'][2]} +/- {result_table['Overcooked standard error'][2]}\n")

        f.write(f"Hanabi Environment Comprehension Score: {result_table['Hanabi accuracy'][0]} +/- {result_table['Hanabi standard error'][0]}\n")
        f.write(f"Hanabi Theory of Mind Score: {result_table['Hanabi accuracy'][1]} +/- {result_table['Hanabi standard error'][1]}\n")
        f.write(f"Hanabi Joint Planning Score: {result_table['Hanabi accuracy'][2]} +/- {result_table['Hanabi standard error'][2]}\n")

        f.write(f"CollabGames Environment Comprehension Score: {result_table['CollabGames accuracy'][0]} +/- {result_table['CollabGames standard error'][0]}\n")
        f.write(f"CollabGames Theory of Mind Score: {result_table['CollabGames accuracy'][1]} +/- {result_table['CollabGames standard error'][1]}\n")
        f.write(f"CollabGames Joint Planning Score: {result_table['CollabGames accuracy'][2]} +/- {result_table['CollabGames standard error'][2]}\n")
        f.write(f"Problems: {evaluator.issues}\n")


def format_results(results):
    return {'EC_SCORE': results['accuracy'][0],
            'TOM_SCORE': results['accuracy'][1],
            'JP_SCORE': results['accuracy'][2],
            'EC_SE': results['standard error'][0],
            'TOM_SE': results['standard error'][1],
            'JP_SE': results['standard error'][2],
            'Overcooked accuracy': results['Overcooked accuracy'],
            'Overcooked standard error': results['Overcooked standard error'],
            'Hanabi accuracy': results['Hanabi accuracy'],
            'Hanabi standard error': results['Hanabi standard error'],
            'CollabGames accuracy': results['CollabGames accuracy'],
            'CollabGames standard error': results['CollabGames standard error']
            }

def extract_test_df(df, n):
    # Extract n samples from each 'Game' from the dataframe
    return df.groupby('Game').apply(lambda x: x.sample(n)).reset_index(drop=True)
    
TEST = False 
if __name__ == '__main__':
    df = pd.read_csv('~/llm_coordination_suite/single_turn_evals/data/single_turn_trials_march_2.csv')
    # if TEST:
    #     df = extract_test_df(df, 2)
    game_name = 'all'
    # if 'Game' not in df:
    #     df['Game'] = game_name
    # model = 'mistralai/Mistral-7B-Instruct-v0.2'
    # model_type = 'mistral'
    # model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    # model_type = 'mistral'
    # model = 'lmsys/vicuna-7b-v1.5'
    # model_type = 'vicuna'
    # model = 'lmsys/vicuna-13b-v1.5'
    # model_type = 'vicuna'
    # model = 'lmsys/vicuna-33b-v1.3'
    # model_type = 'vicuna'
    # model = 'meta-llama/Llama-2-13b-chat-hf'
    # model_type = 'llama'
    # model = 'meta-llama/Llama-2-70b-chat-hf'
    # model_type = 'llama'
    # model = 'random'
    # model_type = 'baseline'
    model = 'gpt-35-turbo'
    model_type = 'openai'
    # model = 'gpt-4-0125'
    # model_type = 'openai'
    timestamp = datetime.now()
    if '/' in model:
        model_nm = model.split('/')[-1]
    else:
        model_nm = model
    evaluator = TestLLMCoordination(df, game_name, model, model_type, f'/home/anthony/llm_coordination/single_turn_evals/logs/{game_name}_{model_type}_{model_nm}')
    results = evaluator.evaluate_llm()
    result_table = format_results(results)
    print('TEST FILE: ', game_name)
    print('MODEL: ', model)
    print('MODEL TYPE: ', model_type)
    print(f"Environment Comprehension Score: {result_table['EC_SCORE']} +/- {result_table['EC_SE']}")
    print(f"Theory of Mind Score: {result_table['TOM_SCORE']} +/- {result_table['TOM_SE']}")
    print(f"Joint Planning Score: {result_table['JP_SCORE']} +/- {result_table['JP_SE']}")

    # Game Wise Scores
    print(f"Overcooked Environment Comprehension Score: {result_table['Overcooked accuracy'][0]} +/- {result_table['Overcooked standard error'][0]}")
    print(f"Overcooked Theory of Mind Score: {result_table['Overcooked accuracy'][1]} +/- {result_table['Overcooked standard error'][1]}")
    print(f"Overcooked Joint Planning Score: {result_table['Overcooked accuracy'][2]} +/- {result_table['Overcooked standard error'][2]}")

    print(f"Hanabi Environment Comprehension Score: {result_table['Hanabi accuracy'][0]} +/- {result_table['Hanabi standard error'][0]}")
    print(f"Hanabi Theory of Mind Score: {result_table['Hanabi accuracy'][1]} +/- {result_table['Hanabi standard error'][1]}")
    print(f"Hanabi Joint Planning Score: {result_table['Hanabi accuracy'][2]} +/- {result_table['Hanabi standard error'][2]}")

    print(f"CollabGames Environment Comprehension Score: {result_table['CollabGames accuracy'][0]} +/- {result_table['CollabGames standard error'][0]}")
    print(f"CollabGames Theory of Mind Score: {result_table['CollabGames accuracy'][1]} +/- {result_table['CollabGames standard error'][1]}")
    print(f"CollabGames Joint Planning Score: {result_table['CollabGames accuracy'][2]} +/- {result_table['CollabGames standard error'][2]}")
    print(f"Problems: {evaluator.issues}")
    with open(f'{model_nm}_{timestamp}_output.txt', 'w') as f:
        f.write('TEST FILE: ' + str(game_name) + '\n')
        f.write('MODEL: ' + str(model) + '\n')
        f.write('MODEL TYPE: ' + str(model_type) + '\n')
        f.write(f"Environment Comprehension Score: {result_table['EC_SCORE']} +/- {result_table['EC_SE']}\n")
        f.write(f"Theory of Mind Score: {result_table['TOM_SCORE']} +/- {result_table['TOM_SE']}\n")
        f.write(f"Joint Planning Score: {result_table['JP_SCORE']} +/- {result_table['JP_SE']}\n")

        # Game Wise Scores
        f.write(f"Overcooked Environment Comprehension Score: {result_table['Overcooked accuracy'][0]} +/- {result_table['Overcooked standard error'][0]}\n")
        f.write(f"Overcooked Theory of Mind Score: {result_table['Overcooked accuracy'][1]} +/- {result_table['Overcooked standard error'][1]}\n")
        f.write(f"Overcooked Joint Planning Score: {result_table['Overcooked accuracy'][2]} +/- {result_table['Overcooked standard error'][2]}\n")

        f.write(f"Hanabi Environment Comprehension Score: {result_table['Hanabi accuracy'][0]} +/- {result_table['Hanabi standard error'][0]}\n")
        f.write(f"Hanabi Theory of Mind Score: {result_table['Hanabi accuracy'][1]} +/- {result_table['Hanabi standard error'][1]}\n")
        f.write(f"Hanabi Joint Planning Score: {result_table['Hanabi accuracy'][2]} +/- {result_table['Hanabi standard error'][2]}\n")

        f.write(f"CollabGames Environment Comprehension Score: {result_table['CollabGames accuracy'][0]} +/- {result_table['CollabGames standard error'][0]}\n")
        f.write(f"CollabGames Theory of Mind Score: {result_table['CollabGames accuracy'][1]} +/- {result_table['CollabGames standard error'][1]}\n")
        f.write(f"CollabGames Joint Planning Score: {result_table['CollabGames accuracy'][2]} +/- {result_table['CollabGames standard error'][2]}\n")
        f.write(f"Problems: {evaluator.issues}\n")
        
        





    
