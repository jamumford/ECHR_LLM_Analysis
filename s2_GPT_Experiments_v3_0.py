"""
Version history:
v3_0 = Added Experiment_3 class, that is aimed at the keyword classification task.
v2_2 = Added ADM experiment type
v2_1 = Added few-shot experiment type
v2_0 = Adapted to receive different article pickle files
v1_3 = Implemented CoT experiments
v1_2 = Implemented variations on Experiment 2
v1_1 = Experiment 2 implementation
v1_0 = Initial version - Experiment 1 - bias detection setup and running for async use-cases
"""

from openai import OpenAI
from config import openai_key
import json
import os
import pandas as pd
import random
import re
from sklearn import model_selection


JSON_SCHEMAS = [
{"Case Outcome": "string (violation, nonviolation)", "Summary": "string (brief description of the case)", "Reasoning": "string (give your reason for the case outcome)"},
{"Case Keywords": "string (relevant keywords separated by a semicolon)", "Summary": "string (brief description of the case)", "Reasoning": "string (explanation for why the selected keywords apply to this case)"}
]


articles_dict = {'article2':['Article 2', 'right to life'], 'article3':['Article 3', 'prohibition of torture'], 'article4':['Article 4', 'prohibition of slavery and forced labour'], 'article5':['Article 5', 'right to liberty and security'], 'article6':['Article 6', 'right to a fair trial'], 'article7':['Article 7', 'no punishment without law'], 'article8':['Article 8', 'right to respect for private and family life'], 'article9':['Article 9', 'freedom of thought, conscience and religion'], 'article10':['Article 10', 'freedom of expression'], 'article11':['Article 11', 'freedom of assembly and association'], 'article12':['Article 12', 'right to marry'], 'article13':['Article 13', 'right to an effective remedy'], 'article14':['Article 14', 'prohibition of discrimination'], 'article15':['Article 15', 'derogation in time of emergency'], 'article16':['Article 16', 'restrictions on political activity of aliens'], 'article17':['Article 17', 'prohibition of abuse of rights'], 'article18':['Article 18', 'limitation on use of restrictions on rights']}


PARAMETERS = {
              'outcome': {
              'exp_zero_shot':{'schema':[JSON_SCHEMAS[0]], 'zero_shot':[True], 'text':[1]},
              'exp_one_shot': {'schema':[JSON_SCHEMAS[0]], 'zero_shot':[False], 'text':[1,2,3]},
              'exp_ADM_skeleton': {'schema':[JSON_SCHEMAS[0]], 'zero_shot':[True], 'text':[1]},
              'exp_ADM_full': {'schema':[JSON_SCHEMAS[0]], 'zero_shot':[True], 'text':[1]}
                  },
              'keywords': {
              'exp_zero_shot':{'schema':[JSON_SCHEMAS[1]], 'zero_shot':[True], 'text':[2]},
              'exp_one_shot': {'schema':[JSON_SCHEMAS[1]], 'zero_shot':[False], 'text':[1,2,3]},
              'exp_ADM_skeleton': {'schema':[JSON_SCHEMAS[1]], 'zero_shot':[True], 'text':[1]},
              'exp_ADM_full': {'schema':[JSON_SCHEMAS[1]], 'zero_shot':[True], 'text':[1]}
                  }
              }



class Experiment_1():
    '''
    Experiment 1 - detecting existing bias concerning cases in the dataset
    Use in run_async to prepare data for to use with the batch API
    '''
    def __init__(self,data,binary=None,grand_chamber=True,reasoning=False, article_id='article3', AI_model='gpt-4o'):
        self.data = data
        try:
            self.process_data()
        except:
            print('can\'t process data so data accepted unprocessed')
            self.data = data

        self.binary = binary
        self.grand_chamber = grand_chamber
        self.reasoning = reasoning
        self.article_id = article_id
        self.AI_model = AI_model

    def run_async(self, schema:dict = JSON_SCHEMAS[0], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first',temperature:int=0,max_tokens:int=550,top_p=1):
        '''
        formats the experiment to the batch API for synch results - still requires the file to be sent to the batch API

        Parameters:
        filepath: str
            path to save the file
        batch_name: str
            name of the batch file
        experiment: int 
            experiment number can be 1 or 2
        schema: dict
            schema to be used for the JSON file
        zero_shot: bool
            determines if the experiment is zero-shot or few-shot
        text: int
            determines the text to be used in the prompt 1= Subject Matter, 2= Questions, 3= Both
        examples: list
            examples to be used in the prompt
        info: bool
            determines if the prompt includes the option to say don't know
        prompt_type: str
            determines the type of prompt to be used
        temperature: int
            determines the temperature to be used in the model
        max_tokens: int
            determines the max tokens to be used in the model
        '''
        
        output = []

        for file in range(len(self.data)):
            #print(self.data.iloc[file])
            if self.binary == 'binary_difficulty':
                prompt = self.get_binary_prompt(self.data.iloc[file], schema, prompt_type=prompt_type, zero_shot=zero_shot, text=text, examples=examples, info=info)
            elif self.binary == 'binary_court':
                prompt = self.get_court_prompt(self.data.iloc[file], schema, prompt_type=prompt_type, zero_shot=zero_shot, text=text, examples=examples, info=info)
            elif self.binary == 'chamber_court':
                prompt = self.get_chamber_prompt(self.data.iloc[file], schema, prompt_type=prompt_type, zero_shot=zero_shot, text=text, examples=examples, info=info, grand_chamber=self.grand_chamber)
            else:
                prompt = self.get_prompt(self.data.iloc[file], schema, prompt_type=prompt_type, zero_shot=zero_shot, text=text, examples=examples, info=info)
            id = self.data.iloc[file]['Filename']
            template = {"custom_id": f'{id}', "method": "POST", "url": "/v1/chat/completions", "body": {"model": self.AI_model, "messages": [{"role": "user", "content": prompt}],'response_format':{'type': 'json_object'},'max_tokens':max_tokens,'temperature':temperature,'top_p':top_p, 'seed':42}}

            output.append(template)
        
        return output

    def process_data(self):
        metadata = pd.read_json('./Data/overlap_cases/pruned_COMMUNICATEDCASES_meta.json',lines=True)
        metadata.rename(columns={'itemid':'Filename'}, inplace=True)
        self.data = pd.merge(self.data, metadata, on='Filename')
        self.data = self.data[['Filename','importance_x','appno','docname']]
        self.data.rename(columns={'importance_x':'importance'}, inplace=True)

    def get_prompt(self, row,schema:dict = JSON_SCHEMAS[0], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first'):
        #print(row)
        name = row['docname']
        appnos = row['appno']

        match prompt_type:
            #option to say don't know, no info on HUDOC given
            case 'first':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights. These values are given to every case after a judgment
                    or decision has been received. 
                    Using the information given to you tell me the case importance giving a response of either: key_case, 1, 2, 3. 
                    If you do not know the importance, state that you do not have enough information.
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''
            #option to say don't know, info on HUDOC given
            case 'second':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights. These values are given to every case after a judgment
                    or decision has been received. The case importance is part of the metadata on HUDOC.
                    Using the information given to you tell me the importance of the case giving a response of either: key_case, 1, 2, 3. 
                    If you do not know the importance, state that you do not have enough information.
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''
            #no option to say don't know, info on HUDOC given
            case 'third':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights. These values are given to every case after a judgment
                    or decision has been received. The case importance is part of the metadata on HUDOC.
                    Using the information given to you tell me the importance of the case giving a response of either: key_case, 1, 2, 3. 
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''
            #no option to say don't know, no info on HUDOC given
            case 'fourth':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights. These values are given to every case after a judgment
                    or decision has been received. 
                    Using the information given to you tell me the importance of the case giving a response of either: key_case, 1, 2, 3. 
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''
            #demands LLM to use HUDOC information, no option to say don't know
            case 'fifth':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights.
                    Find the information from HUDOC and tell me the importance of the case giving a response of either: key_case, 1, 2, 3. 
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''

        return exp1_prompt
    
    def get_binary_prompt():
        raise TypeError('Binary prompt not available for Experiment 1')
    def get_court_prompt():
        raise TypeError('Court prompt not available for Experiment 1')
    def get_chamber_prompt():
        raise TypeError('Chamber prompt not available for Experiment 1')
   
    
class Experiment_2(Experiment_1):
    '''
    Experiment 2 - performing the main experiments of GPT-4o performance across few-shot and zero-shot settings for outcome classification
    '''
    def __init__(self,data,content,binary=False,grand_chamber=False,reasoning=False, article_id='article3', AI_model='gpt-4o'):
        self.data = data
        try:
            self.process_data()
        except:
            print('can\'t process data so data accepted unprocessed')
            self.data = data
        self.binary = binary
        self.grand_chamber = grand_chamber
        self.reasoning = reasoning
        self.article_id = article_id
        self.AI_model = AI_model


    def process_data(self,content='both'):

        match content:
            case 'facts':
                self.data = self.data[['Filename','Facts']]
            case 'law':
                self.data = self.data[['Filename','Law']]
            case 'both':
                self.data = self.data[['Filename','Facts','Law']]    


    def get_prompt(self, row,schema:dict = JSON_SCHEMAS[0], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first'):

        '''Function to generate a prompt for the GPT-4o model.
        
        Parameters: 
        row: pd.Series
            A row from the dataframe containing the data.
        zero_shot: bool
            A boolean to determine if the prompt is for zero-shot learning.
        text: int
            The section/s of the text to include in the prompt:
                1 = The Facts
                2 = The Law
                3 = Both
        examples: list
            A list of the examples to include in the prompt.
            
        Returns:
        prompt: str
            The prompt to be used for the GPT-4o model.
        '''

        match text:
            case 1:
                text = row['Facts']
                text_amount = 'facts of the case'
            case 2:
                text = row['Law']
                text_amount = 'legal reasoning of the Court'
            case 3:
                text = row['Facts'] + ' ' + row['Law']
                text_amount = 'facts of the case and legal reasoning of the Court'
            case _:
                raise ValueError('Invalid text value. Please enter a value between 1 and 3.')

        if self.reasoning is False:
            if zero_shot:
                additional_context = ''
            else:
                additional_context = f'''You are also given a number of examples for each outcome type. 
                                        violation: {examples[0]}; nonviolation: {examples[1]}'''
        else:
            ADMS_dir = 'ADMS'
            if self.reasoning == 'exp_ADM_skeleton':
                intro_prompt = '''Use the following structured model to evaluate the facts of the case and determine whether there has been a violation or nonviolation of Article 3, consisting of enumerated nodes for acceptance along with their children nodes. If all top-level nodes are accepted, this results in a nonviolation outcome. If any top-level node is rejected, this results in a violation. Pay special attention to how Point 1 (general obligations) interacts with Points 2, 3, and 4 (specific obligations), as Point 1 must always align with the conclusion of Points 2, 3, and 4. Hence, if Point 1 is satisfied then all of Points 2, 3, and 4 must also be satisfied resulting in nonviolation. If Point 1 is not satisfied, then at least one of Points 2, 3, or 4 must also be not satisfied resulting in violation. Ensure when giving your reason that you only make reference to nodes in the model, with direct citation of the top-level nodes and their children that are relevant to the case:'''
                ADM_file_path = os.path.join(ADMS_dir, 'article3_skeleton.txt')
            elif self.reasoning == 'exp_ADM_full':
                ADM_file_path = os.path.join(ADMS_dir, 'article3_full.txt')
                intro_prompt = '''Use the following structured model to evaluate the facts of the case and determine whether there has been a violation or nonviolation of Article 3, consisting of enumerated nodes for acceptance along with their children nodes. If all top-level nodes are accepted, this results in a nonviolation outcome. If any top-level node is rejected, this results in a violation. Pay special attention to how Point 1 (general obligations) interacts with Points 2, 3, and 4 (specific obligations), as Point 1 must always align with the conclusion of Points 2, 3, and 4. Hence, if Point 1 is satisfied then all of Points 2, 3, and 4 must also be satisfied resulting in nonviolation. If Point 1 is not satisfied, then at least one of Points 2, 3, or 4 must also be not satisfied resulting in violation. Note that many of the nodes are accompanied by references to important case law that should be used to determine the satisfaction or rejection of the node based on agreement or distinction from the existing case law where the node is relevant to the case. Ensure when giving your reason that you only make reference to nodes in the model, with direct citation of the top-level nodes and their children that are relevant to the case:'''                      
            else:
                raise ValueError(f'Unexpected self.reasoning type: {self.reasoning}')    
            
            # Load ADM content and extent prompt      
            with open(ADM_file_path, "r", encoding='utf-8') as file:
                    ADM_content = file.read()
            additional_context = intro_prompt + ADM_content  

        outcome_types = f'''violation: These are the cases in which the European Court of Human Rights has found a violation of {articles_dict[self.article_id][0]} ({articles_dict[self.article_id][1]}) of the European Convention on Human Rights in favour of the applicant; nonviolation: These are the cases in which the European Court of Human Rights has found no violation of {articles_dict[self.article_id][0]} ({articles_dict[self.article_id][1]}) of the European Convention on Human Rights, but has found at least some aspect of the applicant\'s complaint to be admissible.''''''
                            '''
        if info:
            state_info = 'If you do not know the outcome, state that you do not have enough information.'
        else:
            state_info = ''
        #print('\n', articles_dict[self.article_id][0], '\n')
        prompt = f''' 
        You are a lawyer in the European Court of Human Rights, and your goal is to predict the outcome of a case, based on information provided from a case description. Outcome in a legal setting refers to the final verdict reached by the court to find in favour of the plaintiff and recognise a violation of {articles_dict[self.article_id][0]} ({articles_dict[self.article_id][1]}) of the ECHR, or to find in favour of the defendant state and declare no violation of {articles_dict[self.article_id][0]} ({articles_dict[self.article_id][1]}) of the ECHR has occured. Violation of any other Article of the ECHR must not be evaluated. Provide your answer in no more than 500 tokens. Be concise and avoid excessive elaboration. Do not apply a bias for violation that exists in the real distribution of outcomes from the ECtHR.
        The following information is provided to you:
        You will be given a case description, including the {text_amount}.
        You are given a description of the different types of outcome: {outcome_types}.
        {additional_context}.
        Based on the information given to you, as well as any relevant case law from the European Court of Human Rights, predict the outcome of the case with respect to {articles_dict[self.article_id][0]} ({articles_dict[self.article_id][1]}) of the ECHR according to the criteria given. 
        {state_info}
        The output should be given directly in JSON format, with the following schema: {schema}.
        The case description information you should base your judgement on is as follows: {text}.
        '''
        
        """
        if self.reasoning:
            prompt += ' Ensure when giving your reason you think through it step by step similarly to the example reasoning provided'
        """
        
        return prompt


class Experiment_3(Experiment_1):
    '''
    Experiment 3 - performing the main experiments of GPT-4o performance across few-shot and zero-shot settings for keyword classification
    '''
    def __init__(self,data,content,binary=False,grand_chamber=False,reasoning=False, article_id='article3', AI_model='gpt-4o'):
        self.data = data
        try:
            self.process_data()
        except:
            print('can\'t process data so data accepted unprocessed')
            self.data = data
        self.binary = binary
        self.grand_chamber = grand_chamber
        self.reasoning = reasoning
        self.article_id = article_id
        self.AI_model = AI_model


    def process_data(self,content='both'):

        match content:
            case 'facts':
                self.data = self.data[['Filename','Facts']]
            case 'law':
                self.data = self.data[['Filename','Law']]
            case 'both':
                self.data = self.data[['Filename','Facts','Law']]    


    def get_prompt(self, row, schema:dict = JSON_SCHEMAS[1], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first'):

        '''Function to generate a prompt for the GPT-4o model.
        
        Parameters: 
        row: pd.Series
            A row from the dataframe containing the data.
        zero_shot: bool
            A boolean to determine if the prompt is for zero-shot learning.
        text: int
            The section/s of the text to include in the prompt:
                1 = The Facts
                2 = The Law
                3 = Both
        examples: list
            A list of the examples to include in the prompt.
            
        Returns:
        prompt: str
            The prompt to be used for the GPT-4o model.
        '''

        match text:
            case 1:
                text = row['Facts']
                text_amount = 'facts of the case'
            case 2:
                text = row['Law']
                text_amount = 'legal reasoning of the Court'
            case 3:
                text = row['Facts'] + ' ' + row['Law']
                text_amount = 'facts of the case and legal reasoning of the Court'
            case _:
                raise ValueError('Invalid text value. Please enter a value between 1 and 3.')

        if self.reasoning is False:
            if zero_shot:
                additional_context = ''
            else:
                additional_context = f'''You are also given a number of examples for each outcome type. 
                                        violation: {examples[0]}; nonviolation: {examples[1]}'''
        else:
            ADMS_dir = 'ADMS'
            if self.reasoning == 'exp_ADM_skeleton':
                intro_prompt = '''Use the following structured model to evaluate the facts of the case and determine whether there has been a violation or nonviolation of Article 3, consisting of enumerated nodes for acceptance along with their children nodes. If all top-level nodes are accepted, this results in a nonviolation outcome. If any top-level node is rejected, this results in a violation. Pay special attention to how Point 1 (general obligations) interacts with Points 2, 3, and 4 (specific obligations), as Point 1 must always align with the conclusion of Points 2, 3, and 4. Hence, if Point 1 is satisfied then all of Points 2, 3, and 4 must also be satisfied resulting in nonviolation. If Point 1 is not satisfied, then at least one of Points 2, 3, or 4 must also be not satisfied resulting in violation. Ensure when giving your reason that you only make reference to nodes in the model, with direct citation of the top-level nodes and their children that are relevant to the case:'''
                ADM_file_path = os.path.join(ADMS_dir, 'article3_skeleton.txt')
            elif self.reasoning == 'exp_ADM_full':
                ADM_file_path = os.path.join(ADMS_dir, 'article3_full.txt')
                intro_prompt = '''Use the following structured model to evaluate the facts of the case and determine whether there has been a violation or nonviolation of Article 3, consisting of enumerated nodes for acceptance along with their children nodes. If all top-level nodes are accepted, this results in a nonviolation outcome. If any top-level node is rejected, this results in a violation. Pay special attention to how Point 1 (general obligations) interacts with Points 2, 3, and 4 (specific obligations), as Point 1 must always align with the conclusion of Points 2, 3, and 4. Hence, if Point 1 is satisfied then all of Points 2, 3, and 4 must also be satisfied resulting in nonviolation. If Point 1 is not satisfied, then at least one of Points 2, 3, or 4 must also be not satisfied resulting in violation. Note that many of the nodes are accompanied by references to important case law that should be used to determine the satisfaction or rejection of the node based on agreement or distinction from the existing case law where the node is relevant to the case. Ensure when giving your reason that you only make reference to nodes in the model, with direct citation of the top-level nodes and their children that are relevant to the case:'''                      
            else:
                raise ValueError(f'Unexpected self.reasoning type: {self.reasoning}')    
            
            # Load ADM content and extent prompt      
            with open(ADM_file_path, "r", encoding='utf-8') as file:
                    ADM_content = file.read()
            additional_context = intro_prompt + ADM_content  

        keywords_list = f'''violation: These are the cases in which the European Court of Human Rights has found a violation of {articles_dict[self.article_id][0]} ({articles_dict[self.article_id][1]}) of the European Convention on Human Rights in favour of the applicant; nonviolation: These are the cases in which the European Court of Human Rights has found no violation of {articles_dict[self.article_id][0]} ({articles_dict[self.article_id][1]}) of the European Convention on Human Rights, but has found at least some aspect of the applicant\'s complaint to be admissible.''''''
                            '''
        if info:
            state_info = 'If you do not know the outcome, state that you do not have enough information.'
        else:
            state_info = ''

        prompt = f''' 
        You are a lawyer in the European Court of Human Rights. Your task is to predict the relevant keywords from the HUDOC thesaurus for a legal case, based on the description provided. These keywords are used to categorise the case in relation to {articles_dict[self.article_id][0]} ({articles_dict[self.article_id][1]}) of the European Convention on Human Rights (ECHR).
        
        You must only evaluate keywords associated with {articles_dict[self.article_id][0]} and predict which ones are relevant to the case. Keywords not associated with this Article should not be considered. The keywords should be separated by semicolons.
        
        The following information is provided for your analysis:
        - A case description, including {text_amount}.
        - Keywords related to {articles_dict[self.article_id][0]} ({articles_dict[self.article_id][1]}): {examples}.
        - {additional_context}
        
        Based on the case description and relevant case law, predict the appropriate keywords associated with {articles_dict[self.article_id][0]} of the ECHR. 
        
        Your output must be in JSON format with the following schema: {schema}
        
        The case description information is as follows: {text}.
        '''
        
        """
        if self.reasoning:
            prompt += ' Ensure when giving your reason you think through it step by step similarly to the example reasoning provided'
        """
        
        return prompt


def create_examples(data, text, example_num, outcome=True):
    '''Function to create a specified number of examples for few-shot learning prompt,
    selecting n cases for each outcome type.
    '''
    examples = []
    # Group the data by 'outcome' and sample n examples from each group
    grouped = data.groupby('outcome')

    # Randomly sample n cases for each outcome
    sampled_data = grouped.apply(lambda x: x.sample(n=example_num, random_state=1))
    sampled_data.reset_index(drop=True, inplace=True)

    # Iterate through the sampled data
    for i in range(len(sampled_data)):
        match text:
            case 1:
                example = sampled_data.iloc[i]['Facts']
            case 2:
                example = sampled_data.iloc[i]['Law']
            case 3:
                example = sampled_data.iloc[i]['Facts'] + ' ' + sampled_data.iloc[i]['Law']

        # Append the example to the list
        examples.append(example)

    return examples


def get_keywords(article):

    data_dir = os.path.join('..', 'ECHR', 'Research_Study_2024_Difficulty_Classification')
    data_dir = os.path.abspath(data_dir)   
    json_file_path = os.path.join(data_dir, 'key_labels.json')
    
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        key_labels = json.load(f)
    
    # Extract the article number from the input (e.g., 'article2' -> 2)
    article_number = int(article.replace('article', ''))
    #print(key_labels)
    
    # Create a regular expression to match values like 'Art. {i}' or 'Art. {i}-...'
    pattern = re.compile(rf'^\(Art\. {article_number}(\W.*)?$')
    
    # Find and return where the value matches the pattern - omit first entry as corresponding to the Article itself
    keywords_keys = [key for key, value in key_labels.items() if pattern.match(value)][1:]
    keywords_values = [value for key, value in key_labels.items() if pattern.match(value)][1:]
    #print(f'keywords: {keywords_values}')
    
    return keywords_values


def save_file(output, filepath, batch_name):

    with open(f'{filepath}/{batch_name}.jsonl', 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')


def main(AI_model, article, court_level, experiment_type, task_type):

    print('\n', f'Generating batch for {article} and {court_level} and {experiment_type}', '\n')
    
    #read in data
    data_dir = 'Datasets'
    valid_data_path = os.path.join(data_dir, f'{article}_{court_level}_valid_data.pkl')
    
    # Check if valid_data.pkl exists
    if not os.path.exists(valid_data_path):
        print(f"Valid data file not found for {article} and {court_level} at {valid_data_path}. Exiting function.")
        return  # Exit the function early if the file does not exist
    
    # Read in the valid data
    df = pd.read_pickle(valid_data_path)
    
    # Read in example test data for one-shot experiments if needed
    if experiment_type in ['exp_one_shot']:
        test_data_path = os.path.join(data_dir, f'{article}_{court_level}_test_data.pkl')
        if not os.path.exists(test_data_path):
            print(f"Test data file not found for {article} and {court_level} at {test_data_path}. Exiting function.")
            return  # Exit early if the test data file does not exist
        df_example = pd.read_pickle(test_data_path)

    #set params for experiment being run
    if experiment_type in ['exp_zero_shot', 'exp_one_shot']:
        exp_reasoning = False
    elif experiment_type in ['exp_ADM_skeleton', 'exp_ADM_full']:
        exp_reasoning = experiment_type
    else:
        raise ValueError(f"Unexpected experiment_type: {experiment_type}")
    
    if task_type == 'outcome':
        exp_task = Experiment_2(df, content='both', grand_chamber=True, reasoning=exp_reasoning, article_id=article, AI_model=AI_model)
        article_keywords = []
    elif task_type == 'keywords':
        article_keywords = get_keywords(article)
        exp_task = Experiment_3(df, content='both', grand_chamber=True, reasoning=exp_reasoning, article_id=article, AI_model=AI_model)
    else:
        raise ValueError(f'Unexpected task_type: {task_type}')

    grid = model_selection.ParameterGrid(PARAMETERS[task_type][experiment_type])
    
    # Save prompts to jsonl file
    save_dir = os.path.join('Batches', task_type, AI_model, article, court_level, experiment_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for params in grid:
        #set examples
        if experiment_type in ['exp_one_shot']:
            examples = create_examples(df_example, text=params['text'], example_num = 1)
            # Only input the facts for the submitted case
            case_text_type = 1
        elif experiment_type in ['exp_zero_shot', 'exp_ADM_skeleton', 'exp_ADM_full']:
            examples = article_keywords
            # Input all variants of case descriptions for the submitted case
            case_text_type = params['text']
        else:
            raise ValueError(f"Unexpected experiment_type: {experiment_type}")
        #create batch files for given prompt + parameters
        output = exp_task.run_async(schema=params['schema'], zero_shot=params['zero_shot'], text=case_text_type, examples=examples, info=True, temperature=0, max_tokens=550)
        #save batch file
        save_file(output, filepath=save_dir, batch_name=f'{article}_{court_level}_{experiment_type}_{case_text_type}')


if __name__ == "__main__":
    
    #connect to api key
    client = OpenAI(api_key=openai_key)
    
    # Valid task types: ['outcome', 'keywords']
    task_type = 'keywords'
    # Valid AI models: ['gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125', 'o1-preview-2024-09-12']
    AI_model = 'gpt-3.5-turbo-0125'
    # Valid Court Levels: ['GRANDCHAMBER', 'CHAMBER', 'COMMITTEE']
    valid_court = ['GRANDCHAMBER', 'CHAMBER', 'COMMITTEE']
    # Valid experiment types: ['exp_zero_shot', 'exp_one_shot', 'exp_ADM_skeleton', 'exp_ADM_full']
    valid_experiment = ['exp_zero_shot']
    
    for i in range(2, 19):
        for court_level in valid_court:
            for experiment_type in valid_experiment:
                main(AI_model, f'article{i}', court_level, experiment_type, task_type)
    
