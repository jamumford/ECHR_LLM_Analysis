"""
Version history
v3_1 = extended to allow mcc scores from two dfs to be plotted together.
v3_0 = added ability to process keyword classification task results.
v2_3 = added ability to plot bar chart of mcc scores by Article.
v2_2 = added ability to compile results across different abstraction levels (e.g., across Articles).
v2_1 = adjusted combine_results function to use regular expression, rendering robustness against
  incomplete or malformed json outputs from the OpenAI API.
v2_0 = adapted to new formatting for different articles and court levels
v1_2 = incorporated more experiments including Chamber v Committee v GC and using manual summarisations
v1_1 = implements more flexibility for scoring various experiments such as Court and Judgment vs Decision
v1_0 = implements result processing and scoring
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
import sys


def combine_results(filepath, output_key='Case Outcome'):
    results = {}

    # Regular expression pattern to match the model's output key and its value
    case_outcome_pattern = re.compile(rf'"{output_key}":\s*"(.*?)"')

    for file in os.listdir(f'{filepath}'):
        if file.endswith('.jsonl'):
            individual_result = {}
            data = pd.read_json(f'{filepath}/{file}', lines=True)
            data = data[['custom_id', 'response']]

            for i in range(len(data)):
                result = data['response'][i]['body']['choices'][0]['message']['content']

                # Try to extract the "Case Outcome" using a regular expression
                match = case_outcome_pattern.search(result)
                if match:
                    # If the pattern is found, save the value in the results
                    individual_result[f'{data["custom_id"][i]}'] = match.group(1)
                else:
                    print(f"Could not find '{output_key}' in the result for file {file}, index {i}", flush=True)
                    print(f"Problematic result: {result[:500]}", flush=True)  # Print part of the result for debugging

            results[file] = individual_result

    return results


"""
def confusion_matrix(y_true,y_pred):
    #cm = multilabel_confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay.from_predictions(y_true,y_pred,cmap='Blues',normalize='true')
    print(disp)
    plt.show()
"""

def data_process(article, data, y_true_key, task_type):

    if task_type == 'outcome':
        data = data[['Filename','outcome']]
        # Map the outcome column
        data['outcome'] = data['outcome'].map({'violation': 2, 'nonviolation': 1})
        data = data.rename(columns={'outcome':y_true_key})
        
    elif task_type == 'keywords':    
        initial_df = data[['Filename']]
        
        data_dir = os.path.join('..', 'ECHR', 'Research_Study_2024_Difficulty_Classification')
        data_dir = os.path.abspath(data_dir)
        
        json_file_path = os.path.join(data_dir, 'key_labels.json')
        with open(json_file_path, 'r', encoding='utf-8') as f:
            key_labels = json.load(f)
            
        label_filepath = os.path.join(data_dir, "case_labels", f"{article}_labels.csv")
        labels = pd.read_csv(label_filepath)
        labels['Filename'] = labels['doc_date'].astype(str) + '_' + labels['itemid'].astype(str)
        
        data = pd.merge(initial_df, labels, on='Filename', how='left')

        # Function to map key_word_keys to real keywords, excluding the first key-value pair which corresponds to the Article itself
        def map_keys_to_values(key_word_keys, article):
            if pd.isna(key_word_keys):  # Handle NaN values
                return ''
            
            # Get the correct keywords for the article using get_keywords
            article_keywords = get_keywords(article)  # Ensure this function returns the correct values
            # Split the key_word_keys by semicolon and strip whitespaces
            keys = [key.strip() for key in key_word_keys.split(';')]
            
            if len(keys) == 0:
                raise ValueError(f'Expected at least one keyword for {article}')
        
            # Omit the first key and map the remaining keys to their values
            values = []
            for key in keys:
                if key in key_labels:  # Ensure the key exists in key_labels
                    value = key_labels[key]
                    if value in article_keywords:  # Check if the value matches the keywords for this article
                        values.append(value)
            
            return '; '.join(values)  # Join the values by semicolon
        
        # Apply the function to create the 'real_keywords' column
        data[y_true_key] = data['key_words_keys'].apply(lambda x: map_keys_to_values(x, article))
        data = data[['Filename', y_true_key]]    
    else:
        raise ValueError(f'Unexpected task_type: {task_type}')
        
    return data


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


def make_bar_chart(mcc_df, plot_dir, AI_model, text_type):
    
    # Generate the plot
    plt.figure(figsize=(10, 6))
    
    plt.rcParams.update({'font.size': 16})  # Update global font size
    
    plt.barh(mcc_df['Article'], mcc_df['MCC'], color='skyblue')
    plt.xlabel('MCC Score')
    plt.xlim(0, 1.0)
    plt.gca().invert_yaxis()  # Invert y-axis to show highest MCC at the top
    
    # Save the plot to the specified directory with a filename
    save_path = os.path.join(plot_dir, f'{AI_model}{text_type}_mcc_scores.png')
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}\n")
    #plt.show()
    
    return


def make_double_chart(fact_df, law_df, plot_dir, AI_model):
    # Define bar width and create figure
    bar_width = 0.4
    index = np.arange(len(fact_df))  # Create an index based on the number of articles

    plt.figure(figsize=(10, 6))

    # Increase font sizes for labels, ticks, and legend
    plt.rcParams.update({'font.size': 16})  # Update global font size

    # Plot the 'FACT' results
    plt.barh(index, fact_df['MCC'], bar_width, color='skyblue', label='FACT', edgecolor='black')

    # Plot the 'LAW' results with a small offset to avoid overlap
    plt.barh(index + bar_width, law_df['MCC'], bar_width, color='lightcoral', label='LAW', edgecolor='black')

    # Add labels and formatting with increased font size
    plt.xlabel('MCC Score', fontsize=18)
    plt.xlim(0, 1.0)
    plt.yticks(index + bar_width / 2, fact_df['Article'], fontsize=16)  # Align y-ticks with articles
    plt.gca().invert_yaxis()  # Invert y-axis to show highest MCC at the top

    # Add a legend with increased font size
    plt.legend(fontsize=16)

    # Save the plot to the specified directory with a filename
    save_path = os.path.join(plot_dir, f'{AI_model}_fact_vs_law_mcc_scores.png')
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}\n")

    return



def make_confusion_matrix(y_true, y_pred, analysis_dir, article, AI_model, text_type):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(analysis_dir, article)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=['Nonviolation', 'Violation'], 
                yticklabels=['Nonviolation', 'Violation'],
                annot_kws={"size": 18})  # Increase annotation font size

    # Set labels with larger font sizes
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.xticks(fontsize=14)  # Adjust x-tick label font size
    plt.yticks(fontsize=14)  # Adjust y-tick label font size

    # Save the confusion matrix plot
    save_path = os.path.join(output_dir, f'{AI_model}_confusion_matrix_{article}{text_type}.png')
    plt.savefig(save_path)
    # plt.show()
    print(f"Confusion matrix saved to {save_path}")
    
    return



def multilabel_results(y_pred, y_true):
    # Count empty instances in y_pred and y_true
    empty_y_pred_count = sum(1 for pred in y_pred if not pred or pred == {''})
    empty_y_true_count = sum(1 for true in y_true if not true or true == {''})

    # Combine all possible labels from y_pred and y_true
    all_labels = set.union(*y_pred, *y_true)
    
    # Binarize the labels for multilabel classification
    mlb = MultiLabelBinarizer(classes=sorted(all_labels))
    
    y_pred_binary = mlb.fit_transform(y_pred)
    y_true_binary = mlb.transform(y_true)
    
    # Compute Macro-Averaged Precision, Recall, F1 Score
    precision = precision_score(y_true_binary, y_pred_binary, average='macro')
    recall = recall_score(y_true_binary, y_pred_binary, average='macro')
    f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
    
    # Compute MCC for multilabel classification
    # MCC needs to be calculated per label and then averaged
    mcc_per_label = []
    for i in range(y_true_binary.shape[1]):
        mcc_per_label.append(matthews_corrcoef(y_true_binary[:, i], y_pred_binary[:, i]))
    mcc = sum(mcc_per_label) / len(mcc_per_label)
    
    # Output the results
    print(f'Count of empty instances in y_pred: {empty_y_pred_count}')
    print(f'Count of empty instances in y_true: {empty_y_true_count}')
    print(f'Count of keywords: {len(all_labels)}')
    print(f'Macro-Averaged Precision: {precision:.3f}')
    print(f'Macro-Averaged Recall: {recall:.3f}')
    print(f'Macro-Averaged F1 Score: {f1:.3f}')
    print(f'Macro-Averaged MCC: {mcc:.3f}')
    
    return mcc


def process_results(results, data, experiment, y_true_key, y_pred_key):
    '''
    function processes experiment 1 and 2 results which use different Case Outcome levels to be fairly compared to the ones recorded in the dataset
    it also provides some processing
    '''
    df = pd.DataFrame(results.items(), columns=['Filename', y_pred_key])
    df = pd.merge(data,df,on='Filename')
    y_true = df[y_true_key] #real_court
    y_pred = df[y_pred_key] #Court
    #print(y_pred)
    if experiment == 1:
        #print('\n', "HOWDY!!!", '\n')
        y_pred = y_pred.map({'key_case':1,'1':2,'2':3,'3':4,'I do not have enough information':0})
    elif experiment == 'outcome':
        y_pred = y_pred.map({'violation':2, 'nonviolation':1, 'pass':0})
    elif experiment == 'keywords':
        # Split the labels by semicolon and trim whitespaces
        y_pred = y_pred.apply(lambda x: set([label.strip() for label in x.split(';')]))
        y_true = y_true.apply(lambda x: set([label.strip() for label in x.split(';')]))          
    else:
        #y_pred = y_pred.map({1:1,2:2,3:3,4:4,'I do not have enough information':0})
        raise ValueError(f"Unexpected experiment type: {experiment}")
    #print(y_pred)
    try:
        y_pred = y_pred.astype('int64')
    except:
        print('can\'t convert to int64')
    
    return y_pred, y_true


def score_results(y_true,y_pred):
    scores = precision_recall_fscore_support(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    count_vio = sum(1 for val in y_true if val == 2)
    count_nonvio = sum(1 for val in y_true if val == 1)
    print(f'No cases: {len(y_true)}\nCount vio: {count_vio}\nCount non: {count_nonvio}\nCM: {cm}\nPrecision: {scores[0]}\nRecall: {scores[1]}\nF1: {scores[2]}\nMCC: {mcc}')
    return cm, scores, mcc


def store_predictions(master_df, article, court_level, y_pred, y_true):
    # Append the new data for the specific article
    new_data = pd.DataFrame({'Article': article, 'Court Level': court_level, 'y_pred': y_pred, 'y_true': y_true})
    master_df = pd.concat([master_df, new_data], ignore_index=True)
    return master_df


def main(AI_model, article_range, court_range, experiment_type, task_type, text_type):

    # Initialize an empty DataFrame to store all the data
    master_df = pd.DataFrame(columns=['Article', 'Court Level', 'y_pred', 'y_true'])
    
    analysis_dir = os.path.join('Analysis', task_type, AI_model)
    
    if task_type == 'outcome':
        y_pred_key = 'Case Outcome'
        y_true_key = 'real_outcome'
    elif task_type == 'keywords':
        y_pred_key = 'Case Keywords'
        y_true_key = 'real_keywords'
    else:
        raise ValueError(f'Unexpected task_type: {task_type}')

    for article in article_range:
        for court_level in court_range:
        
            # Check if the directory exists and if it contains any .jsonl files
            experiment_path = os.path.join('Results', task_type, AI_model, article, court_level, experiment_type)
            if not os.path.exists(experiment_path):
                print(f'Directory does not exist: {experiment_path}')
                continue            
            # Check if there are any .jsonl files that match the filename pattern
            found_file = False
            for f in os.listdir(experiment_path):
                if f.endswith('.jsonl') and f'{article}_{court_level}_{experiment_type}{text_type}' in f:
                    found_file = True
                    break          
            if not found_file:
                print(f'No matching .jsonl files found in: {experiment_path}')
                continue
        
            # Path for saving the log/output file
            article_save_dir = os.path.join(analysis_dir, article)
            if not os.path.exists(article_save_dir):
                os.makedirs(article_save_dir)
            log_file = f'{article}_{court_level}_{experiment_type}{text_type}.txt'
            print('log_file:', log_file)
            
            # Open the file and redirect stdout to it        
            with open(os.path.join(article_save_dir, log_file), 'w') as f:
            
                data_dir = 'Datasets'
                data = pd.read_pickle(os.path.join(data_dir, f'{article}_{court_level}_valid_data.pkl'))
                data = data_process(article, data, y_true_key, task_type)   
                results = combine_results(experiment_path, y_pred_key)
                #print(data)
                
                sys.stdout = f  # Redirect stdout to the file
                
                for name, result in results.items():
                    print(name)
                    y_pred, y_true = process_results(result, data, task_type, y_true_key, y_pred_key)
                    #print(y_pred, '\n')
                    #print(y_true)
                    if task_type == 'outcome':
                        score_results(y_true,y_pred)
                    elif task_type == 'keywords':
                        multilabel_results(y_pred, y_true)
                    print('\n')
                    
                # Reset stdout to the default output (console)
                sys.stdout = sys.__stdout__
            
            master_df = store_predictions(master_df, article, court_level, y_pred, y_true)

    # Initialising mcc_scores for use with bar chart plot
    mcc_scores = []
      
    # Get analysis at the Article level
    for article in article_range:
        print(f'\nArticle: {article}\n')
        article_data = master_df.loc[master_df['Article'] == article, ['y_pred', 'y_true']]
        y_pred = article_data['y_pred'].tolist()
        y_true = article_data['y_true'].tolist()
        if not y_true:  # This checks if y_true is an empty list
            print(f'Skipping article {article} because y_true is empty.')
            continue
        if task_type == 'outcome':
            cm, scores, mcc = score_results(y_true, y_pred)
            make_confusion_matrix(y_true, y_pred, analysis_dir, article, AI_model, text_type)
        elif task_type == 'keywords':
            mcc = multilabel_results(y_pred, y_true)
        else:
            raise ValueError(f'Unexpected task_type: {task_type}')
        mcc_scores.append((article, mcc))
        print('\n')

    # Convert the list to a pandas DataFrame for easier sorting and plotting
    mcc_df = pd.DataFrame(mcc_scores, columns=['Article', 'MCC'])
    
    # Sort the DataFrame by MCC in descending order
    mcc_df = mcc_df.sort_values(by='MCC', ascending=False)
    """
    make_bar_chart(mcc_df, analysis_dir, AI_model, text_type)

    # Get analysis at the Court level
    for court_level in court_range:
        print(f'\nCourt Level: {court_level}\n')
        court_data = master_df.loc[master_df['Court Level'] == court_level, ['y_pred', 'y_true']]
        y_pred = court_data['y_pred'].tolist()
        y_true = court_data['y_true'].tolist()
        if task_type == 'outcome':
            cm, scores, mcc = score_results(y_true,y_pred)
        elif task_type == 'keywords':
            mcc = multilabel_results(y_pred, y_true)
        else:
            raise ValueError(f'Unexpected task_type: {task_type}')

    print(f'\nOverall:\n')
    y_pred = master_df['y_pred'].tolist()
    y_true = master_df['y_true'].tolist()
    if task_type == 'outcome':
        cm, scores, mcc = score_results(y_true,y_pred)
        make_confusion_matrix(y_true, y_pred, analysis_dir, 'Overall', AI_model, text_type)
    elif task_type == 'keywords':
        mcc = multilabel_results(y_pred, y_true)
    else:
        raise ValueError(f'Unexpected task_type: {task_type}')
    """
    return mcc_df
  
                  
if __name__ == '__main__':

    # Valid task types: ['outcome', 'keywords']
    task_type = 'keywords'
    # Valid AI models: ['gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125', 'o1-preview-2024-09-12']
    AI_model = 'gpt-4o-2024-05-13'
    article_range = [f'article{i}' for i in range(2, 19)]
    #valid_court_levels = ['GRANDCHAMBER', 'CHAMBER', 'COMMITTEE']
    court_range = ['GRANDCHAMBER', 'CHAMBER', 'COMMITTEE']
    #valid_experiment = ['exp_zero_shot', 'exp_one_shot', 'exp_ADM_skeleton', 'exp_ADM_full']
    valid_experiment = ['exp_zero_shot']
    # Valid text types: ['', '_1', '_2']
    text_type = ''
    plot_dir = os.path.join('Analysis', task_type, AI_model)
        
    for experiment_type in valid_experiment:

        AI_model = 'gpt-4o'
        text_type = ''
        fact_df = main(AI_model, article_range, court_range, experiment_type, task_type, text_type)
        AI_model = 'gpt-4o-2024-05-13'
        text_type = '_2'

        law_df = main(AI_model, article_range, court_range, experiment_type, task_type, text_type)
        make_double_chart(fact_df, law_df, plot_dir, AI_model)
