"""
Version history
v2_1 = added time delay functionality to avoid rate limit errors from the API.
v2_0 = adapts to new formatting by article and court level.
v1_0 = implements ability to send files to api and download the results.
"""

import openai
from openai import OpenAI
from config import openai_key
import os
import time

client = OpenAI(api_key=openai_key)
  
            
def download_batch_files(experiment_name, filepath):
    '''
    Downloads files from OpenAI API after batch processing, 
    appending the batch ID to the file name for uniqueness.
    '''
    print(f"Downloading batch files for experiment: {experiment_name}")
    batches = client.batches.list()
    #print(f"Batches retrieved: {batches}")
    
    for batch in batches:
        metadata = batch.metadata.get('description', '').split(':')
        #print(metadata[0].strip())
        # Ensure that the metadata matches the experiment name
        if len(metadata) > 0 and metadata[0].strip() == experiment_name:
            file_id = batch.output_file_id
            if not file_id:
                print(f"No output file ID found for batch: {batch.id} with description {batch.metadata['description']}")
                continue

            try:
                # Download the batch output file content
                batch_output_file = client.files.content(file_id).content
                
                # Use metadata[1] as the base filename and append the batch ID for uniqueness
                base_filename = metadata[1].strip()
                unique_filename = f'{base_filename}_{batch.id}.jsonl'  # Adjust extension if needed
                
                # Write the file with the unique filename
                with open(os.path.join(filepath, unique_filename), "wb") as f:
                    f.write(batch_output_file)
                print(f"Downloaded batch {batch.id} to {unique_filename}")
            except Exception as e:
                print(f"Error downloading file for batch {batch.id}: {e}")
                

def send_to_api(filepath,experiment_desciption):

    '''
    Sends files to OpenAI API for batch processing

    Experiment_description naming conventions =
        'Experiment {number} {type}:'
    '''
    print(f'filepath: {filepath}')
    for i in os.listdir(f'{filepath}'):
        if i.endswith('2.jsonl'):
            print(f'file: {i}')
            batch_input_file = client.files.create(
                file=open(filepath+f"/{i}", "rb"),
                purpose="batch"
                )

            batch_input_file_id = batch_input_file.id

            client.batches.create(
            input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                "description": f"{experiment_desciption} {i}"
                })
 
 
def main(AI_model, operation_type, article, court_level, experiment_type, task_type):

    experiment_id = f'{task_type}_{AI_model}_{article}_{court_level}_{experiment_type}_2'
    
    if operation_type == "send_to_api":
        prompt_path = os.path.join('Batches', task_type, AI_model, article, court_level, experiment_type)
        if not os.path.exists(prompt_path):
            print(f"Valid batch file not found for {article} and {court_level} at {prompt_path}. Exiting function.")
            return  # Exit the function early if the file does not exist
        send_to_api(filepath=prompt_path, experiment_desciption=f'{experiment_id}: ')
    elif operation_type == "download_batch":
        save_dir = os.path.join('Results', task_type, AI_model, article, court_level, experiment_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        download_batch_files(experiment_name=experiment_id, filepath=save_dir)
    else:
        raise ValueError(f"Unexpected operation_type: {operation_type}")
    
    return
    
    
if __name__ == '__main__':
  
    #operation_type = "send_to_api"
    operation_type = "download_batch"

    # Valid task types: ['outcome', 'keywords']
    task_type = 'keywords'
    # Valid AI models: ['gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125', 'o1-preview-2024-09-12']
    AI_model = 'gpt-3.5-turbo-0125'
    #valid_court_levels = ['GRANDCHAMBER', 'CHAMBER', 'COMMITTEE']
    valid_court = ['GRANDCHAMBER', 'CHAMBER', 'COMMITTEE']
    #valid_experiment = ['exp_zero_shot', 'exp_one_shot', 'exp_ADM_skeleton', 'exp_ADM_full']
    valid_experiment = ['exp_zero_shot']
    
    for i in range(2, 19):
        for court_level in valid_court:
            for experiment_type in valid_experiment:
                try:
                    main(AI_model, operation_type, f'article{i}', court_level, experiment_type, task_type)
                except openai.RateLimitError as e:
                    print(f"Rate limit exceeded: {e}")
                    time.sleep(20)  # Wait for 20 seconds before retrying    
                # Sleep for a small interval between each API call to avoid hitting the rate limit
                time.sleep(6)

    