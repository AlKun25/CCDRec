import os
import random
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

base_dir = os.getenv('BASE_DIR')

def load_data(dataset, data_info):
    if data_info == 'amazon':
        # Read JSONL files line by line to handle potential parsing issues
        data_list = []
        meta_list = []
        
        # 
        
        # Read data file - Updated paths to use correct folder and subset files
        try:
            with open(f'./dataset/subset_reviews_{dataset}.jsonl', 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        if line.strip():  # Skip empty lines
                            data_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed line {line_num + 1} in subset_reviews_{dataset}.jsonl: {e}")
                        continue
            
            # Read meta file - Updated paths to use correct folder and subset files
            with open(f'./dataset/subset_meta_{dataset}.jsonl', 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        if line.strip():  # Skip empty lines
                            meta_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed line {line_num + 1} in subset_meta_{dataset}.jsonl: {e}")
                        continue
        
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None
        
        # Convert to DataFrames
        data = pd.DataFrame(data_list)
        meta = pd.DataFrame(meta_list)
        
        # Map the column names to match the expected format
        # Note: Your subset files already have the correct column names (reviewerID, parent_asin, overall)
        # but we still need to handle the asin mapping for meta
        if 'user_id' in data.columns:
            data = data.rename(columns={'user_id': 'reviewerID', 'rating': 'overall'})
        if 'parent_asin' in data.columns:
            data = data.rename(columns={'parent_asin': 'asin'})
        
        # For meta, we need to map parent_asin to asin for the merge
        if 'parent_asin' in meta.columns:
            meta = meta.rename(columns={'parent_asin': 'asin'})
        
        data_columns = ['reviewerID', 'asin', 'overall']
        meta_columns = ['asin', 'title']

        # Check if required columns exist
        missing_data_cols = [col for col in data_columns if col not in data.columns]
        missing_meta_cols = [col for col in meta_columns if col not in meta.columns]
        
        if missing_data_cols:
            print(f"Missing columns in data: {missing_data_cols}")
            print(f"Available columns in data: {list(data.columns)}")
        
        if missing_meta_cols:
            print(f"Missing columns in meta: {missing_meta_cols}")
            print(f"Available columns in meta: {list(meta.columns)}")
        
        # Filter columns (only keep rows that have all required columns)
        data = data[data_columns].dropna()
        meta = meta[meta_columns].dropna()

        merged_df = pd.merge(data, meta, on='asin')

        return merged_df

def overlapping_users_df(source_data, target_data, k_shot):
    overlapping_users = set(source_data['reviewerID']).intersection(target_data['reviewerID'])
    num_users_to_keep = int(len(overlapping_users) * k_shot / 100)
    print("Overlapping Users: ", len(overlapping_users))
    # print("Overlapping user example:", overlapping_users[0])
    sampled_users = random.sample(list(overlapping_users), num_users_to_keep)

    source_df = source_data[source_data['reviewerID'].isin(overlapping_users)]
    target_df = target_data[target_data['reviewerID'].isin(overlapping_users)]

    return source_df, target_df, sampled_users, overlapping_users

def train_test_split(k_shot, source, target, data_info, prompt_context):
    df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}.csv")

    # Sort the data by the 'overall' rating (or another criteria if needed)
    df = df.sort_values(by=['reviewerID', 'overall'], ascending=[True, False])

    # Splitting the data
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    validation_data = pd.DataFrame()

    # For each user, put the last interaction in the test set, the second-to-last in the validation set, and all others in the training set
    for user, group in df.groupby('reviewerID'):
        if len(group) > 1:
            train_data = pd.concat([train_data, group.iloc[:-2]]) # All but last two
            validation_data = pd.concat([validation_data, group.iloc[-2:-1].reset_index(drop=True)]) # Second-to-last
            test_data = pd.concat([test_data, group.iloc[-1:].reset_index(drop=True)]) # Last one
        else:
            # For users with only one interaction, put it in the test set
            test_data = pd.concat([test_data, group])

    # Save the train, validation, and test data
    train_data.to_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv", index=False)
    validation_data.to_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation.csv", index=False)
    test_data.to_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test.csv", index=False)

def partition_data(source, target, k_shot, data_info, prompt_context):
    source_data = load_data(source, data_info)
    target_data = load_data(target, data_info)

    source_df, target_df, sampled_users, overlapping_users = overlapping_users_df(source_data, target_data, int(k_shot))
    
    print("---------- Data Statistics ----------\n")
    print(f"Source Dataframe:\n\n{source_df}\n")
    print(f"Target Dataframe:\n\n{target_df}\n")
    print(f"{k_shot}% of Sampled Users: {len(sampled_users)}")
    print(f"Total number of overlapping Users: {len(overlapping_users)}")
    
    # Build output path
    output_dir = f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}"
    os.makedirs(output_dir, exist_ok=True)
    
    source_df.to_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv", index=False)
    target_df.to_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}.csv", index=False)   

    train_test_split(k_shot, source, target, data_info, prompt_context)

    return sampled_users






