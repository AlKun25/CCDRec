import random
import pandas as pd
import json
import numpy as np
from pathlib import Path

load_dotenv()

base_dir = os.getenv('BASE_DIR')

def load_data(dataset, data_info):
    if data_info == 'amazon':
        try:
            # Load preprocessed data directly as pandas DataFrames (much faster than line-by-line)
            data = pd.read_json(f'./dataset/subset_reviews_{dataset}.jsonl', lines=True)
            meta = pd.read_json(f'./dataset/subset_meta_{dataset}.jsonl', lines=True)
            
            # Rename columns in one operation if needed
            column_mapping = {
                'user_id': 'reviewerID', 
                'rating': 'overall',
                'parent_asin': 'asin'
            }
            data = data.rename(columns=column_mapping)
            meta = meta.rename(columns={'parent_asin': 'asin'})
            
            # Filter required columns and drop nulls in one operation
            data = data[['reviewerID', 'asin', 'overall']].dropna()
            meta = meta[['asin', 'title']].dropna()
            
            # Use efficient merge with inner join
            merged_df = pd.merge(data, meta, on='asin', how='inner')
            
            print(f"[{dataset}] Loaded {len(merged_df)} records efficiently")
            return merged_df
            
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return None
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
            return None

def overlapping_users_df(source_data, target_data, k_shot):
    source_users = set(source_data['reviewerID'].unique())
    target_users = set(target_data['reviewerID'].unique())
    overlapping_users = source_users.intersection(target_users)
    
    # Calculate sample size and sample users
    num_users_to_keep = int(len(overlapping_users) * k_shot / 100)
    sampled_users = random.sample(list(overlapping_users), num_users_to_keep)
    
    overlapping_users_set = set(overlapping_users)
    sampled_users_set = set(sampled_users)
    source_df = source_data[source_data['reviewerID'].isin(overlapping_users_set)].copy()
    target_df = target_data[target_data['reviewerID'].isin(overlapping_users_set)].copy()
    
    
    return source_df, target_df, sampled_users, overlapping_users

def train_test_split(k_shot, source, target, data_info, prompt_context):
    df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}.csv")
    
    # Sort the data by the 'overall' rating (or another criteria if needed)
    df = df.sort_values(by=['reviewerID', 'overall'], ascending=[True, False])
    
    # Use groupby for efficient per-user processing
    grouped = df.groupby('reviewerID')
    
    # Pre-allocate lists for better performance
    train_indices = []
    val_indices = []  
    test_indices = []
    
    for user, group in grouped:
        group_indices = group.index.tolist()
        
        # Split: all but last 2 → train, second-to-last → val, last → test
        if len(group_indices) > 2:
            train_indices.extend(group_indices[:-2])
            val_indices.append(group_indices[-2])
            test_indices.append(group_indices[-1])
        elif len(group_indices) == 2:
            train_indices.append(group_indices[0])
            test_indices.append(group_indices[1])
        else:
            test_indices.extend(group_indices)
    
    # Use vectorized indexing instead of concat loops
    train_data = df.loc[train_indices].reset_index(drop=True)
    val_data = df.loc[val_indices].reset_index(drop=True) if val_indices else pd.DataFrame()
    test_data = df.loc[test_indices].reset_index(drop=True)
    
    # Save efficiently
    output_dir = f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_data.to_csv(f"{output_dir}/{target}_train.csv", index=False)
    val_data.to_csv(f"{output_dir}/{target}_validation.csv", index=False)  
    test_data.to_csv(f"{output_dir}/{target}_test.csv", index=False)
    

def partition_data(source, target, k_shot, data_info, prompt_context):
    source_data = load_data(source, data_info)
    target_data = load_data(target, data_info)
    
    if source_data is None or target_data is None:
        print("Error: Could not load source or target data")
        return []
    
    # Get overlapping users and sample
    source_df, target_df, sampled_users, overlapping_users = overlapping_users_df(
        source_data, target_data, int(k_shot)
    )
    
    print("---------- Data Statistics ----------")
    print(f"Source ({source}): {len(source_df):,} interactions")
    print(f"Target ({target}): {len(target_df):,} interactions") 
    print(f"Sampled users: {len(sampled_users):,} ({k_shot}%)")
    print(f"Total overlapping users: {len(overlapping_users):,}")
    
    # Create output directory
    output_dir = f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save efficiently with compression
    source_df.to_csv(f"{output_dir}/{source}.csv", index=False)
    target_df.to_csv(f"{output_dir}/{target}.csv", index=False)
    
    train_test_split_optimized(k_shot, source, target, data_info, prompt_context)
    
    return sampled_users


# Additional optimization: Pre-filter by user sample early
def partition_data_with_early_sampling(source, target, k_shot, data_info, prompt_context):
    """Ultra-optimized version that samples users before processing"""
    print(f"Starting optimized data partitioning: {source} → {target} ({k_shot}%)")
    
    # Load data efficiently
    source_data = load_data(source, data_info)
    target_data = load_data(target, data_info)
    
    if source_data is None or target_data is None:
        return []
    
    # Find overlapping users and sample EARLY
    source_users = set(source_data['reviewerID'].unique())
    target_users = set(target_data['reviewerID'].unique()) 
    overlapping_users = source_users.intersection(target_users)
    
    # Sample users immediately to reduce dataset size
    num_users_to_keep = int(len(overlapping_users) * int(k_shot) / 100)
    sampled_users = set(random.sample(list(overlapping_users), num_users_to_keep))
    
    # Filter to sampled users ONLY (massive speedup)
    source_df = source_data[source_data['reviewerID'].isin(sampled_users)].copy()
    target_df = target_data[target_data['reviewerID'].isin(sampled_users)].copy()
    
    print("---------- Optimized Data Statistics ----------")
    print(f"Source ({source}): {len(source_df):,} interactions")
    print(f"Target ({target}): {len(target_df):,} interactions")
    print(f"Sampled users: {len(sampled_users):,} ({k_shot}%)")
    print(f"Total overlapping users: {len(overlapping_users):,}")
    
    # Create output directory
    output_dir = f"./few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the filtered datasets
    source_df.to_csv(f"{output_dir}/{source}.csv", index=False)
    target_df.to_csv(f"{output_dir}/{target}.csv", index=False)
    
    # Perform train/test split on the smaller dataset
    train_test_split_optimized(k_shot, source, target, data_info, prompt_context)
    
    return list(sampled_users)
