import os
import pandas as pd
import random
import numpy as np
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

base_dir = os.getenv('BASE_DIR')

NL = "\n\n"
Q = "\'"

def ranking_with_target_injection_train_dataset(k_shot, source, target, data_info, neg_samples, sampled_users, prompt_context):
    
    train_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    
    sampled_users_set = set(sampled_users)
    train_df = train_df[train_df['reviewerID'].isin(sampled_users_set)]
    source_data = source_data[source_data['reviewerID'].isin(sampled_users_set)]
    
    # Pre-group data by user 
    print("Pre-processing user data...")
    source_groups = source_data.groupby('reviewerID')
    train_groups = train_df.groupby('reviewerID')
    
    #Pre-compute user data lookups
    user_source_cache = {}
    user_target_cache = {}
    
    for user in sampled_users_set:
        if user in source_groups.groups:
            user_source_cache[user] = source_groups.get_group(user).head(10)
        else:
            user_source_cache[user] = pd.DataFrame()
            
        if user in train_groups.groups:
            user_target_cache[user] = train_groups.get_group(user)
        else:
            user_target_cache[user] = pd.DataFrame()
    
    #Pre-compute negative sampling pools per user
    print("Pre-computing negative item pools...")
    all_items = set(train_df['title'].unique())
    user_negative_pools = {}
    
    for user in sampled_users_set:
        user_items = set(user_target_cache[user]['title']) if not user_target_cache[user].empty else set()
        user_negative_pools[user] = list(all_items - user_items)
    
    # Use list to collect results (not repeated concat)
    results = []
    total_rows = len(train_df)
    processed = 0
    
    print(f"Processing {total_rows} interactions...")
    
    # Vectorized iteration using itertuples (faster than iterrows)
    for row in train_df.itertuples():
        user = row.reviewerID
        interacted_item = row.title
        
        # Use pre-cached data (O(1) lookup vs O(n) scan)
        user_source_data = user_source_cache[user]
        user_target_data = user_target_cache[user]
        
        # Choose 10 items from the target domain that the user has not interacted with
        user_target_filtered = user_target_data[user_target_data['title'] != interacted_item].head(10)
        
        # Fast negative sampling from pre-computed pool
        available_negatives = [item for item in user_negative_pools[user] if item != interacted_item]
        if len(available_negatives) >= neg_samples:
            non_interacted_items = random.sample(available_negatives, neg_samples)
        else:
            non_interacted_items = available_negatives
        
        # Generate formatted strings efficiently
        source_items_formatted = [f"title: {title}, rating: {rating}" 
                                for title, rating in zip(user_source_data['title'], user_source_data['overall'])]
        
        target_items_formatted = [f"title: {title}, rating: {rating}" 
                                for title, rating in zip(user_target_filtered['title'], user_target_filtered['overall'])]
        
        # Create candidate items and shuffle
        candidate_items = [interacted_item] + non_interacted_items
        random.shuffle(candidate_items)
        
        perfect_ranking = [interacted_item] + non_interacted_items
        items_format = ', '.join([f'Item{i+1}' for i in range(len(candidate_items))])
        
        # Generate prompt based on context
        ranking_prompt = generate_ranking_prompt(
            source_items_formatted, target_items_formatted, candidate_items, 
            items_format, prompt_context, source, target
        )
        
        # Append to list instead of DataFrame concat
        results.append({
            'prompt': ranking_prompt,
            'correct_ranking': perfect_ranking
        })
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{total_rows} interactions ({processed/total_rows*100:.1f}%)")
    
    # Single DataFrame creation at the end
    print("Creating final dataset...")
    ranking_data = pd.DataFrame(results)
    
    # Save efficiently
    output_path = f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train_ranking_with_injection.csv"
    ranking_data.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(results)} ranking prompts")

def generate_ranking_prompt(source_items, target_items, candidate_items, items_format, prompt_context, source, target):
    """Helper function to generate prompts efficiently"""
    if prompt_context == 'none':
        print("Nothing")
        return ""
    
    elif prompt_context == 'medium':
        return (
            "Here is a user's rating history in the source domain:\n\n" +
            "\n".join(source_items) + "\n\n" +
            "Here is a user's rating history in the target domain:\n\n" +
            "\n".join(target_items) + "\n\n" +
            "This is a list of candidate items in the target domain: " +
            f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
            f"Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact with based on the user's past interactions in the source and target domains. The list should contain only the items from the list of candidate items, don't make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
        )
    
    elif prompt_context == 'high':
        return (
            f"You are a cross-domain recommender. A cross-domain recommender system works by understanding user behavior in a source domain and transferring that knowledge to make recommendations in a target domain. In this example, the source domain is {source} and the target domain is {target}, which means that each domain consists of items related to each other within that domain. Below is the user's rating history in the {source} and {target} domains, where you will see the ratings that the user gave to items in each domain. 1.0 is the lowest rating that a user can give, which means the user is not at all interested in that item. 5.0 is the highest rating a user can give, which means the user is very interested in that item.\n\n" +
            f"Here is a user's rating history in the {source} domain:\n\n" +
            "\n".join(source_items) + "\n\n" +
            f"Here is the same user's rating history in the {target} domain:\n\n" +
            "\n".join(target_items) + "\n\n" +
            f"This is the list of candidate items in the {target} domain: " +
            f"[{', '.join([Q + str(item) + Q for item in candidate_items])}]{NL}" +
            f"You need to infer the user's preferences in the target domain ({target}) based on their rating information in both the {source} domain and {target} domain in order to rank the candidate list of items in the {target} domain. Return a single list in this format: [{items_format}]. The list should have the candidate items ranked in the order of most likely to least likely to interact based on the user's past interactions in the {source} and {target} domains. The list should contain only the items from the list of candidate items, don't make up titles or add other items to the output list that are not present in the candidate list. Don't provide any explanation or analysis, just return a single list in the format above."
        )


def ranking_no_target_injection_train_dataset(k_shot, source, target, data_info, neg_samples, sampled_users, prompt_context):
    
    print(f"Loading data for {source} → {target} ranking (no injection)")
    train_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    
    sampled_users_set = set(sampled_users)
    train_df = train_df[train_df['reviewerID'].isin(sampled_users_set)]
    source_data = source_data[source_data['reviewerID'].isin(sampled_users_set)]
    
    # Pre-group data by user (avoids repeated filtering)
    print("Pre-processing user data...")
    source_groups = source_data.groupby('reviewerID')
    train_groups = train_df.groupby('reviewerID')
    
    # Pre-compute user data lookups
    user_source_cache = {}
    user_target_cache = {}
    
    for user in sampled_users_set:
        if user in source_groups.groups:
            user_source_cache[user] = source_groups.get_group(user).head(10)
        else:
            user_source_cache[user] = pd.DataFrame()
            
        if user in train_groups.groups:
            user_target_cache[user] = train_groups.get_group(user)
        else:
            user_target_cache[user] = pd.DataFrame()
    
    # Pre-compute negative sampling pools per user
    print("Pre-computing negative item pools...")
    all_items = set(train_df['title'].unique())
    user_negative_pools = {}
    
    for user in sampled_users_set:
        user_items = set(user_target_cache[user]['title']) if not user_target_cache[user].empty else set()
        user_negative_pools[user] = list(all_items - user_items)
    
    # Use list to collect results (not repeated concat)
    results = []
    total_rows = len(train_df)
    processed = 0
    
    print(f"Processing {total_rows} interactions...")
    
    # Vectorized iteration using itertuples
    for row in train_df.itertuples():
        user = row.reviewerID
        interacted_item = row.title
        
        user_source_data = user_source_cache[user]
        
        # Fast negative sampling from pre-computed pool
        available_negatives = [item for item in user_negative_pools[user] if item != interacted_item]
        if len(available_negatives) >= neg_samples:
            non_interacted_items = random.sample(available_negatives, neg_samples)
        else:
            non_interacted_items = available_negatives
        
        # Generate formatted strings efficiently
        source_items_formatted = [f"title: {title}, rating: {rating}" 
                                for title, rating in zip(user_source_data['title'], user_source_data['overall'])]
        
        # Create candidate items and shuffle
        candidate_items = [interacted_item] + non_interacted_items
        random.shuffle(candidate_items)
        
        perfect_ranking = [interacted_item] + non_interacted_items
        items_format = ', '.join([f'Item{i+1}' for i in range(len(candidate_items))])
        
        # Generate prompt (no target injection)
        ranking_prompt = generate_ranking_prompt(
            source_items_formatted, [], candidate_items,  # Empty target_items for no injection
            items_format, prompt_context, source, target
        )
        
        # Append to list instead of DataFrame concat
        results.append({
            'prompt': ranking_prompt,
            'correct_ranking': perfect_ranking
        })
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{total_rows} interactions ({processed/total_rows*100:.1f}%)")
    
    # Single DataFrame creation at the end
    print("Creating final dataset...")
    ranking_data = pd.DataFrame(results)
    
    # Save efficiently
    output_path = f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train_ranking_no_injection.csv"
    ranking_data.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(results)} ranking prompts (no injection)")

def ranking_with_target_injection_validation_dataset(k_shot, source, target, data_info, neg_samples, prompt_context):
    
    print(f"Loading validation data for {source} → {target} ranking (with injection)")
    train_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    validation_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation.csv")

    # Pre-group data by user (avoids repeated filtering)
    print("Pre-processing user data...")
    source_groups = source_data.groupby('reviewerID')
    train_groups = train_df.groupby('reviewerID')
    
    # Get unique users from validation set
    validation_users = set(validation_df['reviewerID'].unique())
    
    # Pre-compute user data lookups
    user_source_cache = {}
    user_target_cache = {}
    
    for user in validation_users:
        if user in source_groups.groups:
            user_source_cache[user] = source_groups.get_group(user).head(10)
        else:
            user_source_cache[user] = pd.DataFrame()
            
        if user in train_groups.groups:
            user_target_cache[user] = train_groups.get_group(user)
        else:
            user_target_cache[user] = pd.DataFrame()
    
    # Pre-compute negative sampling pools per user
    print("Pre-computing negative item pools...")
    all_items = set(train_df['title'].unique())
    user_negative_pools = {}
    
    for user in validation_users:
        user_items = set(user_target_cache[user]['title']) if not user_target_cache[user].empty else set()
        user_negative_pools[user] = list(all_items - user_items)
    
    # Use list to collect results (not repeated concat)
    results = []
    total_rows = len(validation_df)
    processed = 0
    
    print(f"Processing {total_rows} validation interactions...")
    
    # Vectorized iteration using itertuples
    for row in validation_df.itertuples():
        user = row.reviewerID
        interacted_item = row.title
        
        # Use pre-cached data (O(1) lookup vs O(n) scan)
        user_source_data = user_source_cache[user]
        user_target_data = user_target_cache[user]
        
        # Filter out current item from target data
        user_target_filtered = user_target_data[user_target_data['title'] != interacted_item].head(10)
        
        # Fast negative sampling from pre-computed pool
        available_negatives = [item for item in user_negative_pools[user] if item != interacted_item]
        if len(available_negatives) >= neg_samples:
            non_interacted_items = random.sample(available_negatives, neg_samples)
        else:
            non_interacted_items = available_negatives
        
        # Generate formatted strings efficiently
        source_items_formatted = [f"title: {title}, rating: {rating}" 
                                for title, rating in zip(user_source_data['title'], user_source_data['overall'])]
        
        target_items_formatted = [f"title: {title}, rating: {rating}" 
                                for title, rating in zip(user_target_filtered['title'], user_target_filtered['overall'])]
        
        # Create candidate items and shuffle
        candidate_items = [interacted_item] + non_interacted_items
        random.shuffle(candidate_items)
        
        perfect_ranking = [interacted_item] + non_interacted_items
        items_format = ', '.join([f'Item{i+1}' for i in range(len(candidate_items))])
        
        # Generate prompt (with target injection)
        ranking_prompt = generate_ranking_prompt(
            source_items_formatted, target_items_formatted, candidate_items,
            items_format, prompt_context, source, target
        )
        
        # Append to list instead of DataFrame concat
        results.append({
            'prompt': ranking_prompt,
            'correct_ranking': perfect_ranking
        })
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{total_rows} interactions ({processed/total_rows*100:.1f}%)")
    
    # Single DataFrame creation at the end
    print("Creating final dataset...")
    ranking_data = pd.DataFrame(results)
    
    # Save efficiently
    output_path = f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation_ranking_with_injection.csv"
    ranking_data.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(results)} validation ranking prompts (with injection)")

def ranking_no_target_injection_validation_dataset(k_shot, source, target, data_info, neg_samples, prompt_context):
    """OPTIMIZED: No target injection validation dataset generation"""
    
    print(f"Loading validation data for {source} → {target} ranking (no injection)")
    train_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    validation_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation.csv")

    # Pre-group data by user (avoids repeated filtering)
    print("Pre-processing user data...")
    source_groups = source_data.groupby('reviewerID')
    train_groups = train_df.groupby('reviewerID')
    
    # Get unique users from validation set
    validation_users = set(validation_df['reviewerID'].unique())
    
    # Pre-compute user data lookups
    user_source_cache = {}
    user_target_cache = {}
    
    for user in validation_users:
        if user in source_groups.groups:
            user_source_cache[user] = source_groups.get_group(user).head(10)
        else:
            user_source_cache[user] = pd.DataFrame()
            
        if user in train_groups.groups:
            user_target_cache[user] = train_groups.get_group(user)
        else:
            user_target_cache[user] = pd.DataFrame()
    
    # Pre-compute negative sampling pools per user
    print("Pre-computing negative item pools...")
    all_items = set(train_df['title'].unique())
    user_negative_pools = {}
    
    for user in validation_users:
        user_items = set(user_target_cache[user]['title']) if not user_target_cache[user].empty else set()
        user_negative_pools[user] = list(all_items - user_items)
    
    # Use list to collect results (not repeated concat)
    results = []
    total_rows = len(validation_df)
    processed = 0
    
    print(f"Processing {total_rows} validation interactions...")
    
    # Vectorized iteration using itertuples
    for row in validation_df.itertuples():
        user = row.reviewerID
        interacted_item = row.title
        
        # Use pre-cached data (O(1) lookup vs O(n) scan)
        user_source_data = user_source_cache[user]
        
        # Fast negative sampling from pre-computed pool
        available_negatives = [item for item in user_negative_pools[user] if item != interacted_item]
        if len(available_negatives) >= neg_samples:
            non_interacted_items = random.sample(available_negatives, neg_samples)
        else:
            non_interacted_items = available_negatives
        
        # Generate formatted strings efficiently
        source_items_formatted = [f"title: {title}, rating: {rating}" 
                                for title, rating in zip(user_source_data['title'], user_source_data['overall'])]
        
        # Create candidate items and shuffle
        candidate_items = [interacted_item] + non_interacted_items
        random.shuffle(candidate_items)
        
        perfect_ranking = [interacted_item] + non_interacted_items
        items_format = ', '.join([f'Item{i+1}' for i in range(len(candidate_items))])
        
        # Generate prompt (no target injection)
        ranking_prompt = generate_ranking_prompt(
            source_items_formatted, [], candidate_items,  # Empty target_items for no injection
            items_format, prompt_context, source, target
        )
        
        # Append to list instead of DataFrame concat
        results.append({
            'prompt': ranking_prompt,
            'correct_ranking': perfect_ranking
        })
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{total_rows} interactions ({processed/total_rows*100:.1f}%)")
    
    # Single DataFrame creation at the end
    print("Creating final dataset...")
    ranking_data = pd.DataFrame(results)
    
    # Save efficiently
    output_path = f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_validation_ranking_no_injection.csv"
    ranking_data.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(results)} validation ranking prompts (no injection)")

def ranking_with_target_injection_test_dataset(k_shot, source, target, data_info, neg_samples, prompt_context):
    """OPTIMIZED: With target injection test dataset generation"""
    
    print(f"Loading test data for {source} → {target} ranking (with injection)")
    train_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    test_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test.csv")

    # Pre-group data by user (avoids repeated filtering)
    print("Pre-processing user data...")
    source_groups = source_data.groupby('reviewerID')
    train_groups = train_df.groupby('reviewerID')
    
    # Get unique users from test set
    test_users = set(test_df['reviewerID'].unique())
    
    # Pre-compute user data lookups
    user_source_cache = {}
    user_target_cache = {}
    
    for user in test_users:
        if user in source_groups.groups:
            user_source_cache[user] = source_groups.get_group(user).head(10)
        else:
            user_source_cache[user] = pd.DataFrame()
            
        if user in train_groups.groups:
            user_target_cache[user] = train_groups.get_group(user)
        else:
            user_target_cache[user] = pd.DataFrame()
    
    # Pre-compute negative sampling pools per user
    print("Pre-computing negative item pools...")
    all_items = set(train_df['title'].unique())
    user_negative_pools = {}
    
    for user in test_users:
        user_items = set(user_target_cache[user]['title']) if not user_target_cache[user].empty else set()
        user_negative_pools[user] = list(all_items - user_items)
    
    # Use list to collect results (not repeated concat)
    results = []
    total_rows = len(test_df)
    processed = 0
    
    print(f"Processing {total_rows} test interactions...")
    
    # Vectorized iteration using itertuples
    for row in test_df.itertuples():
        user = row.reviewerID
        interacted_item = row.title
        
        # Use pre-cached data (O(1) lookup vs O(n) scan)
        user_source_data = user_source_cache[user]
        user_target_data = user_target_cache[user]
        
        # Filter out current item from target data
        user_target_filtered = user_target_data[user_target_data['title'] != interacted_item].head(10)
        
        # Fast negative sampling from pre-computed pool
        available_negatives = [item for item in user_negative_pools[user] if item != interacted_item]
        if len(available_negatives) >= neg_samples:
            non_interacted_items = random.sample(available_negatives, neg_samples)
        else:
            non_interacted_items = available_negatives
        
        # Generate formatted strings efficiently
        source_items_formatted = [f"title: {title}, rating: {rating}" 
                                for title, rating in zip(user_source_data['title'], user_source_data['overall'])]
        
        target_items_formatted = [f"title: {title}, rating: {rating}" 
                                for title, rating in zip(user_target_filtered['title'], user_target_filtered['overall'])]
        
        # Create candidate items and shuffle
        candidate_items = [interacted_item] + non_interacted_items
        random.shuffle(candidate_items)
        
        perfect_ranking = [interacted_item] + non_interacted_items
        items_format = ', '.join([f'Item{i+1}' for i in range(len(candidate_items))])
        
        # Generate prompt (with target injection)
        ranking_prompt = generate_ranking_prompt(
            source_items_formatted, target_items_formatted, candidate_items,
            items_format, prompt_context, source, target
        )
        
        # Append to list instead of DataFrame concat
        results.append({
            'prompt': ranking_prompt,
            'correct_ranking': perfect_ranking
        })
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{total_rows} interactions ({processed/total_rows*100:.1f}%)")
    
    # Single DataFrame creation at the end
    print("Creating final dataset...")
    ranking_data = pd.DataFrame(results)
    
    # Save efficiently
    output_path = f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test_ranking_with_injection.csv"
    ranking_data.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(results)} test ranking prompts (with injection)")

def ranking_no_target_injection_test_dataset(k_shot, source, target, data_info, neg_samples, prompt_context):
    """OPTIMIZED: No target injection test dataset generation"""
    
    print(f"Loading test data for {source} → {target} ranking (no injection)")
    train_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_train.csv")
    source_data = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{source}.csv")
    test_df = pd.read_csv(f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test.csv")

    # Pre-group data by user (avoids repeated filtering)
    print("Pre-processing user data...")
    source_groups = source_data.groupby('reviewerID')
    train_groups = train_df.groupby('reviewerID')
    
    # Get unique users from test set
    test_users = set(test_df['reviewerID'].unique())
    
    # Pre-compute user data lookups
    user_source_cache = {}
    user_target_cache = {}
    
    for user in test_users:
        if user in source_groups.groups:
            user_source_cache[user] = source_groups.get_group(user).head(10)
        else:
            user_source_cache[user] = pd.DataFrame()
            
        if user in train_groups.groups:
            user_target_cache[user] = train_groups.get_group(user)
        else:
            user_target_cache[user] = pd.DataFrame()
    
    # Pre-compute negative sampling pools per user
    print("Pre-computing negative item pools...")
    all_items = set(train_df['title'].unique())
    user_negative_pools = {}
    
    for user in test_users:
        user_items = set(user_target_cache[user]['title']) if not user_target_cache[user].empty else set()
        user_negative_pools[user] = list(all_items - user_items)
    
    # Use list to collect results (not repeated concat)
    results = []
    total_rows = len(test_df)
    processed = 0
    
    print(f"Processing {total_rows} test interactions...")
    
    # Vectorized iteration using itertuples
    for row in test_df.itertuples():
        user = row.reviewerID
        interacted_item = row.title
        
        # Use pre-cached data (O(1) lookup vs O(n) scan)
        user_source_data = user_source_cache[user]
        
        # Fast negative sampling from pre-computed pool
        available_negatives = [item for item in user_negative_pools[user] if item != interacted_item]
        if len(available_negatives) >= neg_samples:
            non_interacted_items = random.sample(available_negatives, neg_samples)
        else:
            non_interacted_items = available_negatives
        
        # Generate formatted strings efficiently
        source_items_formatted = [f"title: {title}, rating: {rating}" 
                                for title, rating in zip(user_source_data['title'], user_source_data['overall'])]
        
        # Create candidate items and shuffle
        candidate_items = [interacted_item] + non_interacted_items
        random.shuffle(candidate_items)
        
        perfect_ranking = [interacted_item] + non_interacted_items
        items_format = ', '.join([f'Item{i+1}' for i in range(len(candidate_items))])
        
        # Generate prompt (no target injection)
        ranking_prompt = generate_ranking_prompt(
            source_items_formatted, [], candidate_items,  # Empty target_items for no injection
            items_format, prompt_context, source, target
        )
        
        # Append to list instead of DataFrame concat
        results.append({
            'prompt': ranking_prompt,
            'correct_ranking': perfect_ranking
        })
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{total_rows} interactions ({processed/total_rows*100:.1f}%)")
    
    # Single DataFrame creation at the end
    print("Creating final dataset...")
    ranking_data = pd.DataFrame(results)
    
    # Save efficiently
    output_path = f"{base_dir}/few_shot_data/{k_shot}_percent/{data_info}_data/{prompt_context}/{source}_to_{target}/{target}_test_ranking_no_injection.csv"
    ranking_data.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(results)} test ranking prompts (no injection)")
