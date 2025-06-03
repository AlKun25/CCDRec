import os
import re
import math
import time
import torch
import gc
from math import sqrt
from openai import OpenAI
from rating_prediction_prompts import transform_data
from fine_tune import load_train_val_data
from peft import PeftModel, LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM
from dotenv import load_dotenv
load_dotenv()

def gpt_evaluate_llm(k_shot, source, target, data_info, model_name, rating_ranking, injection, prompt_context):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fineune_dataset_dict = load_train_val_data(k_shot, source, target, data_info, rating_ranking, injection, prompt_context)
    test_data = fineune_dataset_dict['test'].to_pandas()

    # Use environment variable for API key - more secure
    # api_key = os.getenv('OPENAI_API_KEY')
    # if not api_key:
        # raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:4000/v1")

    inputs = test_data['prompt'].values
    inputs = inputs[:1000]
    ground_truths = None

    if rating_ranking == 'rating':
        ground_truths = test_data['ground_truth'].values
        ground_truths = ground_truths[:1000]

    elif rating_ranking == 'ranking':
        gt = test_data['correct_ranking'].values
        gt = gt[:1000]
        ground_truths = []
        for i in gt:
            i = i.replace(", nan,", ", 'NA',")
            ranked_list = eval(i)
            ground_truths.append(str(ranked_list[0]))

    # Clear test_data from memory after extracting what we need
    del test_data, fineune_dataset_dict
    gc.collect()

    count = 0
    model = ""
    if model_name in ['GPT-4o', 'GPT-4', 'gpt-3.5-turbo', "google/gemma-3-4b-it"]:
        model = model_name.lower()

    best_mae = 1000
    best_rmse = 1000

    total_count = 0

    worst_mae = 0
    worst_rmse = 0

    best_mrr_5 = float('-inf')
    worst_mrr_5 = float('inf')
    best_ndcg_5 = float('-inf')
    worst_ndcg_5 = float('inf')

    best_mrr_10 = float('-inf')
    worst_mrr_10 = float('inf')
    best_ndcg_10 = float('-inf')
    worst_ndcg_10 = float('inf')

    for i in range(2):

        print(f"\n\n Iteration {i+1}/2\n\n")

        base_preds = []

        for prompt_idx, prompt in enumerate(inputs):
            if prompt is None:
                continue

            # Add progress indicator to monitor memory usage
            if prompt_idx % 100 == 0:
                print(f"Processing prompt {prompt_idx}/1000")
                gc.collect()  # Force garbage collection periodically

            system_prompt = ""

            if rating_ranking == 'rating':

                    if prompt_context == 'none':
                        system_prompt = "You will output an option."

                    elif prompt_context == 'medium':
                        if injection == 'with':
                            system_prompt = "Output an option from the prompt."
                        elif injection == 'no':
                            system_prompt = "Output an option from the prompt."
                            
                    elif prompt_context == 'high':
                        if injection == 'with':
                            system_prompt = "You are a cross-domain recommender system. Your task is to understand user behavior from a source domain and apply that knowledge to recommend items in a target domain. Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', or 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Choose the option that best reflects the user's engagement potential for the candiate item."
                        elif injection == 'no':
                            system_prompt = "You are a cross-domain recommender system. Your task is to understand user behavior from a source domain and apply that knowledge to recommend items in a target domain. Output one of the following options: 'Very Unlikely', 'Unlikely', 'Somewhat Unlikely', 'Neutral', 'Likely', or 'Highly Likely'. These options represent the likelihood of the user interacting with the recommended item, based on the information provided. Choose the option that best reflects the user's engagement potential for the candiate item."
                    
            elif rating_ranking == 'ranking':

                if prompt_context == 'none':
                    system_prompt = "You will output a list"

                elif prompt_context == 'medium':
                    system_prompt = "You will output a ranked list, where items will be ranked from most likely to interact with to least likely to interact with."

                elif prompt_context == 'high':
                    system_prompt = "You are a cross-domain recommender system. Your task is to understand user behavior from a source domain and apply that knowledge to recommend items in a target domain. Output a ranked list of candidate items, with the item the user is most likely to interact with at the top and the item they are least likely to interact with at the bottom. Ensure the list follows the expected format provided in the prompt."

            # Reduced max retries to prevent excessive API calls
            max_retries = 3  # Reduced from 10 to 3
            retry_delay = 2  # Increased initial delay

            temperature = 0.0

            if prompt_context == 'high' and injection == 'with':
                temperature = 0.3
            elif prompt_context == 'high' and injection == 'no':
                temperature = 0.3
            elif prompt_context == 'medium' and injection == 'with':
                temperature = 0.3
            elif prompt_context == 'medium' and injection == 'no':
                temperature = 0.3

            for attempt in range(max_retries):
                try:
                    # Call the API and handle response
                    response = client.chat.completions.create(
                        model="google/gemma-3-4b-it",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ], 
                        temperature=temperature
                    )

                    # Accessing the content of the first choice in the response
                    output = response.choices[0].message.content
                    base = output.lower()

                    base_preds.append(base)
                    total_count += 1
                    break 

                except Exception as e:
                    print(f"Error calling API (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5  # Reduced exponential backoff
                    else:
                        print(f"Failed to get response for prompt {prompt_idx} after {max_retries} attempts")
                        base_preds.append("")  # Add empty response to maintain list alignment
                        break

        if rating_ranking == 'rating':

            base_mae = 0.0
            base_rmse = 0.0
            valid_predictions = 0  # Track valid predictions
            
            for x, y in zip(base_preds, ground_truths):
                if not x:  # Skip empty responses from failed API calls
                    continue
                    
                candidate_item_rating = 0.0

                if 'highly likely' in x:
                    candidate_item_rating = 5.0

                elif 'very unlikely' in x:
                    candidate_item_rating = 0.5

                elif 'somewhat unlikely' in x:
                    candidate_item_rating = 2.5

                elif 'unlikely' in x:
                    candidate_item_rating = 1.5

                elif 'likely' in x:
                    candidate_item_rating = 4.5

                elif 'neutral' in x:
                    candidate_item_rating = 3.0

                base_mae += abs(candidate_item_rating - y)
                base_rmse += (candidate_item_rating - y) ** 2
                valid_predictions += 1
            
            if valid_predictions > 0:
                base_mae /= valid_predictions
                base_rmse = (base_rmse / valid_predictions) ** 0.5

                if base_mae < best_mae:
                    best_mae = base_mae
                if base_rmse < best_rmse:
                    best_rmse = base_rmse

                if base_mae > worst_mae:
                    worst_mae = base_mae
                if base_rmse > worst_rmse:
                    worst_rmse = base_rmse
            else:
                print(f"Warning: No valid predictions in iteration {i+1}")

        if rating_ranking == 'ranking':

            total_mrr_5 = 0.0
            total_ndcg_5 = 0.0

            total_mrr_10 = 0.0
            total_ndcg_10 = 0.0
            
            valid_rankings = 0  # Track valid rankings

            for pred, truth in zip (base_preds, ground_truths):
                if not pred:  # Skip empty responses from failed API calls
                    continue
                    
                pred_string = pred.lower().strip()
                ground_truth = truth.lower().strip()

                predictions = [item.strip() for item in pred_string.split(",") if item.strip()]

                position = -1.0
                count = 1.0
                for pred in predictions:
                    if ground_truth in str(pred):
                        position = count
                        break
                    count += 1.0

                mrr_5 = 1 / position if (position > 0 and position <= 5) else 0
                mrr_10 = 1 / position if (position > 0 and position <= 10) else 0

                ndcg_5 = 0
                ndcg_10 = 0

                if position > 0 and position <= 5:
                    # Calculate DCG@5
                    dcg_at_5 = 1 / math.log2(position + 1)

                    # Calculate IDCG (ideal situation is having the relevant item at the top)
                    idcg = 1 / math.log2(2) 

                    # Calculate NDCG@5
                    ndcg_at_5 = dcg_at_5 / idcg
                    ndcg_5 = ndcg_at_5

                if position > 0 and position <= 10:
                    # Calculate DCG@10
                    dcg_at_10 = 1 / math.log2(position + 1)

                    # Calculate IDCG (ideal situation is having the relevant item at the top)
                    idcg = 1 / math.log2(2) 

                    # Calculate NDCG@10
                    ndcg_at_10 = dcg_at_10 / idcg
                    ndcg_10 = ndcg_at_10

                total_mrr_5 += mrr_5
                total_ndcg_5 += ndcg_5

                total_mrr_10 += mrr_10
                total_ndcg_10 += ndcg_10
                valid_rankings += 1

            if valid_rankings > 0:
                total_mrr_5 /= valid_rankings
                total_ndcg_5 /= valid_rankings

                total_mrr_10 /= valid_rankings
                total_ndcg_10 /= valid_rankings

                # Update best and worst values for MRR@5 and NDCG@5
                if total_mrr_5 > best_mrr_5:
                    best_mrr_5 = total_mrr_5
                if total_ndcg_5 > best_ndcg_5:
                    best_ndcg_5 = total_ndcg_5

                if total_mrr_5 < worst_mrr_5:
                    worst_mrr_5 = total_mrr_5
                if total_ndcg_5 < worst_ndcg_5:
                    worst_ndcg_5 = total_ndcg_5

                # Update best and worst values for MRR@10 and NDCG@10
                if total_mrr_10 > best_mrr_10:
                    best_mrr_10 = total_mrr_10
                if total_ndcg_10 > best_ndcg_10:
                    best_ndcg_10 = total_ndcg_10

                if total_mrr_10 < worst_mrr_10:
                    worst_mrr_10 = total_mrr_10
                if total_ndcg_10 < worst_ndcg_10:
                    worst_ndcg_10 = total_ndcg_10
            else:
                print(f"Warning: No valid rankings in iteration {i+1}")

        # Clear predictions from memory after each iteration
        del base_preds
        gc.collect()
        
        print(f"Completed iteration {i+1}/2")

    print("\n\n----- Final Statistics -----\n\n")

    if rating_ranking == "rating":

        print("Base Model *BEST* MAE:", best_mae)
        print("Base Model *BEST* RMSE:", sqrt(best_rmse))

        print()

        print("Base Model *WORST* MAE:", worst_mae)
        print("Base Model *WORST* RMSE:", sqrt(worst_rmse))

        print()

    elif rating_ranking == "ranking":

        print("Base Model *BEST* MRR@5: " + str(best_mrr_5))
        print("Base Model *BEST* NDCG@5: " + str(best_ndcg_5))

        print()

        print("Base Model *BEST* MRR@10: " + str(best_mrr_10))
        print("Base Model *BEST* NDCG@10: " + str(best_ndcg_10))

        print()

        print("Base Model *WORST* MRR@5: " + str(worst_mrr_5))
        print("Base Model *WORST* NDCG@5: " + str(worst_ndcg_5))

        print()

        print("Base Model *WORST* MRR@10: " + str(worst_mrr_10))
        print("Base Model *WORST* NDCG@10: " + str(worst_ndcg_10))

    print("Total count is: " + str(total_count))





