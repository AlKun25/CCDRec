#!/usr/bin/env python3
"""
Cross-domain prediction with OpenAI-style API using log probabilities and evaluation metrics.
Usage: python cross_domain_prediction.py --input_file path/to/input.csv --output_file path/to/output.csv

Dependencies: pip install pandas numpy scikit-learn openai
"""

import pandas as pd
import numpy as np
import ast
import re
import argparse
import logging
from typing import List, Dict, Tuple, Optional
import time
from sklearn.metrics import ndcg_score
from openai import OpenAI

# Configuration for API
API_CONFIG = {
    "base_url": "http://localhost:4000/v1",  # Change this for other providers
    "api_key": "EMPTY",  # Set this via environment variable
    "model": "google/gemma-3-4b-it",  # or gpt-4, etc.
    "max_tokens": 400,
    "temperature": 0.1,  # Low temperature for more consistent rankings
    "logprobs": True,
    "top_logprobs": 0  # Only sampled token logprobs, matching original behavior
}

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('log_file.log')

# Create a formatter and set the format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

class CrossDomainPredictor:
    def __init__(self, api_config: Dict):
        self.api_config = api_config
        self.client = OpenAI(
            api_key=api_config['api_key'],
            base_url=api_config['base_url']
        )
    
    def make_linear_scores(self, n: int) -> List[float]:
        """Create linear relevance scores from 1.0 to 0.0"""
        if n <= 1:
            return [1.0]
        step = 1.0 / (n - 1)
        return [1.0 - i * step for i in range(n)]
    
    def call_llm_api(self, prompt: str, max_retries: int = 3) -> Tuple[str, List]:
        """Call OpenAI-style API using official client with retry and validation"""
        return self.get_valid_ranking_with_retry(prompt, max_retries)
    
    def process_prompts_batch(self, prompts: List[str], batch_size: int = 16, 
                             delay_between_batches: float = 0.0) -> List[Tuple[str, List]]:
        """Process prompts in batches with rate limiting"""
        results = []
        total_prompts = len(prompts)
        
        # Process in batches
        for batch_start in range(0, total_prompts, batch_size):
            batch_end = min(batch_start + batch_size, total_prompts)
            batch_prompts = prompts[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(total_prompts-1)//batch_size + 1} "
                  f"(prompts {batch_start+1}-{batch_end})")
            
            batch_results = []
            for i, prompt in enumerate(batch_prompts):
                prompt_idx = batch_start + i + 1
                print(f"  Processing prompt {prompt_idx}/{total_prompts}")
                
                # Get valid response with retry logic
                content, logprobs = self.get_valid_ranking_with_retry(prompt)
                batch_results.append((content, logprobs))
                
                # Rate limiting within batch (adjust based on your API limits)
                if i < len(batch_prompts) - 1:  # Don't sleep after last request in batch
                    ...
                    # time.sleep(0.1)  # 10 requests per second
            
            results.extend(batch_results)
            
            # Delay between batches
            if batch_end < total_prompts:
                print(f"  Batch complete. Waiting {delay_between_batches}s before next batch...")
                # time.sleep(delay_between_batches)
        
        return results
    
    # def parse_and_validate_ranking(self, content: str) -> Optional[List[str]]:
    #     """Parse and validate ranking output"""
    #     try:
    #         # Try to parse as Python list
    #         if content.startswith('[') and content.endswith(']'):
    #             ranking = ast.literal_eval(content)
    #             if isinstance(ranking, list) and len(ranking) > 0:
    #                 return ranking
            
    #         # Try to extract list from text
    #         list_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
    #         if list_match:
    #             list_str = '[' + list_match.group(1) + ']'
    #             ranking = ast.literal_eval(list_str)
    #             if isinstance(ranking, list) and len(ranking) > 0:
    #                 return ranking
            
    #         # Try to extract quoted items
    #         quotes = re.findall(r'"([^"]*)"', content)
    #         if not quotes:
    #             quotes = re.findall(r"'([^']*)'", content)
            
    #         if quotes and len(quotes) >= 2:  # At least 2 items for a ranking
    #             return quotes
                
    #         return None
            
    #     except Exception as e:
    #         print(f"Parsing error: {e}")
    #         return None
    
    def parse_and_validate_ranking(self, content: str) -> Optional[List[str]]:
        """Parse and validate ranking output with robust error handling"""
        
        # Method 1: Extract quoted items (most reliable)
        quotes = re.findall(r'"([^"]*?)"', content)
        if not quotes:
            quotes = re.findall(r"'([^']*?)'", content)
        
        if quotes and len(quotes) >= 2:
            return quotes
        
        # Method 2: Try JSON parsing (safer than ast.literal_eval)
        try:
            import json
            if content.startswith('[') and content.endswith(']'):
                ranking = json.loads(content)
                if isinstance(ranking, list) and len(ranking) >= 2:
                    return ranking
        except Exception:
            pass
        
        # Method 3: Extract from bracketed content with manual parsing
        list_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
        if list_match:
            inner_content = list_match.group(1)
            # Split by comma and clean up
            items = [item.strip().strip('"\'') for item in inner_content.split(',')]
            items = [item for item in items if item]  # Remove empty items
            if len(items) >= 2:
                return items
        
        # Method 4: Line-by-line extraction
        lines = content.split('\n')
        items = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith(']'):
                # Remove quotes, numbers, bullets
                clean_line = re.sub(r'^[\d\.\-\*\+\s]*["\']?|["\']?\s*$', '', line)
                if clean_line:
                    items.append(clean_line)
        
        return items if len(items) >= 2 else None
    
    def validate_ranking_quality(self, ranking: List[str]) -> bool:
        """Advanced validation for ranking quality"""
        
        # Basic checks
        if not ranking or len(ranking) < 2:
            return False
        
        # Check for duplicates
        if len(ranking) != len(set(ranking)):
            return False
        
        # Check for reasonable string lengths
        if any(len(item.strip()) == 0 for item in ranking):
            return False
        
        # Check for non-empty strings
        if any(not item.strip() for item in ranking):
            return False
        
        return True
    
    def get_valid_ranking_with_retry(self, prompt: str, max_retries: int = 3) -> Tuple[str, List]:
        """Get valid ranking output with retry logic"""
        
        for attempt in range(max_retries):
            try:
                # Adjust temperature slightly on retries to get different outputs
                temp = self.api_config['temperature'] + (attempt * 0.1)
                
                response = self.client.chat.completions.create(
                    model=self.api_config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=min(temp, 1.0),  # Cap at 1.0
                    logprobs=self.api_config['logprobs'],
                    top_logprobs=self.api_config['top_logprobs'],
                    max_tokens=self.api_config['max_tokens']
                )
                
                content = response.choices[0].message.content.strip()
                logprobs_data = response.choices[0].logprobs.content if response.choices[0].logprobs else []
                
                # Validate the ranking output
                ranking = self.parse_and_validate_ranking(content)
                
                # Check if the ranking is valid
                if ranking: #and self.validate_ranking_quality(ranking):
                    print(f"✓ Valid ranking generated (attempt {attempt + 1})")
                    return content, logprobs_data
                else:
                    # print(f"✗ Invalid ranking (attempt {attempt + 1}): {content[:100]}...")
                    logger.error(f"Invalid ranking (attempt {attempt + 1}):\n{content}...")
                    
            except Exception as e:
                print(f"✗ API error (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < max_retries - 1:
                ...
                # time.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        print(f"⚠️ Failed to get valid ranking after {max_retries} attempts")
        return "", []
    
    def extract_token_logprobs_for_titles(self, titles: List[str], logprobs_data: List, 
                                         generated_text: str) -> Dict[str, float]:
        """
        Extract log probabilities for specific titles from the OpenAI client response.
        This mimics the original HuggingFace approach of getting per-token logprobs for each title.
        """
        title_logprobs = {}
        
        if not logprobs_data or not titles:
            return title_logprobs
        
        # Create a mapping of token positions to logprobs
        token_to_logprob = {}
        full_text = ""
        
        for i, token_data in enumerate(logprobs_data):
            # Handle OpenAI client response format
            token = token_data.token if hasattr(token_data, 'token') else token_data.get('token', '')
            logprob = token_data.logprob if hasattr(token_data, 'logprob') else token_data.get('logprob', -10.0)
            
            # Build the full text to track positions
            full_text += token
            token_to_logprob[i] = {
                'token': token,
                'logprob': logprob,
                'position': len(full_text) - len(token)
            }
        
        # For each title, find its tokens in the response and calculate average logprob
        for title in titles:
            # Look for the title in the generated text
            title_pattern = rf'["\']({re.escape(title)})["\']'
            match = re.search(title_pattern, generated_text)
            
            if not match:
                # Fallback: look for title without quotes
                title_start = generated_text.find(title)
                if title_start == -1:
                    title_logprobs[title] = -10.0  # Very low probability for missing titles
                    continue
            else:
                title_start = match.start(1)
            
            # Find tokens that correspond to this title
            title_tokens = []
            title_end = title_start + len(title)
            
            for token_idx, token_info in token_to_logprob.items():
                token_pos = token_info['position']
                token_end = token_pos + len(token_info['token'])
                
                # Check if this token overlaps with the title
                if (token_pos < title_end and token_end > title_start):
                    title_tokens.append(token_info['logprob'])
            
            if title_tokens:
                # Calculate average log probability for this title
                avg_logprob = np.mean(title_tokens)
                title_logprobs[title] = avg_logprob
            else:
                title_logprobs[title] = -10.0
        
        return title_logprobs
    
    def process_single_prompt(self, prompt: str) -> Tuple[List[str], List[str]]:
        """Process a single prompt and return generated and final orders"""
        try:
            # Generate initial ranking with logprobs and validation
            generated_text, logprobs_data = self.get_valid_ranking_with_retry(prompt)
            
            if not generated_text:
                print("Warning: No valid ranking generated for prompt")
                return [], []
            
            # Parse the validated ranking
            generated_order = self.parse_and_validate_ranking(generated_text)
            
            if not generated_order:
                print("Warning: Failed to parse ranking even after validation")
                return [], []
            
            # Calculate ranking scores (position-based)
            ranking_scores = {}
            scoring_values = self.make_linear_scores(len(generated_order))
            
            for i, title in enumerate(generated_order):
                if i < len(scoring_values):
                    ranking_scores[title] = scoring_values[i]
                else:
                    ranking_scores[title] = 0.0
            
            # Extract log probabilities for each title
            log_probs = self.extract_token_logprobs_for_titles(
                generated_order, logprobs_data, generated_text
            )
            
            # Normalize log probabilities to [0, 1] range for combination
            if log_probs:
                log_values = list(log_probs.values())
                log_min, log_max = min(log_values), max(log_values)
                
                if log_max == log_min:
                    normalized_log_probs = {title: 1.0 for title in log_probs}
                else:
                    normalized_log_probs = {
                        title: (log_prob - log_min) / (log_max - log_min)
                        for title, log_prob in log_probs.items()
                    }
            else:
                normalized_log_probs = {title: 0.0 for title in generated_order}
            
            # Combine scores (same as original: alpha * position + beta * logprob)
            alpha, beta = 0.5, 0.5
            composite_scores = {}
            
            for title in generated_order:
                rank_score = ranking_scores.get(title, 0.0)
                logprob_score = normalized_log_probs.get(title, 0.0)
                composite_scores[title] = alpha * rank_score + beta * logprob_score
            
            # Create final order based on composite scores
            final_order = sorted(
                generated_order,
                key=lambda t: composite_scores.get(t, 0.0),
                reverse=True
            )
            
            return generated_order, final_order
            
        except Exception as e:
            print(f"Error processing prompt: {e}")
            return [], []
    
    def calculate_mrr_at_k(self, predicted: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate Mean Reciprocal Rank at K"""
        if not predicted or not ground_truth:
            return 0.0
        
        predicted_k = predicted[:k]
        
        for i, item in enumerate(predicted_k):
            if item in ground_truth:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def calculate_ndcg_at_k(self, predicted: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        if not predicted or not ground_truth:
            return 0.0
        
        predicted_k = predicted[:k]
        
        # Create relevance scores (1 if in ground truth, 0 otherwise)
        relevance = [1 if item in ground_truth else 0 for item in predicted_k]
        
        # Pad to length k if necessary
        while len(relevance) < k:
            relevance.append(0)
        
        # Create ideal ranking (all relevant items first)
        ideal_relevance = sorted(relevance, reverse=True)
        
        try:
            # sklearn expects shape (1, k) for single sample
            ndcg = ndcg_score([ideal_relevance], [relevance], k=k)
            return ndcg
        except Exception:
            return 0.0
    
    def calculate_precision_at_k(self, predicted: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate Precision at K"""
        if not predicted or not ground_truth:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant_items = sum(1 for item in predicted_k if item in ground_truth)
        return relevant_items / min(k, len(predicted_k))
    
    def calculate_recall_at_k(self, predicted: List[str], ground_truth: List[str], k: int) -> float:
        """Calculate Recall at K"""
        if not predicted or not ground_truth:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant_items = sum(1 for item in predicted_k if item in ground_truth)
        return relevant_items / len(ground_truth)
    
    def calculate_metrics(self, predictions: List[List[str]], ground_truths: List[List[str]], 
                         k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Calculate MRR@K, NDCG@K, Precision@K, and Recall@K metrics"""
        metrics = {}
        
        for k in k_values:
            mrr_scores = []
            ndcg_scores = []
            precision_scores = []
            recall_scores = []
            
            for pred, gt in zip(predictions, ground_truths):
                mrr_scores.append(self.calculate_mrr_at_k(pred, gt, k))
                ndcg_scores.append(self.calculate_ndcg_at_k(pred, gt, k))
                precision_scores.append(self.calculate_precision_at_k(pred, gt, k))
                recall_scores.append(self.calculate_recall_at_k(pred, gt, k))
            
            metrics[f'MRR@{k}'] = np.mean(mrr_scores)
            metrics[f'NDCG@{k}'] = np.mean(ndcg_scores)
            metrics[f'Precision@{k}'] = np.mean(precision_scores)
            metrics[f'Recall@{k}'] = np.mean(recall_scores)
        
        return metrics
    
    def process_dataset(self, input_file: str, output_file: str, 
                       batch_size: int = 10, delay_between_batches: float = 1.0) -> pd.DataFrame:
        """Process entire dataset using batch processing"""
        print(f"Loading dataset from {input_file}")
        df = pd.read_csv(input_file)
        
        if 'prompt' not in df.columns:
            raise ValueError("Input file must contain a 'prompt' column")
        
        prompts = df['prompt'].tolist()
        print(f"Processing {len(prompts)} prompts in batches of {batch_size}...")
        
        # Process all prompts in batches with retry logic
        batch_results = self.process_prompts_batch(
            prompts, 
            batch_size=batch_size, 
            delay_between_batches=delay_between_batches
        )
        
        # Process the batch results to get final rankings
        generated_orders = []
        final_orders = []
        
        print("Processing batch results to create final rankings...")
        for i, (generated_text, logprobs_data) in enumerate(batch_results):
            print(f"Processing result {i+1}/{len(batch_results)}")
            
            if not generated_text:
                # Failed to get valid response
                generated_orders.append([])
                final_orders.append([])
                continue
            
            # Parse the validated ranking
            generated_order = self.parse_and_validate_ranking(generated_text)
            
            if not generated_order:
                generated_orders.append([])
                final_orders.append([])
                continue
            
            # Calculate ranking scores (position-based)
            ranking_scores = {}
            scoring_values = self.make_linear_scores(len(generated_order))
            
            for j, title in enumerate(generated_order):
                if j < len(scoring_values):
                    ranking_scores[title] = scoring_values[j]
                else:
                    ranking_scores[title] = 0.0
            
            # Extract log probabilities for each title
            log_probs = self.extract_token_logprobs_for_titles(
                generated_order, logprobs_data, generated_text
            )
            
            # Normalize log probabilities to [0, 1] range for combination
            if log_probs:
                log_values = list(log_probs.values())
                log_min, log_max = min(log_values), max(log_values)
                
                if log_max == log_min:
                    normalized_log_probs = {title: 1.0 for title in log_probs}
                else:
                    normalized_log_probs = {
                        title: (log_prob - log_min) / (log_max - log_min)
                        for title, log_prob in log_probs.items()
                    }
            else:
                normalized_log_probs = {title: 0.0 for title in generated_order}
            
            # Combine scores (same as original: alpha * position + beta * logprob)
            alpha, beta = 0.5, 0.5
            composite_scores = {}
            
            for title in generated_order:
                rank_score = ranking_scores.get(title, 0.0)
                logprob_score = normalized_log_probs.get(title, 0.0)
                composite_scores[title] = alpha * rank_score + beta * logprob_score
            
            # Create final order based on composite scores
            final_order = sorted(
                generated_order,
                key=lambda t: composite_scores.get(t, 0.0),
                reverse=True
            )
            
            generated_orders.append(generated_order)
            final_orders.append(final_order)
        
        # Add results to dataframe (matching original column names)
        df['generated'] = generated_orders
        df['final order'] = final_orders
        
        # Save results (matching original code format)
        print(f"Saving results to {output_file}")
        df.to_csv(output_file, index=False, encoding="utf-8")
        
        return df
    
    def calculate_and_save_metrics(self, df: pd.DataFrame, output_file: str, 
                                  ground_truth_column: str, k_values: List[int] = [1, 3, 5, 10]):
        """Calculate and save metrics separately after processing"""
        print("Calculating metrics...")
        
        if ground_truth_column not in df.columns:
            print(f"Warning: Ground truth column '{ground_truth_column}' not found in dataframe")
            return
        
        # Convert ground truth to lists if they're strings
        ground_truths = []
        for gt in df[ground_truth_column]:
            if isinstance(gt, str):
                try:
                    gt_list = ast.literal_eval(gt)
                except Exception as e:
                    print(f"Error parsing ground truth: {e}")
                    gt_list = [gt]  # Single item
            else:
                gt_list = gt if isinstance(gt, list) else [gt]
            ground_truths.append(gt_list)
        
        # Get predictions from the dataframe
        generated_orders = df['generated'].tolist()
        final_orders = df['final order'].tolist()
        
        # Calculate metrics for both generated and final orders
        gen_metrics = self.calculate_metrics(generated_orders, ground_truths, k_values)
        final_metrics = self.calculate_metrics(final_orders, ground_truths, k_values)
        
        print("\nGenerated Order Metrics:")
        for metric, value in gen_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nFinal Order Metrics:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save detailed metrics to a separate file
        metrics_df = pd.DataFrame([
            {'Method': 'Generated', **gen_metrics},
            {'Method': 'Final', **final_metrics}
        ])
        metrics_file = output_file.replace('.csv', '_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Detailed metrics saved to {metrics_file}")
        
        return gen_metrics, final_metrics


def main():
    parser = argparse.ArgumentParser(description="Cross-domain prediction with OpenAI API and log probabilities")
    parser.add_argument("--input_file", required=True, help="Input CSV file with prompts")
    parser.add_argument("--output_file", required=True, help="Output CSV file")
    parser.add_argument("--api_key", help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--base_url", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--ground_truth_column", help="Column name containing ground truth rankings")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--top_logprobs", type=int, default=0, help="Number of top log probabilities to return (0 matches original HF behavior - only sampled tokens)")
    
    args = parser.parse_args()
    
    # Set up API configuration
    # import os
    api_key = API_CONFIG['api_key'] #args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via --api_key or OPENAI_API_KEY env var")
    
    # api_config = {
    #     "base_url": args.base_url,
    #     "api_key": api_key,
    #     "model": args.model,
    #     "max_tokens": args.max_tokens,
    #     "temperature": args.temperature,
    #     "logprobs": True,
    #     "top_logprobs": args.top_logprobs
    # }
    
    api_config = API_CONFIG
    
    # Create predictor and process dataset
    predictor = CrossDomainPredictor(api_config)
    # result_df = predictor.process_dataset(args.input_file, args.output_file)
    
    result_df = pd.read_csv(args.output_file)
    
    # Calculate metrics separately if ground truth is provided
    if args.ground_truth_column:
        predictor.calculate_and_save_metrics(
            result_df, 
            args.output_file,
            args.ground_truth_column
        )
    
    print("Processing complete!")


if __name__ == "__main__":
    main()