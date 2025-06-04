from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

def convert_to_jsonl(raw_datasets, save_path):
    for split, dataset in raw_datasets.items():
        dataset.to_json(f"{save_path}-{split}.jsonl")
        
def download_by_category(category: str, save_dir: str):
    raw_datasets = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        cache_dir="/home/kunal/code/CCDRec/dataset/cache",
    )
    raw_datasets.to_json(f"{save_dir}/reviews_{category}.jsonl")
    
    meta_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        split="full",
        cache_dir="/home/kunal/code/CCDRec/dataset/cache",
    )
    
    meta_dataset.to_json(f"{save_dir}/meta_{category}.jsonl")

def main():
    base_dir = os.getenv('BASE_DIR')
    # download_by_category("Books", f"{base_dir}/data")
    # download_by_category("Movies_and_TV", f"{base_dir}/data")
    # download_by_category("CDs_and_Vinyl", f"{base_dir}/data")
    # download_by_category("Electronics", f"{base_dir}/data")
    # download_by_category("Grocery_and_Gourmet_Food", f"{base_dir}/data")
    


if __name__ == "__main__":
    main()