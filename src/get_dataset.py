from datasets import load_dataset


def convert_to_jsonl(raw_datasets, save_path):
    for split, dataset in raw_datasets.items():
        dataset.to_json(f"{save_path}-{split}.jsonl")
        
def get_by_category(category: str, save_dir: str):
    raw_datasets = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        cache_dir="/home/kunal/code/CDR_Meets_LLMs/dataset/cache",
    )
    raw_datasets.to_json(f"{save_dir}/reviews_{category}.jsonl")
    
    meta_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        split="full",
        cache_dir="/home/kunal/code/CDR_Meets_LLMs/dataset/cache",
    )
    
    meta_dataset.to_json(f"{save_dir}/meta_{category}.jsonl")

def main():
    get_by_category("Books", "/home/kunal/code/CDR_Meets_LLMs/data")
    get_by_category("Movies_and_TV", "/home/kunal/code/CDR_Meets_LLMs/data")
    get_by_category("CDs_and_Vinyl", "/home/kunal/code/CDR_Meets_LLMs/data")
    get_by_category("Digital_Music", "/home/kunal/code/CDR_Meets_LLMs/data")
    get_by_category("Electronics", "/home/kunal/code/CDR_Meets_LLMs/data")
    get_by_category("Grocery_and_Gourmet_Food", "/home/kunal/code/CDR_Meets_LLMs/data")
    

# books_reviews = load_dataset(
#     "McAuley-Lab/Amazon-Reviews-2023",
#     "raw_review_Books",
#     split="full",
#     cache_dir="/home/kunal/code/CDR_Meets_LLMs/dataset/cache",
#     # trust_remote_code=True
# )
# books_reviews.to_json("/home/kunal/code/CDR_Meets_LLMs/data/reviews_Books.jsonl")

# print("Books Reviews count:", len(books_reviews))
# print("Example:", books_reviews[0])


# books_meta = load_dataset(
#     "McAuley-Lab/Amazon-Reviews-2023",
#     "raw_meta_Books",
#     split="full",
#     cache_dir="/home/kunal/code/CDR_Meets_LLMs/dataset/cache",
#     # trust_remote_code=True
# )

# books_meta.to_json("/home/kunal/code/CDR_Meets_LLMs/data/meta_Books.jsonl")
# print("\nBooks Metadata count:", len(books_meta))
# print("Example:", books_meta[0])


# movies_reviews = load_dataset(
#     "McAuley-Lab/Amazon-Reviews-2023",
#     "raw_review_Movies_and_TV",
#     split="full",
#     cache_dir="/home/kunal/code/CDR_Meets_LLMs/dataset/cache",
#     # trust_remote_code=True
# )

# movies_reviews.to_json("/home/kunal/code/CDR_Meets_LLMs/data/reviews_Movies_and_TV.jsonl")
# print("Movies & TV Reviews (10%) count:", len(movies_reviews))
# print("Example:", movies_reviews[0])

# movies_meta = load_dataset(
#     "McAuley-Lab/Amazon-Reviews-2023",
#     "raw_meta_Movies_and_TV",
#     split="full",
#     cache_dir="/home/kunal/code/CDR_Meets_LLMs/dataset/cache",
#     # trust_remote_code=True
# )
# movies_meta.to_json("/home/kunal/code/CDR_Meets_LLMs/data/meta_Movies_and_TV.jsonl")
# print("Movies & TV Metadata (10%) count:", len(movies_meta))

if __name__ == "__main__":
    main()