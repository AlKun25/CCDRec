import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class ReviewDatasetProcessor:
    def __init__(self, name, review_path, meta_path, output_dir):
        self.name = name
        self.review_path = review_path
        self.meta_path = meta_path
        self.output_dir = output_dir
        self.reviews = pd.DataFrame()
        self.meta = pd.DataFrame()
        self.filtered_reviews = pd.DataFrame()

    def load_reviews(self):
        print(f"[{self.name}] Loading reviews from: {self.review_path}")
        records = []
        try:
            with open(self.review_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            record = json.loads(line)
                            reviewer_id = record.get("user_id")
                            asin = record.get("parent_asin")
                            rating = record.get("rating")

                            if reviewer_id and asin and rating is not None:
                                records.append(
                                    {
                                        "reviewerID": reviewer_id,
                                        "parent_asin": asin,
                                        "overall": rating,
                                    }
                                )

                        except json.JSONDecodeError:
                            if i % 100000 == 0:
                                print(f"[{self.name}] Skipping malformed line {i + 1}")
                            continue
                    if i % 500000 == 0:
                        print(f"[{self.name}] Processed {i} lines...")
        except FileNotFoundError:
            print(f"[{self.name}] File not found: {self.review_path}")
        self.reviews = pd.DataFrame(records)
        print(f"[{self.name}] Loaded {len(self.reviews)} valid review records")

    def filter_users(self, valid_users):
        self.filtered_reviews = self.reviews[
            self.reviews["reviewerID"].isin(valid_users)
        ]

    def filter_min_reviews(self, min_reviews: int = 10):
        counts = self.filtered_reviews["reviewerID"].value_counts()
        active_users = counts[counts >= min_reviews].index
        self.filtered_reviews = self.filtered_reviews[
            self.filtered_reviews["reviewerID"].isin(active_users)
        ]

    def sample_users(self, user_list, fraction, seed=42):
        sampled = pd.Series(user_list).sample(frac=fraction, random_state=seed)
        self.filtered_reviews = self.filtered_reviews[
            self.filtered_reviews["reviewerID"].isin(sampled)
        ]
        return sampled.tolist()

    def load_metadata(self):
        print(f"[{self.name}] Loading metadata from: {self.meta_path}")
        records = []
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        asin = record.get("parent_asin")
                        title = record.get("title")
                        if asin and title:
                            records.append({"parent_asin": asin, "title": title})
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"[{self.name}] File not found: {self.meta_path}")
        self.meta = pd.DataFrame(records)
        print(f"[{self.name}] Loaded {len(self.meta)} metadata records")

    def filter_metadata_by_reviews(self):
        self.meta = self.meta[
            self.meta["parent_asin"].isin(self.filtered_reviews["parent_asin"])
        ]

    def save_to_jsonl(self):
        reviews_out = os.path.join(self.output_dir, f"subset_reviews_{self.name}.jsonl")
        meta_out = os.path.join(self.output_dir, f"subset_meta_{self.name}.jsonl")
        self.filtered_reviews.to_json(reviews_out, orient="records", lines=True)
        self.meta.to_json(meta_out, orient="records", lines=True)
        print(
            f"[{self.name}] Saved {len(self.filtered_reviews)} reviews to {reviews_out}"
        )
        print(f"[{self.name}] Saved {len(self.meta)} metadata records to {meta_out}")


def main():
    base_path = os.getenv("BASE_DIR")
    output_dir = os.path.join(base_path, "dataset")
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate processors
    books = ReviewDatasetProcessor(
        name="Books",
        review_path=os.path.join(base_path, "data", "reviews_Books.jsonl"),
        meta_path=os.path.join(base_path, "data", "meta_Books.jsonl"),
        output_dir=output_dir,
    )

    movies = ReviewDatasetProcessor(
        name="Movies_and_TV",
        review_path=os.path.join(base_path, "data", "reviews_Movies_and_TV.jsonl"),
        meta_path=os.path.join(base_path, "data", "meta_Movies_and_TV.jsonl"),
        output_dir=output_dir,
    )

    # Load review data
    books.load_reviews()
    movies.load_reviews()

    # Find common users
    common_users = set(books.reviews["reviewerID"]).intersection(
        movies.reviews["reviewerID"]
    )
    print(f"\nFound {len(common_users)} users in both domains.")

    # Filter both datasets to common users
    books.filter_users(common_users)
    movies.filter_users(common_users)

    # Keep only users with >= N reviews in each domain
    min_reviews = 5
    books.filter_min_reviews(min_reviews)
    movies.filter_min_reviews(min_reviews)

    common_active_users = set(books.filtered_reviews["reviewerID"]).intersection(
        movies.filtered_reviews["reviewerID"]
    )

    print(
        f"\nUsers with ≥{min_reviews} reviews in both domains: {len(common_active_users)}"
    )

    # Re-filter based on updated common users
    books.filter_users(common_active_users)
    movies.filter_users(common_active_users)

    # Sample 10% of users
    # sample_fraction = 0.10
    # sampled_users = books.sample_users(common_active_users, sample_fraction)
    # movies.sampled_reviews = movies.filtered_reviews[movies.filtered_reviews['reviewerID'].isin(sampled_users)]

    # print(f"\nSampled {len(sampled_users)} users ({sample_fraction*100}%)")

    # Load metadata and filter
    books.load_metadata()
    movies.load_metadata()
    books.filter_metadata_by_reviews()
    movies.filter_metadata_by_reviews()

    # Save all outputs
    print("\nSaving outputs...")
    books.save_to_jsonl()
    movies.save_to_jsonl()

    print("\n✅ Dataset preparation complete!")


if __name__ == "__main__":
    main()
