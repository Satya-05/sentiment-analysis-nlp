from datasets import load_dataset
import pandas as pd

# Load full Magazine_Subscriptions category (~71.5K reviews, ~90MB)
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Magazine_Subscriptions",
    trust_remote_code=True,
    split="full"
)

df = pd.DataFrame(dataset)

print("Full shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Rating distribution:\n", df['rating'].value_counts().sort_index())

# Keep only the columns relevant to sentiment classification
df_filtered = df[['rating', 'title', 'text']].copy()

# Drop rows with missing/empty review text
df_filtered = df_filtered.dropna(subset=['text'])
df_filtered = df_filtered[df_filtered['text'].str.strip() != '']

print("\nFiltered shape (after dropping empty text):", df_filtered.shape)

df_filtered.to_csv("data/amazon_reviews_raw.csv", index=False)
print("Saved to data/amazon_reviews_raw.csv")