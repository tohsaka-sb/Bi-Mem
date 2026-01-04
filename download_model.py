from sentence_transformers import SentenceTransformer
from bert_score import score

print("Downloading SentenceTransformer model: all-MiniLM-L6-v2...")
try:
    SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model downloaded and cached successfully.")
except Exception as e:
    print(f"Failed to download SentenceTransformer model: {e}")

print("\nDownloading BERT-score model...")
try:
    score(["hello"], ["world"], lang='en', verbose=True)
    print("BERT-score model downloaded and cached successfully.")
except Exception as e:
    print(f"Failed to download BERT-score model: {e}")

print("\nAll required models have been cached.")