# Information Retrieval System using TF-IDF + BM25 by Avaya Khatri

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# Load dataset
df = pd.read_csv("IMDB_cleaned.csv")
df = df.dropna(subset=["review"])
documents = df["review"].tolist()

# Simple preprocessing: lowercase and split into words
def preprocess(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words

# Prepare documents for BM25
tokenized_docs = [preprocess(doc) for doc in documents]

# Build indexes
bm25 = BM25Okapi(tokenized_docs) 
tfidf_vectorizer = TfidfVectorizer(stop_words='english')  
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Search function
def search(query, top_k=5):
    # Preprocess query
    q_words = preprocess(query)
    
    # Get BM25 scores
    bm25_scores = bm25.get_scores(q_words)
    
    # Get TF-IDF scores
    query_vector = tfidf_vectorizer.transform([query])
    tfidf_scores = (tfidf_matrix @ query_vector.T).toarray().flatten()
    
    # Combine scores (60% BM25, 40% TF-IDF)
    combined_scores = 0.6 * bm25_scores + 0.4 * tfidf_scores
    
    # Get top results
    top_indices = combined_scores.argsort()[::-1][:top_k]
    
    # Format results
    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "doc_id": idx,
            "score": round(combined_scores[idx], 4),
            "snippet": documents[idx][:200]
        })
    
    return results

# Example usage
if __name__ == "__main__":
    print("Movie Review Search System")
    
    query = "exciting action movie with great special effects"
    print(f"\nSearching for: '{query}'\n")
    
    results = search(query, top_k=3)
    
    for r in results:
        print(f"{r['rank']}. Document #{r['doc_id']} (Score: {r['score']})")
        print(f"   Preview: {r['snippet']}\n")