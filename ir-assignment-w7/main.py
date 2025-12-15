#IR system evaluation

import pandas as pd
import numpy as np
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# 1. DATA LOADING & PREPROCESSING
print("[INFO] Loading IMDB dataset...")
df = pd.read_csv("IMDB_cleaned.csv")
df = df.dropna(subset=["review"])
documents = df["review"].tolist()
doc_ids = list(range(len(documents)))

def preprocess(text):
    """Converts text to lowercase and extracts alphanumeric tokens."""
    text = str(text).lower()
    words = re.findall(r'\b\w+\b', text)
    return words

print(f"[INFO] Preprocessing {len(documents)} documents...")
tokenized_docs = [preprocess(doc) for doc in documents]

# 2. BUILDING RETRIEVAL INDEXES
print("[INFO] Building BM25 and TF-IDF indexes...")
bm25 = BM25Okapi(tokenized_docs)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 3. HYBRID SEARCH FUNCTION
def hybrid_search(query, top_k=10):
    """
    Retrieves top-k documents using the hybrid BM25+TF-IDF model.
    Returns: list of document IDs sorted by descending relevance score.
    """
    q_words = preprocess(query)
    
    # BM25 scoring
    bm25_scores = bm25.get_scores(q_words)
    
    # TF-IDF scoring (cosine similarity)
    query_vector = tfidf_vectorizer.transform([query])
    tfidf_scores = (tfidf_matrix @ query_vector.T).toarray().flatten()
    
    # Combine scores (60% BM25, 40% TF-IDF)
    combined_scores = 0.6 * bm25_scores + 0.4 * tfidf_scores
    
    # Get top-k document indices
    top_indices = combined_scores.argsort()[::-1][:top_k]
    return top_indices.tolist()

# 4. RELEVANCE JUDGMENT HEURISTIC
def get_relevant_docs(query, tokenized_docs, threshold=0.3):
    """
    Heuristic to determine relevant documents for a query.
    A doc is relevant if it contains at least `threshold` fraction of unique query terms.
    Returns: set of relevant document IDs.
    """
    query_terms = set(preprocess(query))
    if not query_terms:
        return set()
    
    relevant_set = set()
    for doc_id, doc_tokens in enumerate(tokenized_docs):
        doc_token_set = set(doc_tokens)
        match_ratio = len(query_terms.intersection(doc_token_set)) / len(query_terms)
        if match_ratio >= threshold:
            relevant_set.add(doc_id)
    return relevant_set

# 5. EVALUATION METRICS
def precision_at_k(retrieved, relevant, k):
    """Calculates Precision@k."""
    if k == 0:
        return 0.0
    rel_in_topk = len(set(retrieved[:k]).intersection(relevant))
    return rel_in_topk / k

def recall_at_k(retrieved, relevant, k):
    """Calculates Recall@k."""
    if len(relevant) == 0:
        return 0.0
    rel_in_topk = len(set(retrieved[:k]).intersection(relevant))
    return rel_in_topk / len(relevant)

def average_precision(retrieved, relevant, k=10):
    """Calculates Average Precision (AP) for a single query."""
    if len(relevant) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            num_hits += 1
            score += num_hits / i
    return score / len(relevant)

def dcg_at_k(retrieved, relevant, k):
    """Calculates Discounted Cumulative Gain (DCG) at position k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            # Relevance grade = 1 for relevant, 0 for non-relevant
            rel = 1
            dcg += rel / math.log2(i + 1)  # log base 2 discount
    return dcg

def ndcg_at_k(retrieved, relevant, k):
    """Calculates Normalized Discounted Cumulative Gain (nDCG) at position k."""
    # Ideal DCG: top-k positions filled with relevant documents
    ideal_rel_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_rel_count + 1))
    
    if idcg == 0:
        return 0.0
    
    dcg = dcg_at_k(retrieved, relevant, k)
    return dcg / idcg

# 6. MAIN EVALUATION LOOP
print("\n" + "="*60)
print("PERFORMANCE EVALUATION OF HYBRID IR SYSTEM")
print("="*60)

# Define test queries
test_queries = [
    "funny comedy movie",
    "scary horror film",
    "great acting and emotional story",
    "action movie with special effects",
    "boring and disappointing plot"
]

K = 10  # Evaluation depth
results = []

print(f"\nEvaluating on {len(test_queries)} queries at depth k={K}...")
print("-"*60)

for query in test_queries:
    # Step 1: Retrieve documents using the hybrid model
    retrieved = hybrid_search(query, top_k=K)
    
    # Step 2: Determine relevant documents (simulated ground truth)
    relevant = get_relevant_docs(query, tokenized_docs, threshold=0.3)
    
    # Step 3: Calculate all metrics
    p_at_k = precision_at_k(retrieved, relevant, K)
    r_at_k = recall_at_k(retrieved, relevant, K)
    ap = average_precision(retrieved, relevant, K)
    ndcg = ndcg_at_k(retrieved, relevant, K)
    
    # Store results
    results.append({
        "Query": query,
        "Retrieved": retrieved[:5],  # Show first 5 retrieved IDs for inspection
        "Relevant_Count": len(relevant),
        "P@10": round(p_at_k, 4),
        "R@10": round(r_at_k, 4),
        "AP": round(ap, 4),
        "nDCG@10": round(ndcg, 4)
    })
    
    # Print per-query summary
    print(f"Query: '{query}'")
    print(f"  Relevant Docs: {len(relevant)} | Retrieved (Top-5): {retrieved[:5]}")
    print(f"  P@10: {p_at_k:.4f} | R@10: {r_at_k:.4f} | AP: {ap:.4f} | nDCG@10: {ndcg:.4f}")
    print()

# 7. SUMMARY & FINAL METRICS
results_df = pd.DataFrame(results)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.width', 120)

print("\n" + "="*60)
print("DETAILED EVALUATION RESULTS")
print("="*60)
print(results_df.to_string(index=False))

# Calculate overall system performance
map_score = results_df['AP'].mean()
mean_ndcg = results_df['nDCG@10'].mean()

print("\n" + "="*60)
print("SYSTEM SUMMARY METRICS")
print("="*60)
print(f"Mean Average Precision (MAP): {map_score:.4f}")
print(f"Average nDCG@10: {mean_ndcg:.4f}")
print("="*60)