import os
import re
from collections import defaultdict
from math import log

#Preprocessing
def preprocess(text):
    """Tokenize and lowercase text."""
    return re.findall(r'\b\w+\b', text.lower())

#Loading documents
def load_documents(folder_path):
    """Load all .txt files in the folder and preprocess them."""
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                docs[filename] = preprocess(f.read())
    return docs

#loading queries
def load_queries(query_file_path):
    """Load queries from a .txt file (one query per line)."""
    with open(query_file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

#computing statistics
def compute_statistics(docs):
    """Compute TF, DF, document length, and average document length."""
    term_freq = defaultdict(lambda: defaultdict(int))
    doc_freq = defaultdict(int)
    doc_len = {}
    total_terms = 0

    for doc_id, words in docs.items():
        doc_len[doc_id] = len(words)
        total_terms += len(words)

        unique_terms = set(words)

        for w in words:
            term_freq[doc_id][w] += 1

        for w in unique_terms:
            doc_freq[w] += 1

    avg_len = total_terms / len(docs)

    return term_freq, doc_freq, doc_len, avg_len

#BM25 Scoring
def bm25_score(query_terms, term_freq, doc_freq, doc_len, avg_len, total_docs, k1=1.5, b=0.75):
    """Compute BM25 scores."""
    scores = {}

    for doc_id in term_freq:
        score = 0

        for term in query_terms:
            if term in term_freq[doc_id]:
                tf = term_freq[doc_id][term]
                df = doc_freq.get(term, 0)

                idf = log((total_docs - df + 0.5) / (df + 0.5) + 1)

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len[doc_id] / avg_len))

                score += idf * (numerator / denominator)

        scores[doc_id] = score

    return scores

#Jelinek Mercer Language Model Scoring
def jm_score(query_terms, term_freq, doc_len, collection_freq, collection_len, lam=0.7):
    """Compute Jelinek-Mercer smoothing LM score (LOG version to avoid underflow)."""
    scores = {}

    for doc_id in term_freq:
        score = 0.0  # log space

        for term in query_terms:
            p_doc = term_freq[doc_id].get(term, 0) / doc_len[doc_id]
            p_coll = collection_freq.get(term, 0) / collection_len

            p = lam * p_doc + (1 - lam) * p_coll
            if p > 0:
                score += log(p)

        scores[doc_id] = score

    return scores


#Ranking result
def rank_results(scores, top_k=0):
    """
    Rank documents based on scores.
    top_k = 0 --> show all documents.
    """
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_k == 0:
        return ranked
    return ranked[:top_k]

def retrieve_documents(folder_path, query_file_path, top_k=0):
    """Main retrieval pipeline."""
    docs = load_documents(folder_path)
    queries = load_queries(query_file_path)

    term_freq, doc_freq, doc_len, avg_len = compute_statistics(docs)
    total_docs = len(docs)

    # Build collection model for LM
    collection_freq = defaultdict(int)
    collection_len = 0

    for doc_id in docs:
        for term in docs[doc_id]:
            collection_freq[term] += 1
            collection_len += 1

    # Process queries
    for query in queries:
        print("\n==============================================")
        print(f"QUERY: {query}")
        print("==============================================")

        query_terms = preprocess(query)

        # BM25
        bm25_scores = bm25_score(query_terms, term_freq, doc_freq, doc_len, avg_len, total_docs)
        bm25_ranked = rank_results(bm25_scores, top_k)

        print("\n--- BM25 RESULTS ---")
        for doc, score in bm25_ranked:
            print(f"{doc}: {score:.5f}")

        # LM Jelinek-Mercer
        jm_scores = jm_score(query_terms, term_freq, doc_len, collection_freq, collection_len)
        jm_ranked = rank_results(jm_scores, top_k)

        print("\n--- Jelinek-Mercer RESULTS ---")
        for doc, score in jm_ranked:
            print(f"{doc}: {score:.12f}")

        print("\n----------------------------------------------\n")

#Execution
folder_path = "Trump_Speeches"
query_file_path = "queries1.txt"

TOP_K = 10    

retrieve_documents(folder_path, query_file_path, top_k=TOP_K)
