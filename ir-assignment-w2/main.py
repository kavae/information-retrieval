# Done by Avaya Khatri

import os
import re

#  1: Preprocess and Tokenize
def preprocess(text):
    """Clean and tokenize text into lowercase words."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    tokens = text.split()
    return tokens



# Step 2: Build Inverted Index

def build_index(folder):
    inverted_index = {}
    doc_ids = {}
    doc_count = 0

    print("\n Building inverted index...")
    for filename in os.listdir(folder):
        if filename.endswith('.txt') and not filename.startswith('query'):
            doc_count += 1
            doc_id = f"D{doc_count}"
            doc_ids[doc_id] = filename

            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                tokens = preprocess(content)
                print(f"\n {filename} → {len(tokens)} tokens")
                print(f"Cleaned Tokens: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")

            for word in set(tokens):
                inverted_index.setdefault(word, set()).add(doc_id)

    print("\n✔ Inverted index built successfully!\n")
    return inverted_index, doc_ids


# Step 3: Boolean Query Processing
def boolean_retrieval(query, inverted_index):
    tokens = query.lower().split()
    result = set()

    def get_docs(term):
        return inverted_index.get(term, set())

    all_docs = set().union(*inverted_index.values())
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "and":
            i += 1
            result = result & get_docs(tokens[i])
        elif token == "or":
            i += 1
            result = result | get_docs(tokens[i])
        elif token == "not":
            i += 1
            result = all_docs - get_docs(tokens[i])
        else:
            if i == 0:
                result = get_docs(token)
        i += 1
    return result


# Step 4: Read Queries & Write Results
def process_queries(folder, inverted_index, doc_ids):
    query_file = os.path.join(folder, 'queries.txt')
    output_file = os.path.join(folder, 'query_results.txt')

    print("🔍 Processing queries...\n")
    with open(query_file, 'r', encoding='utf-8') as qf, open(output_file, 'w') as of:
        for query in qf:
            query = query.strip()
            if not query:
                continue
            results = boolean_retrieval(query, inverted_index)
            filenames = [doc_ids[doc_id] for doc_id in results]
            of.write(f"Query: {query}\nResults: {', '.join(filenames) if filenames else 'No match'}\n\n")
            print(f"→ {query} : {', '.join(filenames) if filenames else 'No match'}")

    print("\n✔ Query results saved to query_results.txt\n")


# Step 5: Main Execution
if __name__ == "__main__":
    folder = '.'  # Current working directory
    inverted_index, doc_ids = build_index(folder)
    process_queries(folder, inverted_index, doc_ids)
