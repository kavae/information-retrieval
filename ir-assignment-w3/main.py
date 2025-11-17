# Avaya Khatry

import requests, csv
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords', quiet=True)

# Load URLs
urls = [u.strip() for u in open("urls.txt").read().splitlines()]

docs = []
names = []

# Download & extract text (HTML only for simplicity)
for i, url in enumerate(urls):
    print("Downloading:", url)
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    text = "\n".join([p.get_text() for p in soup.find_all("p")])
    docs.append(text)
    names.append(f"doc{i+1}")

# TF-IDF + cosine similarity
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(docs)
sim = cosine_similarity(X)

# To Print similarity
print("\nSimilarity Matrix:\n", sim)

# To Save CSV
with open("similarity.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([""] + names)
    for i in range(len(names)):
        writer.writerow([names[i]] + list(sim[i]))

# For Heatmap 
plt.imshow(sim, cmap="viridis")
plt.colorbar()
plt.xticks(range(len(names)), names, rotation=90)
plt.yticks(range(len(names)), names)
plt.title("Document Similarity Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")
print("\nSaved similarity.csv and heatmap.png")
