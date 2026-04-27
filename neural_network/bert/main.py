from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

reviews = [
    "This movie was amazing and enjoyable.",
    "The product was terrible and waste of money.",
    "Customer service was excellent and helpful.",
    "The food was cold and tasteless."
]

for review in reviews:
    result = classifier(review)
    
    print("Review:", review)
    print("Prediction:", result[0]['label'])
    print("Confidence:", round(result[0]['score'], 4))
    print("-" * 50)