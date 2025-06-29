import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Clean text function to normalize input
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Training data (intents)
training_data = [
    ("hello", "greet"),
    ("hi", "greet"),
    ("hey there", "greet"),
    ("good morning", "greet"),

    ("I want to book a flight", "book_flight"),
    ("can you find me a flight", "book_flight"),
    ("get me a ticket to Paris", "book_flight"),
    ("I need to travel to New York", "book_flight"),

    ("bye", "goodbye"),
    ("goodbye", "goodbye"),
    ("see you later", "goodbye"),
    ("talk to you soon", "goodbye")
]

# Split data
texts, labels = zip(*training_data)
texts = [clean_text(text) for text in texts]

# Model pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(texts, labels)

# Interactive input loop
print("Intent recognizer is ready! Type a message (or 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    cleaned = clean_text(user_input)
    intent = pipeline.predict([cleaned])[0]
    print(f"Predicted intent: {intent}")
