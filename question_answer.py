import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('data.csv', encoding='utf-8')
# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the dataset and transform it
tfidf = vectorizer.fit_transform(df['question'])

# Define a function to calculate the similarity between the question and the documents
def calculate_similarity(question):
    question_vector = vectorizer.transform([question])
    similarity = cosine_similarity(question_vector, tfidf).flatten()
    return similarity

# Define a function to answer the question
def answer_question(question):
    similarity = calculate_similarity(question)
    indices = np.argsort(similarity)[::-1]
    for idx in indices:
        if similarity[idx] > 0.5:
            return df.iloc[idx]['answer']
    return "I don't know!"

# Test the function
question = "where is ball?"
print(answer_question(question))  