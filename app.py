from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the dataset
try:
    df = pd.read_csv('data.csv', encoding='utf-8')
except FileNotFoundError:
    print("The dataset file was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print("The dataset file is empty.")
    exit(1)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the dataset and transform it
try:
    tfidf = vectorizer.fit_transform(df['question'])
except KeyError:
    print("The 'question' column was not found in the dataset.")
    exit(1)

# Define a function to calculate the similarity between the question and the documents
def calculate_similarity(question):
    try:
        question_vector = vectorizer.transform([question])
        similarity = cosine_similarity(question_vector, tfidf).flatten()
        return similarity
    except ValueError:
        return np.array([])

# Define a function to answer the question
def answer_question(question):
    similarity = calculate_similarity(question)
    if len(similarity) == 0:
        return "I don't know!"
    indices = np.argsort(similarity)[::-1]
    for idx in indices:
        if similarity[idx] > 0.5:
            return df.iloc[idx]['answer']
    return "I don't know!"

@app.route('/answer', methods=['OPTIONS', 'POST'])
def answer():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'})  # Return a response for OPTIONS requests
    elif request.method == 'POST':
        try:
            question = request.json['question']
            if not isinstance(question, str) or len(question.strip()) == 0:
                return jsonify({'error': 'Invalid question'}), 400
            answer = answer_question(question)
            return jsonify({'answer': answer})
        except KeyError:
            return jsonify({'error': 'Missing question'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)