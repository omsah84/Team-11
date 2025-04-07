
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import nltk
import pickle

# ================== NLTK Configuration ==================
nltk.data.path = ["C:\\nltk_data"]  # Set custom path without spaces

# Download required NLTK resources to custom path
nltk.download('punkt', download_dir="C:\\nltk_data")
nltk.download('stopwords', download_dir="C:\\nltk_data")
nltk.download('wordnet', download_dir="C:\\nltk_data")
nltk.download('punkt_tab', download_dir="C:\\nltk_data")  # Add missing resource

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize Flask app
app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load trained model and vectorizer
loaded_model = pickle.load(open("model.pkl", 'rb'))
vector = pickle.load(open("vector.pkl", 'rb'))

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

# Prediction function
def fake_news_det(news):
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    tokens = word_tokenize(review)
    
    corpus = [lemmatizer.lemmatize(word) for word in tokens if word not in stpwrds]
    
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    
    return prediction

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)
        return render_template("prediction.html", 
                             prediction_text="📰 Real News" if pred[0] == 0 else "📰 Fake News")
    return render_template('prediction.html', prediction_text="Please submit a news article")

if __name__ == '__main__':
    app.run(debug=True)