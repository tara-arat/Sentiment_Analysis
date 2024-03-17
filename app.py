from flask import Flask, request, jsonify, render_template
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

app = Flask(__name__)

def preprocess_text(text):
    try:
        # Convert text to lowercase
        text = str(text).lower()
        
        # Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join the tokens back into a single string
        clean_text = ' '.join(tokens)
        
        return clean_text
    except Exception as e:
        print("Error preprocessing text:", e)
        return ''

def load_model():
    try:
        model = joblib.load("logistic_regression_model.pkl")
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, tfidf_vectorizer
    except Exception as e:
        print("Error loading model:", e)
        return None, None

@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.form.get('review')
        if not data:
            return jsonify({'error': 'Empty review text received'}), 400
        
        review_cleaned = preprocess_text(data)
        if not review_cleaned:
            return jsonify({'error': 'Text preprocessing failed'}), 500
        
        model, tfidf_vectorizer = load_model()
        if model is None or tfidf_vectorizer is None:
            return jsonify({'error': 'Model loading failed'}), 500
        
        review_tfidf = tfidf_vectorizer.transform([review_cleaned])
        prediction = model.predict(review_tfidf)
        
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        return render_template('result.html', sentiment=sentiment)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
