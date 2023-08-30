from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification, RobertaTokenizer, TFRobertaForSequenceClassification
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# The exact same functions you had in Streamlit:
@app.before_first_request
def load_models_and_preprocessors():
    global models, tokenizer, vectorizer, bert_tokenizer, roberta_tokenizer, bert_model, roberta_model

    # LSTM model
    lstm_model = load_model('model/lstm_model.h5')

    # sklearn models
    lr = joblib.load('model/logistic_regression_model.joblib')
    svc = joblib.load('model/svc_model.joblib')
    rf = joblib.load('model/random_forest_model.joblib')
    nb = joblib.load('model/naive_bayes_model.joblib')

    # BERT models
    bert_tokenizer = BertTokenizer.from_pretrained('model/outputs')
    bert_model = TFBertForSequenceClassification.from_pretrained('model/outputs', from_pt=True)

    # RoBERTa models
    roberta_tokenizer = RobertaTokenizer.from_pretrained('model/outputs_roberta')
    roberta_model = TFRobertaForSequenceClassification.from_pretrained('model/outputs_roberta', from_pt=True)

    # Tokenizer and Vectorizer
    tokenizer = joblib.load('model/tokenizer.joblib')
    vectorizer = joblib.load('model/vectorizer.joblib')

    models = {
        'LSTM': lstm_model,
        'Logistic Regression': lr,
        'SVC': svc,
        'Random Forest': rf,
        'Naive Bayes': nb,
        'BERT': bert_model,
        'RoBERTa': roberta_model
    }

def predict(model_name, user_input):
    right_percentage = 0
    
    if model_name == 'LSTM':
        user_input_seq = tokenizer.texts_to_sequences([user_input])
        user_input_seq_padded = pad_sequences(user_input_seq, maxlen=100, padding='post', truncating='post')
        prediction = models[model_name].predict(user_input_seq_padded)[0][0]
        right_percentage = round(prediction * 100)

    elif model_name in ['Logistic Regression', 'SVC', 'Random Forest', 'Naive Bayes']:
        user_input_tfidf = vectorizer.transform([user_input])
        prediction = models[model_name].predict(user_input_tfidf)[0]
        right_percentage = round(prediction * 100)

    elif model_name == 'BERT':
        inputs = bert_tokenizer(user_input, return_tensors='tf', truncation=True, padding=True, max_length=128)
        outputs = bert_model(**inputs)
        probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
        prediction = np.argmax(probabilities)
        if prediction:
            right_percentage = round(probabilities[prediction] * 100)

    elif model_name == 'RoBERTa':
        inputs = roberta_tokenizer(user_input, return_tensors='tf', truncation=True, padding=True, max_length=128)
        outputs = roberta_model(**inputs)
        probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
        prediction = np.argmax(probabilities)
        if prediction:
            right_percentage = round(probabilities[prediction] * 100)

    # Calculate left percentage as complementary value
    left_percentage = 100 - right_percentage
    return {'left': left_percentage, 'right': right_percentage}

@app.route("/", methods=['GET', 'POST'])
def index():
    models_list = list(models.keys())
    prediction = None

    if request.method == 'POST':
        user_input = request.form.get("comment")
        prediction = {}
        for model_name in models_list:
            prediction[model_name] = predict(model_name, user_input)
    
    return render_template("index.html", models=models_list, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
