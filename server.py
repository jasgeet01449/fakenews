from flask import Flask, flash, render_template, redirect, jsonify, url_for, make_response, request, session
from keras.models import load_model
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')
nltk.download('punkt')
import base64, os


app = Flask(__name__)
app.secret_key = "mys3cr3tk3y"
app.config['USE_PERMANENT_SESSION']=True

model = load_model('model.h5')

ps = PorterStemmer()
stop_words = stopwords.words("english")


def get_response_msg(sentence):

    data = []
    review = re.sub('[^a-zA-Z]', ' ', sentence)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    data.append(review)

    onehot_t = [one_hot(words, 5000) for words in data]
    docs = pad_sequences(onehot_t, padding = "pre", maxlen=20)

    predict_list = np.array(docs)

    THRESHOLD = 0.5

    result = ""

    if model.predict(predict_list)[0][0] > THRESHOLD:
        result = "REAL"
    else:
        result = "FAKE"

    print(result)
    return result


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/get_response', methods = ['POST', 'GET'])
def get_response():
    if request.method == "POST":
        sentence = request.form['text']
        return get_response_msg(sentence)

        

if __name__ == '__main__':
    app.run(debug = True, port = 8181 ,host="0.0.0.0")



    
