from unittest.util import _MAX_LENGTH
from flask import Flask, redirect, render_template, request, url_for
import cv2
import numpy as np
from keras.applications import ResNet50
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import requests
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from tensorflow import keras

# Load vocabulary
vocab = np.load('vocab.npy', allow_pickle=True).item()
inv_vocab = {v: k for k, v in vocab.items()}
print("+" * 50)
print("vocabulary loaded")

# Load ResNet50 model
resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
print("+" * 50)
print("resnet loaded")

# Load model
max_len = 40
model = keras.models.load_model('model.h5')
print("=" * 50)
print("MODEL LOADED")

# Define Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# Define routes
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and request.form['username'] == 'admin' and request.form['password'] == 'password':
        print("Login successful. Redirecting to index page...")
        return redirect(url_for('index'))
    print("Login page requested.")
    return render_template('login.html')

@app.route('/index')
def index():
    print("Index page requested.")
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, resnet, vocab, inv_vocab

    img = request.files['file1']
    img.save('static/file.jpg')
    print("=" * 50)
    print("IMAGE SAVED")

    image = cv2.imread('static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.reshape(image, (1, 224, 224, 3))
    incept = resnet.predict(image).reshape(1, 2048)
    print("=" * 50)
    print("Predict Features")
    text_in = ['startofseq']
    final = ''
    print("=" * 50)
    print("GETING Captions")
    count = 0
    while count < 20: 
        count += 1
        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)
    
    caption= final
    url = "https://api.edenai.run/v2/text/generation"

    payload = {
    "response_as_dict": True,
    "attributes_as_list": False,
    "show_original_response": False,
    "temperature": 0,
    "max_tokens": 1000,
    "providers": "openai",
    "text": "Generate trending 20 without numbers Hashatgs  for this caption:"+caption}
    
    headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiODlhZTdjYjctNjFiYi00MzZiLTliNzgtYTIyZjI3ZGYyNjQwIiwidHlwZSI6ImFwaV90b2tlbiJ9.A3DAnEg-gv2YvCZhrM_GEnZrXbKY8oXC7ma8Rmy6RNk"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        if 'openai' in response_json:
            if 'generated_text' in response_json['openai']:
                generated_text = response_json['openai']['generated_text'].strip()
                print(generated_text)
            else:
                print("generated_text not found in response.")
        else:
            print("openai not found in response.")
    else:
        print("Error:", response.status_code, response.text)

    return render_template('predict.html', final=generated_text,text=final)

if __name__ == "__main__":
    app.run(debug=True)