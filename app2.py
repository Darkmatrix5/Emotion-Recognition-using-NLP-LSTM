from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st
import re
from nltk.stem import PorterStemmer
import pickle
import nltk

nltk.download('stopwords')
stopwords=set(nltk.corpus.stopwords.words('english'))

model = load_model("model1.h5", compile=False)

with open("lb1.pkl","rb") as f:
    lb=pickle.load(f)

with open("vocab_info.pkl","rb") as f:
    vocab_info=pickle.load(f)

with open("tokenizer.pkl","rb") as f:
    tokenizer=pickle.load(f)

vocab_size=vocab_info["vocab_size"]
max_len=vocab_info["max_len"]


def sentence_cleaning(sentence):
    stemmer=PorterStemmer()
    corpus = []

    text=re.sub("[^a-zA-Z]"," ",sentence)
    text=text.lower()
    text=text.split()
    text=[stemmer.stem(word) for word in text if word not in stopwords]
    text=" ".join(text)
    corpus.append(text)
    seqs = tokenizer.texts_to_sequences(corpus)
    padded = pad_sequences(seqs, maxlen=max_len, padding='pre')

    return padded

def predict_emotion_lstm(text):
    x=sentence_cleaning(text)
    preds=model.predict(x)

    idx=np.argmax(preds)
    emotion=lb.inverse_transform([idx])[0]
    confidence=np.max(preds)

    return emotion,confidence


#==================================creating app====================================
# App
st.markdown("## 🎭 Six Human Emotions Detection App")
st.caption("Detect emotion from text using LSTM")

st.markdown("***Supported Emotions:*** 😊 Joy | 😨 Fear | 😡 Anger | ❤️ Love | 😢 Sadness | 😲 Surprise")

st.divider()

user_input = st.text_area(
    "✍️ Enter your text here:",
    height=120,
    placeholder="Example: I feel really happy today!"
)

if st.button("🔍 Predict Emotion"):
    if user_input.strip()=="":
        st.warning("Please enter some text.")
    else:
        predicted_emotion,label=predict_emotion_lstm(user_input)
        st.success(f"**Predicted Emotion:** {predicted_emotion}")
        st.info(f"**Confidence:** {label:.4f}")



