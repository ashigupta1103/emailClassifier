import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

st.title("Email Spam Classifier")
st.markdown("This model predicts whether an email is **Spam** or **Safe** using Machine Learning.")

st.sidebar.header("About the Model")
st.sidebar.write("Model: Multinomial Naive Bayes")
st.sidebar.write("Vectorizer: TF-IDF (3000 features)")
st.sidebar.write("Trained on Enron Spam Dataset")

user_input = st.text_area("Paste your email message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter an email to classify.")
    else:
        clean_text = preprocess_text(user_input)
        input_vec = vectorizer.transform([clean_text])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("SPAM!")
        else:
            st.success("SAFE!")

st.caption("Developed by ~ashi<3")
