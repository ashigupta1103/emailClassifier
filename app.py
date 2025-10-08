import streamlit as st
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

csv_path = os.path.join(os.path.dirname(__file__), "enronSpamSubset.csv")
df = pd.read_csv(csv_path)

if 'Body' in df.columns and 'Label' in df.columns:
    df.rename(columns={'Body': 'text', 'Label': 'label'}, inplace=True)
    
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess_text)

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

st.title("Email Spam Classifier")
st.markdown("This model predicts whether an email is **Spam** or **Not Spam** using Machine Learning.")

st.sidebar.header("Model Information")
st.sidebar.write(f"**Accuracy:** {accuracy:.3f}")
st.sidebar.write("**Model:** Multinomial Naive Bayes")
st.sidebar.write("**Vectorizer:** TF-IDF (3000 features)")

user_input = st.text_area("Paste your email message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter an email to classify.")
    else:
        clean_text = preprocess_text(user_input)
        input_vec = vectorizer.transform([clean_text])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error(" SPAM!")
        else:
            st.success("SAFE!")

st.caption("Developed by ~ashi<3")
