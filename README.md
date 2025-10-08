# Email Spam Classifier

This is a **Machine Learning web app** that classifies emails as **Spam** or **Not Spam** using **Natural Language Processing (NLP)** and **Naive Bayes Classifier**.
It is built entirely in **Python** and deployed using **Streamlit Cloud** — simple, fast, and beginner-friendly.

---

## Project Overview

This project reads an email dataset, cleans and preprocesses the text using NLP techniques, then trains a **TF-IDF + Multinomial Naive Bayes** model to classify spam emails.

Unlike typical ML apps, this one **does not use pickle files** — the model trains directly within the app.
This makes it easy to understand, run, and deploy for learners.

---

## How It Works

1. **Data Cleaning:**
   Converts text to lowercase, removes punctuation, stopwords, and applies stemming.

2. **Feature Extraction:**
   Converts text into numerical form using **TF-IDF Vectorization** (up to 3000 features).

3. **Model Training:**
   Uses **Multinomial Naive Bayes** — a simple yet powerful algorithm for text classification.

4. **Prediction:**
   The user inputs an email message, and the app predicts if it’s **Spam** or **Not Spam**.

---

## Tech Stack

* **Frontend & Deployment:** Streamlit
* **Backend (ML):** scikit-learn
* **Data Handling:** pandas
* **NLP Processing:** nltk (stopwords, stemming)
* **Language:** Python 3.x

---

## Installation & Setup

### 1. Clone or download this repository

```bash
git clone https://github.com/ashigupta1103/emailClassifier.git
cd emailClassifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app locally

```bash
streamlit run app.py
```

Your app will launch at [http://localhost:8501](http://localhost:8501)

---

## Deployment (Streamlit Cloud)

1. Push this project to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Choose `app.py` as the entry point
5. Click **Deploy** 🚀

Your app will be live in a few minutes!

---

## Model Details

| Step               | Technique                  | Description                 |
| ------------------ | -------------------------- | --------------------------- |
| Text Cleaning      | Regex, Stopwords, Stemming | Removes noise from emails   |
| Feature Extraction | TF-IDF                     | Converts text → numeric     |
| Model              | Multinomial Naive Bayes    | Ideal for word frequencies  |
| Accuracy           | ~95% (varies by dataset)   | Reliable for spam detection |

---

## 💡 Example Predictions

| Input Email                               | Prediction |
| ----------------------------------------- | ---------- |
| “Congratulations! You won a free iPhone!” | Spam |
| “Your meeting is rescheduled to 3 PM.”    | Safe |

---

## Credits

Developed by ashi<3 using:

* Python 
* scikit-learn 
* Streamlit 
* NLTK 

---

## Future Improvements

* Add dataset upload feature
* Include performance metrics visualization
* Use pre-trained embeddings (Word2Vec / BERT)
* Create dark mode UI in Streamlit
