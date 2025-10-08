#!/usr/bin/env python
# coding: utf-8

# # Email Classification using Machine Learning
# 
# ### Objective
# Predict whether an email is spam or not using a dataset.
# 
# ### Approach
# 1. Preprocess text data by:
#    - Converting to lowercase
#    - Removing special characters and stop words
#    - Stemming words
# 2. Convert text into numerical features using TF-IDF vectorization.
# 3. Train a simple classifier (Naive Bayes) to classify emails.
# 4. Evaluate the model’s performance on a test set.

# In[6]:


import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[7]:


df = pd.read_csv("enronSpamSubset.csv")
df.head()


# In[8]:


df.info()
df['Label'].value_counts()


# In[11]:


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

df['clean_text'] = df['Body'].apply(preprocess_text)


# In[12]:


X = df['clean_text']
y = df['Label']


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# In[15]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




