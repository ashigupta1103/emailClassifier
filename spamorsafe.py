#!/usr/bin/env python
# coding: utf-8

# # Email Classification using Machine Learning
# 
# ### Objective
# Predict whether an email is spam or not using a dataset.
# 
# ### Approach
# 1. Preprocess text data by:
#     - Converting to lowercase
#     - Removing special characters and stop        words
#     - Stemming words
# 2. Convert text into numerical features using TF-IDF vectorization.
# 3. Train a Naive Bayes classifier to classify emails.
# 4. Evaluate the model’s performance on a test set.
# 5. Save the trained model and vectorizer using Pickle for deployment.

# In[2]:


import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle


# In[5]:


df = pd.read_csv("enronSpamSubset.csv")
df.head()


# In[6]:


df.info()
df['Label'].value_counts()


# In[14]:


nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)


# In[15]:


df['clean_text'] = df['Body'].apply(preprocess_text)
df.head()


# In[18]:


X = df['clean_text']
y = df['Label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[22]:


with open("spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)


# In[ ]:


with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and vectorizer saved successfully!")


# In[ ]:




