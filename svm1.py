import pandas as pd
import numpy as np
import random
import json





#Makine Öğrenmesi için gerekli olan Kütüphaneler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

urls_data = pd.read_csv("urldata.csv")

type(urls_data)

urls_data.head()

y = urls_data["label"]

url_list = urls_data["url"]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(url_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logit = LogisticRegression()	
logit.fit(X_train, y_train)

print("Accuracy ",logit.score(X_test, y_test))