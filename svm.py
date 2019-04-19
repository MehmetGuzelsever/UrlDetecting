import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

urls_data = pd.read_csv("urldata.csv")

y = urls_data["label"]

url_list = urls_data["url"]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(url_list)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = RandomForestClassifier(min_samples_leaf=20)
clf.fit(X, y)
print("Random Forest", clf.score(X,y))

