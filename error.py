svr_lin = SVR(kernel='linear', C=100, gamma='auto')

y = urls_data["label"]

url_list = urls_data["url"]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(url_list)

y_lin = svr_lin.fit(X, y).predict(X)

print("bitti")