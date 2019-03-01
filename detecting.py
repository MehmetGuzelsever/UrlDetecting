import pandas as pd
import numpy as np
import random


#Makine Öğrenmesi için gerekli olan Kütüphaneler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Url dosyasının yüklenmesi
urls_data = pd.read_csv("urldata.csv")


#url datanın frame'in öğrenilmesi
type(urls_data)


urls_data.head()



#Url'rin "/" "." gibi sabit gereksiz uzantıların kaldırılması.
def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')	# make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens


#Etiketler
y = urls_data["label"]

#Özellikler
url_list = urls_data["url"]

# Using Default Tokenizer
vectorizer = TfidfVectorizer()

# Using Custom Tokenizer
#vectorizer = TfidfVectorizer(tokenizer=makeTokens)


# Vektörleri X değişkeninde sakladık.
X = vectorizer.fit_transform(url_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Mimarimiz
#using logistic regression
logit = LogisticRegression()	
logit.fit(X_train, y_train)

# Modelimizin Doğruluk oranı
#print("Accuracy ",logit.score(X_test, y_test))


#Model Tahmini
X_predict = ["google.com/search=jcharistech",
"google.com/search=faizanahmad",
"pakistanifacebookforever.com/getpassword.php/", 
"www.radsport-voggel.de/wp-admin/includes/log.exe", 
"ahrenhei.without-transfer.ru/nethost.exe ",
"www.itidea.it/centroesteticosothys/img/_notes/gum.exe",
"github.com",
"https://www.youtube.com/watch?v=9gOCzxYxR2g",
"www.npmjs.com/package/kafka"]


X_predict = vectorizer.transform(X_predict)
New_predict = logit.predict(X_predict)

#Yeni Tahminiz
print(New_predict)





