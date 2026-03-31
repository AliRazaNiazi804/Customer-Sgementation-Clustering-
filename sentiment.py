#!/usr/bin/env python3
import cgi
import joblib

print("Content-Type: text/html\n")

form = cgi.FieldStorage()
text = form.getvalue("news")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

X = vectorizer.transform([text])
prediction = model.predict(X)[0]

print(f"<h2>Sentiment: {prediction}</h2>")
