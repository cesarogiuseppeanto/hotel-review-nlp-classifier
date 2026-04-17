import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from preprocess import preprocess


# 1. Caricamento dataset
df = pd.read_csv("data/reviews_dataset.csv")

# 2. Creazione campo testo unificato
df["text"] = (df["title"] + " " + df["body"]).apply(preprocess)

# 3. Split train/test
X_train, X_test, y_dep_train, y_dep_test = train_test_split(
    df["text"], df["department"], test_size=0.2, random_state=42
)

_, _, y_sent_train, y_sent_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)

# 4. Vettorizzazione TF-IDF
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Addestramento modelli
dep_model = LogisticRegression()
dep_model.fit(X_train_vec, y_dep_train)

sent_model = LogisticRegression()
sent_model.fit(X_train_vec, y_sent_train)

# 6. Valutazione

# Department
dep_pred = dep_model.predict(X_test_vec)
print("=== Department Classification ===")
print("Accuracy:", accuracy_score(y_dep_test, dep_pred))
print("F1-score:", f1_score(y_dep_test, dep_pred, average="macro"))

# Sentiment
sent_pred = sent_model.predict(X_test_vec)
print("\n=== Sentiment Classification ===")
print("Accuracy:", accuracy_score(y_sent_test, sent_pred))
print("F1-score:", f1_score(y_sent_test, sent_pred, average="macro"))

# 7. Salvataggio modelli
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
pickle.dump(dep_model, open("models/dep_model.pkl", "wb"))
pickle.dump(sent_model, open("models/sent_model.pkl", "wb"))

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModelli salvati nella cartella models/")
