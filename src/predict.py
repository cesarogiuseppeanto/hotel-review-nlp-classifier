import pickle
from preprocess import preprocess

# Caricamento modelli
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
dep_model = pickle.load(open("models/dep_model.pkl", "rb"))
sent_model = pickle.load(open("models/sent_model.pkl", "rb"))

# Test con una frase
text = "La camera era pulita e il personale molto gentile"

# Preprocessing
text_clean = preprocess(text)

# Vettorizzazione
text_vec = vectorizer.transform([text_clean])

# Predizioni
dep_pred = dep_model.predict(text_vec)[0]
sent_pred = sent_model.predict(text_vec)[0]

print("Testo:", text)
print("Dipartimento:", dep_pred)
print("Sentiment:", sent_pred)
