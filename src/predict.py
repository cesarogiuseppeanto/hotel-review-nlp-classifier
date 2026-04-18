import pickle
from preprocess import preprocess

# Caricamento modelli
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
dep_model = pickle.load(open("models/dep_model.pkl", "rb"))
sent_model = pickle.load(open("models/sent_model.pkl", "rb"))

def predict_review(text):
    # Preprocessing
    text_clean = preprocess(text)

    # Vettorizzazione
    text_vec = vectorizer.transform([text_clean])

    # Predizioni
    dep_pred = dep_model.predict(text_vec)[0]
    sent_pred = sent_model.predict(text_vec)[0]

    return dep_pred, sent_pred


# test opzionale (NON necessario)
if __name__ == "__main__":
    text = "La camera era pulita e il personale molto gentile"
    dep, sent = predict_review(text)
    print("Testo:", text)
    print("Dipartimento:", dep)
    print("Sentiment:", sent)
