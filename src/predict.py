import pickle
import os
from preprocess import preprocess

# Percorso assoluto della root del progetto
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Percorso cartella modelli
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Caricamento modelli (robusto)
vectorizer = pickle.load(open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb"))
dep_model = pickle.load(open(os.path.join(MODEL_DIR, "dep_model.pkl"), "rb"))
sent_model = pickle.load(open(os.path.join(MODEL_DIR, "sent_model.pkl"), "rb"))


def predict_review(title, body):
    # Combina titolo + corpo (come nel training)
    text = (title or "") + " " + (body or "")

    # Preprocessing
    clean_text = preprocess(text)

    # Vettorizzazione
    text_vec = vectorizer.transform([clean_text])

    # Predizioni
    dep_pred = dep_model.predict(text_vec)[0]
    sent_pred = sent_model.predict(text_vec)[0]

    return dep_pred, sent_pred


# Test locale
if __name__ == "__main__":
    title = "Camera sporca"
    body = "Il bagno era in pessime condizioni e il letto scomodo"

    dep, sent = predict_review(title, body)

    print("Titolo:", title)
    print("Testo:", body)
    print("Reparto:", dep)
    print("Sentiment:", sent)
