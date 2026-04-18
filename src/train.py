import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess


def load_data(path):
    """Carica il dataset CSV"""
    return pd.read_csv(path)


def preprocess_data(df):
    """Applica preprocessing al testo"""
    df["clean_text"] = df["text"].apply(preprocess)
    return df


def split_data(df):
    """Divide il dataset in train e test"""
    return train_test_split(df, test_size=0.2, random_state=42)


def vectorize_text(X_train, X_test):
    """Crea e applica TF-IDF"""
    vectorizer = TfidfVectorizer()
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return vectorizer, X_train_vec, X_test_vec


def train_models(X_train_vec, y_dep_train, y_sent_train):
    """Addestra i modelli"""
    dep_model = LogisticRegression(max_iter=1000)
    sent_model = LogisticRegression(max_iter=1000)

    dep_model.fit(X_train_vec, y_dep_train)
    sent_model.fit(X_train_vec, y_sent_train)

    return dep_model, sent_model


def evaluate_model(model, X_test_vec, y_test, label):
    """Valuta il modello"""
    predictions = model.predict(X_test_vec)

    print(f"\n=== Valutazione {label} ===")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


def save_models(vectorizer, dep_model, sent_model):
    """Salva i modelli su disco"""
    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    pickle.dump(dep_model, open("models/dep_model.pkl", "wb"))
    pickle.dump(sent_model, open("models/sent_model.pkl", "wb"))


def main():
    # 1. Caricamento dati
    df = load_data("data/reviews_dataset.csv")

    # 2. Preprocessing
    df = preprocess_data(df)

    # 3. Split
    train_df, test_df = split_data(df)

    # 4. Vettorizzazione
    vectorizer, X_train_vec, X_test_vec = vectorize_text(
        train_df["clean_text"], test_df["clean_text"]
    )

    # 5. Training
    dep_model, sent_model = train_models(
        X_train_vec,
        train_df["department"],
        train_df["sentiment"]
    )

    # 6. Valutazione
    evaluate_model(dep_model, X_test_vec, test_df["department"], "Reparto")
    evaluate_model(sent_model, X_test_vec, test_df["sentiment"], "Sentiment")

    # 7. Salvataggio
    save_models(vectorizer, dep_model, sent_model)


if __name__ == "__main__":
    main()
