import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    print("Colonne dataset:", df.columns)

    # individua automaticamente la colonna testo
    if "text" in df.columns:
        text_col = "text"
    elif "review" in df.columns:
        text_col = "review"
    elif "review_text" in df.columns:
        text_col = "review_text"
    else:
        raise ValueError("❌ Nessuna colonna testo trovata nel dataset")

    df["clean_text"] = df[text_col].apply(preprocess)
    return df


def split_data(df):
    return train_test_split(df, test_size=0.2, random_state=42)


def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return vectorizer, X_train_vec, X_test_vec


def train_models(X_train_vec, y_dep_train, y_sent_train):
    dep_model = LogisticRegression(max_iter=1000)
    sent_model = LogisticRegression(max_iter=1000)

    dep_model.fit(X_train_vec, y_dep_train)
    sent_model.fit(X_train_vec, y_sent_train)

    return dep_model, sent_model


def evaluate_model(model, X_test_vec, y_test, label):
    predictions = model.predict(X_test_vec)

    print(f"\n=== Valutazione {label} ===")
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


def save_models(vectorizer, dep_model, sent_model):
    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    pickle.dump(dep_model, open("models/dep_model.pkl", "wb"))
    pickle.dump(sent_model, open("models/sent_model.pkl", "wb"))


def main():
    df = load_data("data/reviews_dataset.csv")

    df = preprocess_data(df)

    # individua automaticamente colonne target
    if "department" in df.columns:
        DEP_COL = "department"
    elif "dep" in df.columns:
        DEP_COL = "dep"
    else:
        raise ValueError("❌ Colonna reparto non trovata")

    if "sentiment" in df.columns:
        SENT_COL = "sentiment"
    elif "sent" in df.columns:
        SENT_COL = "sent"
    else:
        raise ValueError("❌ Colonna sentiment non trovata")

    train_df, test_df = split_data(df)

    vectorizer, X_train_vec, X_test_vec = vectorize_text(
        train_df["clean_text"], test_df["clean_text"]
    )

    dep_model, sent_model = train_models(
        X_train_vec,
        train_df[DEP_COL],
        train_df[SENT_COL]
    )

    evaluate_model(dep_model, X_test_vec, test_df[DEP_COL], "Reparto")
    evaluate_model(sent_model, X_test_vec, test_df[SENT_COL], "Sentiment")

    save_models(vectorizer, dep_model, sent_model)


if __name__ == "__main__":
    main()
