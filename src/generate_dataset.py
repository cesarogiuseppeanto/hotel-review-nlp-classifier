import random
import pandas as pd

# Frasi per ogni reparto
departments = {
    "Housekeeping": {
        "pos": ["camera pulita", "bagno impeccabile", "lenzuola fresche"],
        "neg": ["camera sporca", "odore sgradevole", "polvere ovunque"]
    },
    "Reception": {
        "pos": ["staff gentile", "check-in veloce", "personale disponibile"],
        "neg": ["personale scortese", "check-in lento", "servizio pessimo"]
    },
    "F&B": {
        "pos": ["colazione ottima", "buffet abbondante", "cibo delizioso"],
        "neg": ["colazione scarsa", "cibo freddo", "servizio lento"]
    }
}

def generate_review(department, sentiment):
    phrase = random.choice(departments[department][sentiment])

    # aggiunge un po' di varietà
    if random.random() < 0.2:
        extra_dep = random.choice(list(departments.keys()))
        phrase += " e " + random.choice(departments[extra_dep][sentiment])

    title = phrase.split()[0].capitalize() + " esperienza"
    body = f"La mia esperienza è stata: {phrase}."

    return title, body


def create_dataset(n=300):
    data = []

    for i in range(n):
        dep = random.choice(list(departments.keys()))
        sent = random.choice(["pos", "neg"])

        title, body = generate_review(dep, sent)

        data.append({
            "id": i,
            "title": title,
            "body": body,
            "department": dep,
            "sentiment": sent
        })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = create_dataset(300)
    df.to_csv("data/reviews_dataset.csv", index=False)
    print("Dataset creato in data/reviews_dataset.csv")
