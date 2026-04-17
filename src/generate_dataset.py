import random
import pandas as pd

# Frasi più realistiche (stile recensioni vere)
departments = {
    "Housekeeping": {
        "pos": [
            "la camera era molto pulita",
            "bagno impeccabile e ordinato",
            "lenzuola fresche e profumate"
        ],
        "neg": [
            "la camera era sporca",
            "odore sgradevole in bagno",
            "polvere ovunque nella stanza"
        ]
    },
    "Reception": {
        "pos": [
            "staff molto gentile",
            "check-in veloce e senza problemi",
            "personale disponibile e professionale"
        ],
        "neg": [
            "personale scortese",
            "check-in molto lento",
            "poca disponibilità alla reception"
        ]
    },
    "F&B": {
        "pos": [
            "colazione ottima e varia",
            "buffet abbondante",
            "cibo davvero buono"
        ],
        "neg": [
            "colazione scarsa",
            "cibo freddo",
            "servizio ristorante lento"
        ]
    }
}

# Frasi introduttive per rendere il testo più naturale
intro_phrases = [
    "Durante il soggiorno",
    "Nel complesso",
    "La mia esperienza",
    "Devo dire che",
    "Purtroppo"
]


def generate_review(department, sentiment):
    phrase = random.choice(departments[department][sentiment])
    intro = random.choice(intro_phrases)

    # Introduzione di ambiguità (recensioni che parlano di più reparti)
    if random.random() < 0.2:
        extra_dep = random.choice(list(departments.keys()))
        extra_phrase = random.choice(departments[extra_dep][sentiment])
        phrase = phrase + " e " + extra_phrase

    title = phrase.split()[0].capitalize() + " esperienza"
    body = f"{intro}, {phrase}."

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
    print("Dataset creato con successo!")
