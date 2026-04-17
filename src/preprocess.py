import string

def preprocess(text):
    """
    Funzione di preprocessing del testo.

    Operazioni eseguite:
    - conversione in minuscolo
    - rimozione della punteggiatura
    - rimozione spazi extra
    """

    # conversione in minuscolo
    text = text.lower()

    # rimozione della punteggiatura
    text = text.translate(str.maketrans('', '', string.punctuation))

    # rimozione spazi iniziali/finali
    text = text.strip()

    return text
