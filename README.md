# Hotel Review NLP Classifier

## Descrizione

Questo progetto implementa un sistema di classificazione automatica delle recensioni alberghiere utilizzando tecniche di Natural Language Processing (NLP) e Machine Learning.

L'obiettivo è duplice:
- classificare ogni recensione in base al reparto di riferimento (Housekeeping, Reception, Food & Beverage)
- determinare il sentiment espresso (positivo o negativo)

Il sistema è progettato come prototipo didattico, con particolare attenzione alla semplicità, interpretabilità e riproducibilità.

---

## Architettura del sistema

Il progetto è strutturato come una pipeline di elaborazione composta dalle seguenti fasi:

1. **Generazione dataset sintetico**  
   Creazione di recensioni simulate per addestramento e test.

2. **Preprocessing del testo**  
   - conversione in minuscolo  
   - rimozione della punteggiatura  

3. **Vettorizzazione (TF-IDF)**  
   Trasformazione del testo in vettori numerici utilizzabili dai modelli.

4. **Addestramento modelli**
   - classificazione multiclasse (reparto)
   - classificazione binaria (sentiment)

5. **Valutazione**
   - Accuracy  
   - F1-score (macro)

6. **Interfaccia utente**
   Applicazione interattiva sviluppata con Streamlit.

---

## Scelte progettuali

- **TF-IDF**: scelto per la sua semplicità e per l'efficacia nei problemi di classificazione testuale lineare  
- **Regressione logistica**: modello interpretabile e adatto a dataset di dimensioni ridotte  
- **Dataset sintetico**: utilizzato per mantenere controllo sulle etichette e sulla distribuzione delle classi  

---

## Struttura del progetto
