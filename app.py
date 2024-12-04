from flask import Flask, request, jsonify
import numpy as np
import joblib
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# Pipeline para Análise de Sentimentos
nlp = pipeline("text-classification", model="distilbert-base-uncased")

# Modelo de Machine Learning
MODEL_PATH = "modelo_treinado.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = RandomForestClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([
        [
            data.get("idade", 0),
            1 if data.get("genero") == "M" else 0,
            data.get("historico_criminal", 0),
            data.get("impulsividade", 0),
            data.get("estabilidade_familiar", 0),
            data.get("condicao_economica", 0),
            data.get("normas_culturais", 0),
        ]
    ])

    risco_predito = model.predict(features)[0]
    probabilidade = model.predict_proba(features)[0]

    relatorio = data.get("relatorio", "")
    sentiment_analysis = nlp(relatorio)

    return jsonify({
        "risco_reincidencia": "ALTO" if risco_predito == 1 else "BAIXO",
        "probabilidade": probabilidade.tolist(),
        "sentimento": sentiment_analysis,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
