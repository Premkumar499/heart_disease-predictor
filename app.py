import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open("models/model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
features = data["features"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        input_df = pd.DataFrame([{
            "age":      float(payload["age"]),
            "sex":      float(payload["sex"]),
            "cp":       float(payload["cp"]),
            "trestbps": float(payload["trestbps"]),
            "chol":     float(payload["chol"]),
            "fbs":      float(payload["fbs"]),
            "restecg":  float(payload["restecg"]),
            "thalach":  float(payload["thalach"]),
            "exang":    float(payload["exang"]),
            "oldpeak":  float(payload["oldpeak"]),
            "slope":    float(payload["slope"]),
            "ca":       float(payload["ca"]),
            "thal":     float(payload["thal"]),
        }], columns=features)

        scaled = scaler.transform(input_df)
        prediction = int(model.predict(scaled)[0])
        probability = model.predict_proba(scaled)[0].tolist()

        return jsonify({
            "prediction": prediction,
            "prob_no_disease": round(probability[0] * 100, 1),
            "prob_disease":    round(probability[1] * 100, 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
