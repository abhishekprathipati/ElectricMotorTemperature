from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and feature order
model, features = joblib.load("model/motor_temp_model.pkl")

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = []

        for feature in features:
            value = float(request.form[feature])
            input_data.append(value)

        final_input = np.array([input_data])
        prediction = model.predict(final_input)[0]

        return render_template(
            "home.html",
            prediction_text=f"Predicted Electric MotorTemperature: {prediction:.2f} °C"
        )

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run()

