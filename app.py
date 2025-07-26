from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model, scaler, and encoders
model = joblib.load('Churn_Prediction_Model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 1. Get user inputs
        tenure = float(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        total_charges = float(request.form["total_charges"])
        gender = request.form["gender"]
        senior_citizen = int(request.form["senior_citizen"])
        dependents = request.form["dependents"]
        phone_service = request.form["phone_service"]
        multiple_lines = request.form["multiple_lines"]

        # 2. Encode categorical inputs
        gender = label_encoders["gender"].transform([gender])[0]
        dependents = label_encoders["Dependents"].transform([dependents])[0]
        phone_service = label_encoders["PhoneService"].transform([phone_service])[0]
        multiple_lines = label_encoders["MultipleLines"].transform([multiple_lines])[0]

        # 3. Prepare input for model
        input_data = np.array([[tenure, monthly_charges, total_charges,
                                gender, senior_citizen, dependents,
                                phone_service, multiple_lines]])
        input_scaled = scaler.transform(input_data)

        # 4. Predict
        prediction = model.predict(input_scaled)[0]
        result = "No Churn 😄" if prediction == 0 else "Churn 😢"

        return render_template("result.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
