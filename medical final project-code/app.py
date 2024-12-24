from flask import Flask, render_template, request, send_file
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import joblib
import io
import google.generativeai as genai
import re


def format_gemini_text(raw_text):

    # '**' remove
    raw_text = raw_text.replace("**", "")
    raw_text = re.sub(r"[*•]", "", raw_text)  # Removes any bullet points

    return raw_text


# configure Google Gemini API
genai.configure(api_key="Your_API")  # API

def get_gemini_suggestions(input_data, prediction_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            f"Patient Data: "
            f"Age: {input_data[0]}, "
            f"Sex: {input_data[1]} (1 = Male, 0 = Female), "
            f"Chest Pain Type: {input_data[2]}, "
            f"Blood Pressure: {input_data[3]} mmHg, "
            f"Cholesterol Level: {input_data[4]} mg/dL, "
            f"FBS over 120: {'Yes' if input_data[5] == 1 else 'No'}, "
            f"EKG Results: {input_data[6]}, "
            f"Max Heart Rate: {input_data[7]} bpm, "
            f"Exercise Angina: {'Yes' if input_data[8] == 1 else 'No'}, "
            f"ST Depression: {input_data[9]}, "
            f"Slope of ST: {input_data[10]}, "
            f"Number of Vessels: {input_data[11]}, "
            f"Thallium Test: {input_data[12]}. "
            f"Prediction: {prediction_text}. "
            "Please provide specific health recommendations based on this information. just text answers don't include titles or etc."
        )

        response = model.generate_content(prompt)
        return response.text if response else "No suggestions available."

    except Exception as e:
        print(f"Error with Google Gemini API: {e}")
        return "Unable to fetch suggestions at this time."


app = Flask(__name__)

# Connection Model and database
model = joblib.load("heart_disease_model.pkl")
DATABASE = "user_data.db"

# setting up connection database
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Age INTEGER, Sex INTEGER, `Chest pain type` INTEGER,
        BP INTEGER, Cholesterol INTEGER, `FBS over 120` INTEGER,
        `EKG results` INTEGER, `Max HR` INTEGER, `Exercise angina` INTEGER,
        `ST depression` REAL, `Slope of ST` INTEGER, `Number of vessels fluro` INTEGER,
        Thallium INTEGER, Prediction TEXT, Confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

# mainpage
@app.route('/')
def home():
    return render_template('index.html')

# make predict and save the result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form[key]) for key in request.form.keys()]
        input_data = np.array(features).reshape(1, -1)

        # make predict
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0]
        prediction_text = "High Risk" if prediction == 1 else "Low Risk"
        confidence_score = confidence[1] if prediction == 1 else confidence[0]
        risk_rate = 1 - confidence

        # get the suggestion
        raw_suggestions = get_gemini_suggestions(features, prediction_text)
        formatted_suggestions = format_gemini_text(raw_suggestions)
        # save on database
        save_to_db(features, prediction_text, confidence_score)

        return render_template(
            'result.html',
            prediction=prediction_text,
            confidence=confidence_score,
            features=features,
            suggestions=formatted_suggestions,
            risk_rate=(1 - confidence))

def save_to_db(features, prediction, confidence):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO predictions (
        Age, Sex, `Chest pain type`, BP, Cholesterol, `FBS over 120`,
        `EKG results`, `Max HR`, `Exercise angina`, `ST depression`,
        `Slope of ST`, `Number of vessels fluro`, Thallium, Prediction, Confidence
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', features + [prediction, confidence])
    conn.commit()
    conn.close()


# creating pdf report
@app.route('/download_pdf')
def download_pdf():
    try:
        # PDF dosyasını oluştur
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer)
        c.drawString(100, 800, "Heart Disease Prediction Report")

        # Gelen parametreleri oku ve PDF'ye yaz
        prediction = request.args.get('prediction', 'N/A')
        risk_rate = request.args.get('risk_rate', 'N/A')
        c.drawString(100, 780, f"Prediction: {prediction}")
        c.drawString(100, 760, f"Risk Rate: {risk_rate}")

        y = 740
        for key, value in request.args.items():
            if key not in ['prediction', 'risk_rate']:
                c.drawString(100, y, f"{key}: {value}")
                y -= 20

        # save pdf
        c.save()
        buffer.seek(0)

        # offer pdf as download
        return send_file(
            buffer,
            as_attachment=True,
            download_name="prediction_report.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        return f"Error generating PDF: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
