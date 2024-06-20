from flask import Flask, jsonify, request
import os
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.preprocessing import StandardScaler
import joblib
import skimage.io
from core.main import core_ocr
from ultralytics import YOLO
from dotenv import load_dotenv
import cv2

load_dotenv()  # Load environment variables from .env file if present

app = Flask(__name__)

# Load models globally
classify_model = load_model('models/model_new.h5')
grade_model = load_model('models/model_grade_predict_dum.h5')
scaler = joblib.load('models/scaler.joblib')
grade_labels = ['A', 'B', 'C', 'D', 'E']

dirname = os.path.dirname(os.path.realpath(__file__))
ocr_model = YOLO(os.path.join(dirname, "core/models/detect-nutrition-label.pt"))

# Default values
default_values = {
    "energy": 1155.71,
    "protein": 6.1,
    "fat": 7.8,
    "carbs": 17.86,
    "sugar": 4.2,
    "sodium": 0.24,
    "saturated_fat": 0.0,
    "fiber": 0.0
}

def calculate_bmr(gender, age, body_height, body_weight, activity):
    body_height_cm = body_height * 100
    if gender == 0:  # Female
        bmr = 447.593 + (9.247 * body_weight) + (3.098 * body_height_cm) - (4.330 * age)
    else:  # Male
        bmr = 88.362 + (13.397 * body_weight) + (4.799 * body_height_cm) - (5.677 * age)
    if activity == 1:
        tdee = bmr * 1.465
    elif activity == 2:
        tdee = bmr * 1.2
    else:
        tdee = bmr * 1.8125

    return tdee

def normalize_nutrition_facts(prediction):
    # Mapping for OCR terms to standard keys
    ocr_to_standard_key = {
        "fat": "fat",
        "total fat": "fat",
        "lemak": "fat",
        "lemak total": "fat",
        "garam": "sodium",
        "natrium": "sodium",
        "sodium": "sodium",
        "lemak jenuh": "saturated_fat",
        "jenuh": "saturated_fat",
        "saturated fat": "saturated_fat",
        "protein": "protein",
        "total energi": "energy",
        "energi": "energy",
        "energy": "energy",
        "total energy": "energy",
        "fiber": "fiber",
        "serat pangan": "fiber",
        "dietary fiber": "fiber",
        "serat": "fiber",
        "total carbohydrate": "carbs",
        "karbohidrat total": "carbs",
        "karbohidrat": "carbs",
        "carbohydrate": "carbs",
        "sugar": "sugar",
        "gula": "sugar"
    }

    # Initialize an empty dictionary to hold the aggregated values
    aggregated_values = {key: 0 for key in default_values}

    # Sum the values from OCR predictions
    for key, value in prediction.items():
        standard_key = ocr_to_standard_key.get(key.lower())
        if standard_key:
            aggregated_values[standard_key] += float(value)

    # Fill in missing values with defaults
    for key in default_values:
        if aggregated_values[key] == 0:
            aggregated_values[key] = default_values[key]

    return aggregated_values

@app.route("/", methods=["GET"])
def main():
    return "Hello, world!"

@app.route("/ocr", methods=["POST"])
def process_image_and_predict():
    try:
        req = request.get_json()
        image_url = req["url"]

        try:
            # read the image url, and convert to numpy array
            image = skimage.io.imread(image_url)
            # convert to 3 channels (ignore the alpha)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            return jsonify({"error": f"Image failed to load: {e}"}), 500

        try:
            nutrients_txt_path = os.path.join(dirname, "core/data/nutrients.txt")
            tessdata_dir = os.path.join(dirname, "core/data/tessdata")

            try:
                prediction = core_ocr(image, ocr_model, tessdata_dir, nutrients_txt_path, debug=True)
            except Exception as e:
                return jsonify({"error": f"Something went wrong with the OCR! {e}"}), 500

            nutrition_facts = normalize_nutrition_facts(prediction)

            input_data = np.array([[nutrition_facts['energy'], nutrition_facts['protein'], nutrition_facts['fat'],
                                    nutrition_facts['carbs'], nutrition_facts['sugar'], nutrition_facts['sodium'],
                                    nutrition_facts['saturated_fat'], nutrition_facts['fiber']]])
            input_data = preprocess_input(input_data)

            predictions = grade_model.predict(input_data)
            predicted_grade_index = np.argmax(predictions)
            grade = grade_labels[predicted_grade_index]

            result = {
                "nutrition_facts": nutrition_facts,
                "grade": grade
            }

            return jsonify(result), 200

        except (FileNotFoundError, ValueError) as e:
            return jsonify({"error": str(e)}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    except KeyError:
        return jsonify({"error": "Missing 'url' field in the request"}), 400

@app.route('/grade', methods=['POST'])
def grade_endpoint():
    content = request.json
    try:
        nutrition_facts = {
            "energy": float(content['energy']),
            "protein": float(content['protein']),
            "fat": float(content['fat']),
            "carbs": float(content['carbs']),
            "sugar": float(content['sugar']),
            "sodium": float(content['sodium']),
            "saturated_fat": float(content['saturated_fat']),
            "fiber": float(content['fiber'])
        }

        input_data = np.array([[nutrition_facts['energy'], nutrition_facts['protein'], nutrition_facts['fat'],
                                nutrition_facts['carbs'], nutrition_facts['sugar'], nutrition_facts['sodium'],
                                nutrition_facts['saturated_fat'], nutrition_facts['fiber']]])
        input_data = preprocess_input(input_data)

        predictions = grade_model.predict(input_data)
        predicted_grade_index = np.argmax(predictions)
        grade = grade_labels[predicted_grade_index]

        result = {
            "nutrition_facts": nutrition_facts,
            "grade": grade
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'message': 'Error processing the request', 'details': str(e)}), 400

@app.route('/classify', methods=['POST'])
def classify():
    content = request.json
    try:
        gender = int(content['gender'])
        age = float(content['age'])
        body_weight = float(content['body_weight'])
        body_height = float(content['body_height'])
        activity_level = int(content['activity_level'])

        calories = calculate_bmr(gender, age, body_height, body_weight, activity_level)

        user_data = np.array([[gender, age, body_height, body_weight, activity_level]])
        user_data = scaler.transform(user_data)

        predictions = classify_model.predict(user_data)
        predicted_weight_status_index = np.argmax(predictions)
        weight_status = {
            0: 'Insufficient_Weight',
            1: 'Normal_Weight',
            2: 'Obesity_Type_I',
            3: 'Obesity_Type_II',
            4: 'Obesity_Type_III',
            5: 'Overweight_Level_I',
            6: 'Overweight_Level_II'
        }

        result = {
            "weight_status": weight_status[predicted_weight_status_index],
            "calories": calories
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'message': 'Error processing the request', 'details': str(e)}), 400

if __name__ == "__main__":
    print("Models loaded!")
    app.run(debug=True, host="0.0.0.0", port=8080)
