from flask import Flask, jsonify, request
import os
import numpy as np
from datetime import datetime
<<<<<<< HEAD
import mysql.connector
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
import skimage.io
from core.main import core_ocr
from dotenv import load_dotenv
=======
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.preprocessing import StandardScaler
import joblib
import skimage.io
from core.main import core_ocr
from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
>>>>>>> master

load_dotenv()  # Load environment variables from .env file if present

app = Flask(__name__)

<<<<<<< HEAD
# Configuration for Google Cloud Storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
storage_client = storage.Client()

# Establish a connection to the MySQL database using environment variables
db_connection = mysql.connector.connect(
    host=os.getenv('DB_HOST'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    database=os.getenv('DB_NAME')
)

# Load models
classify_model = load_model('models/model_checkpoint.h5')
grade_model = load_model('models/model_label_pred.h5')

# Initialize the scaler for classification
scaler = StandardScaler()
scaler = scaler.fit([[0, 0, 0, 0, 0]])

grade_labels = ['A', 'B', 'C', 'D', 'E']

=======
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

>>>>>>> master
def calculate_bmr(gender, age, body_height, body_weight, activity):
    body_height_cm = body_height * 100
    if gender == 0:  # Female
        bmr = 447.593 + (9.247 * body_weight) + (3.098 * body_height_cm) - (4.330 * age)
    else:  # Male
        bmr = 88.362 + (13.397 * body_weight) + (4.799 * body_height_cm) - (5.677 * age)
<<<<<<< HEAD

=======
>>>>>>> master
    if activity == 1:
        tdee = bmr * 1.465
    elif activity == 2:
        tdee = bmr * 1.2
    else:
        tdee = bmr * 1.8125

    return tdee

<<<<<<< HEAD
=======
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

>>>>>>> master
@app.route("/", methods=["GET"])
def main():
    return "Hello, world!"

@app.route("/ocr", methods=["POST"])
<<<<<<< HEAD
def ocr_endpoint():
    # get parent/working dir
    dirname = os.path.dirname(os.path.realpath(__file__))

    try:
        # get req body
        req = request.get_json()
        # get the image url stored in the body
=======
def process_image_and_predict():
    try:
        req = request.get_json()
>>>>>>> master
        image_url = req["url"]

        try:
            # read the image url, and convert to numpy array
            image = skimage.io.imread(image_url)
<<<<<<< HEAD
        except Exception as e:
            # if error, return error
            return jsonify({"error": f"Image failed to load: {e}"}), 500

        try:
            # load the nutrition label detection model
            model_path = os.path.join(dirname, "core/models/detect-nutrition-label.pt")
            # load the nutrients list text
            nutrients_txt_path = os.path.join(dirname, "core/data/nutrients.txt")
            # load local tessdata_dir
            tessdata_dir = os.path.join(dirname, "core/data/tessdata")

            try:
                # run the ocr
                prediction = core_ocr(
                    image, model_path, tessdata_dir, nutrients_txt_path, debug=True
                )

                # return the successfully read nutrition labels
                return jsonify(prediction)

            # catch specific errors from core_ocr
            except (
                FileNotFoundError,
                ValueError,
            ) as e:
                return (
                    jsonify({"": str(e)}),
                    400,
                )

            # or return 500 if it's a server-side/unexpected error
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # catch general model prediction errors
        except Exception as e:
            return jsonify({"error": f"Model prediction failed: {e}"}), 500

    # catch missing url
    except KeyError:
        return jsonify({"error": "Missing 'url' field in the request"}), 400

@app.route('/grade', methods=['POST'])
def grade_endpoint():
    content = request.json
    try:
        product_id = int(content['product_id'])

        cursor = db_connection.cursor()
        cursor.execute("SELECT energy, protein, fat, carbs, sugar, salt FROM products WHERE product_id = %s", (product_id,))
        result = cursor.fetchone()

        if result is not None:
            nutrition_facts = {
                "energy": float(result[0]),
                "protein": float(result[1]),
                "fat": float(result[2]),
                "carbs": float(result[3]),
                "sugar": float(result[4]),
                "salt": float(result[5])
            }

            # Prepare the input data for the model
            input_data = np.array([[nutrition_facts['energy'], nutrition_facts['protein'], nutrition_facts['fat'], 
                                    nutrition_facts['carbs'], nutrition_facts['sugar'], nutrition_facts['salt']]])
            input_data = preprocess_input(input_data)

            # Model prediction
=======
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

>>>>>>> master
            predictions = grade_model.predict(input_data)
            predicted_grade_index = np.argmax(predictions)
            grade = grade_labels[predicted_grade_index]

<<<<<<< HEAD
            # Save the prediction result and nutrition facts to the MySQL database
            cursor.execute(
                "INSERT INTO product_grades (product_id, grade, energy, protein, fat, carbs, sugar, salt, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (product_id, grade, nutrition_facts['energy'], nutrition_facts['protein'], nutrition_facts['fat'], nutrition_facts['carbs'], nutrition_facts['sugar'], nutrition_facts['salt'], datetime.now(), datetime.now())
            )
            db_connection.commit()

=======
>>>>>>> master
            result = {
                "nutrition_facts": nutrition_facts,
                "grade": grade
            }

            return jsonify(result), 200
<<<<<<< HEAD
        else:
            return jsonify({'message': 'No record found for the given product_id'}), 404

    except Exception as e:
        db_connection.rollback()
        return jsonify({'message': 'Error processing the request', 'details': str(e)}), 400


=======

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

>>>>>>> master
@app.route('/classify', methods=['POST'])
def classify():
    content = request.json
    try:
<<<<<<< HEAD
        user_id = int(content['id'])

        cursor = db_connection.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()

        if result is not None:
            gender = int(result[1])
            age = float(result[2])
            body_weight = float(result[4])
            body_height = float(result[3])
            activity_level = int(result[5])

            calories = calculate_bmr(gender, age, body_height, body_weight, activity_level)

            # Prepare the input data for the model
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

            # Save the prediction result to the MySQL database
            cursor.execute(
                "INSERT INTO weight_classification (weight_status, calories, created_at, updated_at) VALUES (%s, %s, %s, %s)",
                (weight_status[predicted_weight_status_index], calories, datetime.now(), datetime.now())
            )
            db_connection.commit()

            result = {
                "weight_status": weight_status[predicted_weight_status_index],
                "calories": calories
            }

            return jsonify(result), 200
        else:
            return jsonify({'message': 'No record found for the given id'}), 404

    except Exception as e:
        db_connection.rollback()
        return jsonify({'message': 'Error processing the request', 'details': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
=======
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
>>>>>>> master
