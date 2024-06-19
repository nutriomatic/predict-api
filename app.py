from flask import Flask, request, jsonify
from core.main import core_ocr
import skimage
import os
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet import preprocess_input
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from sqlalchemy import create_engine, exc, text
from sqlalchemy.orm import sessionmaker

load_dotenv()
# Initialize the app
app = Flask(__name__)

# Initialize database
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")

# Using TCP/IP for MySQL connection
db_uri = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

try:
    # Create a new engine
    engine = create_engine(db_uri)
    Session = sessionmaker(bind=engine)
    session = Session()
    print("Database connection successful")
except exc.SQLAlchemyError as e:
    print(f"Failed to connect to database: {e}")

classify_model = load_model('models/model_checkpoint.h5')
grade_model = load_model('models/model_grade_predict_dummy_4.h5')

# Initialize the scaler for classification
scaler = StandardScaler()
scaler = scaler.fit([[0, 0, 0, 0, 0]])

grade_labels = ['A', 'B', 'C', 'D', 'E']

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

# Load the models after the application starts
ocr_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "core/models/detect-nutrition-label.pt")
grade_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/model_grade_predict_dummy_4.h5")

model = YOLO(ocr_model_path)
grade_model = load_model(grade_model_path)
print("Models loaded!")

# Define grade labels
grade_labels = ['A', 'B', 'C', 'D', 'E']

# Route to test connection
@app.route("/", methods=["GET"])
def main():
    return "Hello, world!"

# Route to do OCR and grade
@app.route("/ocr", methods=["POST"])
def ocr_and_grade_endpoint():
    try:
        # Get request body
        req = request.get_json()
        # Get the image URL stored in the body
        image_url = req["url"]

        try:
            # Read the image URL, and convert to numpy array
            image = skimage.io.imread(image_url)
            # Convert to 3 channels (ignore the alpha)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            # If error, return error
            return jsonify({"error": f"Image failed to load: {e}"}), 500

        try:
            # Load the nutrients list text and tessdata_dir
            nutrients_txt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "core/data/nutrients.txt")
            tessdata_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "core/data/tessdata")

            # Run the OCR
            prediction = core_ocr(
                image, model, tessdata_dir, nutrients_txt_path, debug=True
            )

            if not prediction:
                return jsonify({"error": "OCR failed to extract any data"}), 500

            # Extract nutrition facts from OCR prediction
            nutrition_facts = {
                "energy": float(prediction.get('energy', 0)),
                "protein": float(prediction.get('protein', 0)),
                "fat": float(prediction.get('fat', 0)),
                "carbs": float(prediction.get('carbs', 0)),
                "sugar": float(prediction.get('sugar', 0)),
                "sodium": float(prediction.get('sodium', 0)),
                "saturated_fat": float(prediction.get('saturated-fat', 0)),
                "fiber": float(prediction.get('fiber', 0))
            }

            # Prepare the input data for the model
            input_data = np.array([[nutrition_facts['energy'], nutrition_facts['protein'], nutrition_facts['fat'], 
                                    nutrition_facts['carbs'], nutrition_facts['sugar'], nutrition_facts['sodium'],
                                    nutrition_facts['saturated_fat'], nutrition_facts['fiber']]])

            # Model prediction
            predictions = grade_model.predict(input_data)
            predicted_grade_index = np.argmax(predictions)
            grade = grade_labels[predicted_grade_index]

            # Save the prediction result and nutrition facts to the MySQL database
            with engine.connect() as connection:
                connection.execute(
                    text("INSERT INTO product_grades (grade, energy, protein, fat, carbs, sugar, sodium, saturated_fat, fiber, created_at, updated_at) VALUES (:grade, :energy, :protein, :fat, :carbs, :sugar, :sodium, :saturated_fat, :fiber, :created_at, :updated_at)"),
                    {'grade': grade, 'energy': nutrition_facts['energy'], 'protein': nutrition_facts['protein'], 'fat': nutrition_facts['fat'], 'carbs': nutrition_facts['carbs'], 'sugar': nutrition_facts['sugar'], 'sodium': nutrition_facts['sodium'], 'saturated_fat': nutrition_facts['saturated_fat'], 'fiber': nutrition_facts['fiber'], 'created_at': datetime.now(), 'updated_at': datetime.now()}
                )

            result = {
                "nutrition_facts": nutrition_facts,
                "grade": grade
            }

            return jsonify(result), 200

        # Catch general model prediction errors
        except Exception as e:
            return jsonify({"error": f"Model prediction failed: {e}"}), 500

    # Catch missing URL or product_id
    except KeyError:
        return jsonify({"error": "Missing 'url' field in the request"}), 400

@app.route('/grade', methods=['POST'])
def grade_endpoint():
    content = request.json
    try:
        product_id = int(content['product_id'])

        # Execute SQL query
        with engine.connect() as connection:
            result = connection.execute(text("SELECT energy, protein, fat, carbs, sugar, sodium, saturated_fat, fiber FROM products WHERE product_id = :product_id"), {'product_id': product_id})
            row = result.fetchone()

        if row:
            nutrition_facts = {
                "energy": float(row[0]),
                "protein": float(row[1]),
                "fat": float(row[2]),
                "carbs": float(row[3]),
                "sugar": float(row[4]),
                "sodium": float(row[5]),
                "saturated_fat": float(row[6]),
                "fiber": float(row[7])
            }

            # Prepare the input data for the model
            input_data = np.array([[nutrition_facts['energy'], nutrition_facts['protein'], nutrition_facts['fat'], 
                                    nutrition_facts['carbs'], nutrition_facts['sugar'], nutrition_facts['sodium'],
                                    nutrition_facts['saturated_fat'], nutrition_facts['fiber']]])
            input_data = preprocess_input(input_data)

            # Model prediction
            predictions = grade_model.predict(input_data)
            predicted_grade_index = np.argmax(predictions)
            grade = grade_labels[predicted_grade_index]

            # Save the prediction result and nutrition facts to the MySQL database
            with engine.connect() as connection:
                connection.execute(
                    text("INSERT INTO product_grades (product_id, grade, energy, protein, fat, carbs, sugar, sodium, saturated_fat, fiber, created_at, updated_at) VALUES (:product_id, :grade, :energy, :protein, :fat, :carbs, :sugar, :sodium, :saturated_fat, :fiber, :created_at, :updated_at)"),
                    {'product_id': product_id, 'grade': grade, 'energy': nutrition_facts['energy'], 'protein': nutrition_facts['protein'], 'fat': nutrition_facts['fat'], 'carbs': nutrition_facts['carbs'], 'sugar': nutrition_facts['sugar'], 'sodium': nutrition_facts['sodium'], 'saturated_fat': nutrition_facts['saturated_fat'], 'fiber': nutrition_facts['fiber'], 'created_at': datetime.now(), 'updated_at': datetime.now()}
                )

            result = {
                "nutrition_facts": nutrition_facts,
                "grade": grade
            }

            return jsonify(result), 200
        else:
            return jsonify({'message': 'No record found for the given product_id'}), 404

    except Exception as e:
        return jsonify({'message': 'Error processing the request', 'details': str(e)}), 400

@app.route('/classify', methods=['POST'])
def classify():
    content = request.json
    try:
        user_id = int(content['id'])

        # Retrieve user from database using SQLAlchemy session
        user = session.execute(text("SELECT * FROM users WHERE id = :id"), {'id': user_id}).fetchone()

        if user:
            gender = int(user[1])
            age = float(user[2])
            body_weight = float(user[4])
            body_height = float(user[3])
            activity_level = int(user[5])

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
            with engine.connect() as connection:
                connection.execute(
                    text("INSERT INTO weight_classification (weight_status, calories, created_at, updated_at) VALUES (:weight_status, :calories, :created_at, :updated_at)"),
                    {'weight_status': weight_status[predicted_weight_status_index], 'calories': calories, 'created_at': datetime.now(), 'updated_at': datetime.now()}
                )

            result = {
                "weight_status": weight_status[predicted_weight_status_index],
                "calories": calories
            }

            return jsonify(result), 200
        else:
            return jsonify({'message': 'No record found for the given id'}), 404

    except Exception as e:
        return jsonify({'message': 'Error processing the request', 'details': str(e)}), 400

if __name__ == "__main__":
    # Set debug=True for local development
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)