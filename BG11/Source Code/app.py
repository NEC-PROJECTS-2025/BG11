import os
from flask import Flask, render_template, request, jsonify, send_from_directory, g
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create 'uploads' folder if it doesn't exist
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"csv"}  # Only allow CSV files

# Load dataset for preprocessing reference
DATA_FILE = "heart.csv"
df = pd.read_csv(DATA_FILE)

# Define categorical and numerical columns
categorical_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numerical_columns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Encode categorical features
encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    df[col] = encoders[col].fit_transform(df[col])

# Scale numerical values
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Load Model
MODEL_FILE = "my_model.h5"

def get_model():
    """Load the CNN model only once per request for efficiency."""
    if "model" not in g:
        g.model = tf.keras.models.load_model(MODEL_FILE)
    return g.model

def allowed_file(filename):
    """Check if the uploaded file has a valid CSV extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_csv(file_path=None, input_data=None):
    """
    Read and preprocess CSV file or form input before prediction.
    """
    try:
        if file_path:
            df = pd.read_csv(file_path)
        elif input_data:
            df = pd.DataFrame([input_data])  # Convert form input to DataFrame
        else:
            return None, "No data provided."

        # Validate required columns
        required_columns = numerical_columns + categorical_columns
        if not all(col in df.columns for col in required_columns):
            return None, "CSV file is missing required columns."

        # Convert numerical columns to float
        df[numerical_columns] = df[numerical_columns].astype(float)
        df[numerical_columns] = scaler.transform(df[numerical_columns])  # Scale

        # Encode categorical features
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else 0)

        # Reshape for CNN input
        processed_array = df.values.reshape(len(df), -1, 1)
        return processed_array, None

    except Exception as e:
        return None, str(e)

# 🔹 **Home Page**
@app.route('/')
def home():
    return render_template('home.html')

# 🔹 **Upload Page (`uploads.html`)**
@app.route('/uploads', methods=['GET'])
def upload_page():
    return render_template('uploads.html')
 # Correctly serves the upload page

# 🔹 **Upload CSV File for Prediction**
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles CSV file upload and returns predictions."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), 400

    # Save file to uploads folder
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Preprocess uploaded CSV
    processed_data, error = preprocess_csv(file_path=file_path)
    if error:
        return jsonify({"error": error}), 400

    # Make Predictions
    model = get_model()
    predictions = model.predict(processed_data)
    y_pred = ["Heart Disease Detected" if p[0] > 0.5 else "No Heart Disease" for p in predictions]

    # Response JSON
    response = {
        "message": f"File '{file.filename}' uploaded and processed successfully.",
        "predictions": y_pred
    }
    return jsonify(response)

# 🔹 **Prediction Page**
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'GET':
#         return render_template('predict.html')

#     if request.method == 'POST':
#         try:
#             input_data = request.form.to_dict()

#             # Convert numerical inputs to float
#             for col in numerical_columns:
#                 input_data[col] = float(input_data[col])

#             # Preprocess input
#             processed_input, error = preprocess_csv(input_data=input_data)
#             if error:
#                 return render_template('predict.html', error=error)

#             # Make prediction
#             prediction = get_model().predict(processed_input)
#             result = "Heart Disease Detected" if prediction[0][0] > 0.5 else "No Heart Disease"

#             return render_template('predict.html', prediction=result)

#         except Exception as e:
#             return render_template('predict.html', error=str(e))

# 🔹 **Prediction Page (GET - Shows the form)**
@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

# 🔹 **Prediction API (POST - Handles form submission)**
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form data to dictionary
        input_data = request.form.to_dict()
        print("Received data:", input_data)  # Debug print

        # Convert numerical inputs to float
        numerical_columns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        for col in numerical_columns:
            input_data[col] = float(input_data[col])

        # Convert categorical values to numerical encoding
        categorical_mapping = {
            "Sex": {"M": 1, "F": 0},
            "ChestPainType": {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3},
            "FastingBS": {"0": 0, "1": 1},
            "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
            "ExerciseAngina": {"N": 0, "Y": 1},
            "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2}
        }
        for col, mapping in categorical_mapping.items():
            input_data[col] = mapping[input_data[col]]

        # Convert to NumPy array (Ensure all values are floats)
        input_array = np.array([list(map(float, input_data.values()))], dtype=np.float32)
        print("Input shape:", input_array.shape)  # Debug print

        # Load model
        model = get_model()
        print("Model loaded successfully!")  # Debug print

        # Make prediction
        prediction = model.predict(input_array)
        print("Raw prediction output:", prediction)  # Debug print

        result = "Heart Disease Detected" if prediction[0][0] > 0.5 else "No Heart Disease"
        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", str(e))  # Debug print
        return jsonify({"error": str(e)}), 500


# 🔹 **Metrics Page**
@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

# 🔹 **About Page**
@app.route('/about')
def about():
    return render_template('about.html')

# 🔹 **Flowchart Page**
@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

# 🔹 **Favicon Route**
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# 🔹 **Run Flask App**
if __name__ == '__main__':
    app.run(debug=True)
