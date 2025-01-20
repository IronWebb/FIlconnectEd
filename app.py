from flask import Flask, jsonify, send_file, request
from mediapipe_model_maker import gesture_recognizer
import os
import requests
import zipfile
import tempfile
import shutil

# Initialize Flask app
app = Flask(__name__)

# Dropbox direct download URL
DATASET_URL = "https://www.dropbox.com/scl/fo/kp6gjrwc86dkx0ont30z3/ANa8AsZsx0h6i0NUxvpEoWk?rlkey=9r6y5d1fpv7xqklnpowsnfzu6&dl=1"

# Function to download and extract dataset to a temporary directory
def load_dataset_from_dropbox():
    print("Fetching dataset from Dropbox...")
    response = requests.get(DATASET_URL, stream=True)
    if response.status_code == 200:
        # Create a temporary directory to store the dataset
        temp_dir = tempfile.mkdtemp()

        # Save the downloaded zip file to the temporary directory
        dataset_zip_path = os.path.join(temp_dir, "gesture_dataset.zip")
        with open(dataset_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        # Extract the zip file into the temporary directory
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Return the path to the extracted dataset
        extracted_dataset_path = os.path.join(temp_dir, "gesture_dataset")
        return extracted_dataset_path
    else:
        raise Exception(f"Failed to fetch dataset from Dropbox. Status code: {response.status_code}")

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Gesture Recognition Model API!"})

@app.route('/train', methods=['GET'])
def train_model():
    try:
        # Fetch dataset directly from Dropbox and extract to a temporary directory
        dataset_path = load_dataset_from_dropbox()

        # Load and preprocess dataset from the temporary directory
        labels = [i for i in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, i))]
        print(f"Labels: {labels}")

        data = gesture_recognizer.Dataset.from_folder(
            dirname=dataset_path,
            hparams=gesture_recognizer.HandDataPreprocessingParams()
        )
        train_data, rest_data = data.split(0.8)
        validation_data, test_data = rest_data.split(0.5)

        # Train the model
        hparams = gesture_recognizer.HParams(export_dir="exported_model")
        options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
        model = gesture_recognizer.GestureRecognizer.create(
            train_data=train_data,
            validation_data=validation_data,
            options=options
        )

        # Evaluate the model
        loss, acc = model.evaluate(test_data, batch_size=1)
        print(f"Test loss: {loss}, Test accuracy: {acc}")

        # Export the model
        model.export_model()
        task_file = "exported_model/gesture_recognizer.task"

        # Ensure the file exists
        if os.path.exists(task_file):
            return jsonify({
                "message": "Model trained and exported successfully!",
                "test_loss": loss,
                "test_accuracy": acc,
                "download_url": "/download-task"
            })
        else:
            return jsonify({"error": "Model training completed but .task file was not found!"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download-task', methods=['GET'])
def download_task():
    try:
        task_file = "exported_model/gesture_recognizer.task"
        if os.path.exists(task_file):
            return send_file(task_file, as_attachment=True)
        else:
            return jsonify({"error": "No task file found! Train the model first."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)
