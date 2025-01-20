from flask import Flask, jsonify, send_file, request
from mediapipe_model_maker import gesture_recognizer
import os
import requests
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Dropbox direct download URL (you can use it as-is for direct access)
DATASET_URL = "https://www.dropbox.com/scl/fo/kp6gjrwc86dkx0ont30z3/ANa8AsZsx0h6i0NUxvpEoWk?rlkey=9r6y5d1fpv7xqklnpowsnfzu6&dl=1"
DATASET_PATH = "gesture_dataset"  # Direct access path to external storage

# Function to fetch dataset from Dropbox (no need to unzip)
def load_dataset_from_dropbox():
    print("Fetching dataset from Dropbox...")

    response = requests.get(DATASET_URL, stream=True)
    if response.status_code == 200:
        # You can directly process the data here without storing it locally
        # For example, stream it into memory or into a temporary folder
        print("Dataset fetched successfully.")
    else:
        raise Exception(f"Failed to fetch dataset from Dropbox. Status code: {response.status_code}")

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Gesture Recognition Model API!"})

@app.route('/train', methods=['GET'])
def train_model():
    try:
        # Fetch dataset directly from Dropbox
        load_dataset_from_dropbox()

        # Load and preprocess dataset (adjust to your structure)
        # Here we assume dataset files are directly accessible
        labels = [i for i in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, i))]
        print(f"Labels: {labels}")

        data = gesture_recognizer.Dataset.from_folder(
            dirname=DATASET_PATH,
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
