from flask import Flask, render_template, request
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Google Drive configuration
google_drive_file_id = '1A1qK8qHJ02_PJI_ziCXX2nRMapBx69A7'  # Replace with your actual file ID
model_file_path = 'vegetable_classification_model.h5'
google_drive_url = f'https://drive.google.com/uc?export=download&id={google_drive_file_id}'

# Function to download the model file from Google Drive
def download_model():
    if not os.path.exists(model_file_path):
        print("Downloading model from Google Drive...")
        response = requests.get(google_drive_url, stream=True)
        if response.status_code == 200:
            with open(model_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Model downloaded successfully: {model_file_path}")
        else:
            print(f"Failed to download model: {response.status_code}")
        # Verify if the file is downloaded correctly
        if os.path.exists(model_file_path):
            print(f"Downloaded model file size: {os.path.getsize(model_file_path)} bytes")
        else:
            print("Model file was not downloaded.")

# Ensure the model is downloaded before loading
download_model()

# Check if the model file exists and is not empty
if os.path.exists(model_file_path) and os.path.getsize(model_file_path) > 0:
    model = load_model(model_file_path)
else:
    raise FileNotFoundError("Model file not found or is empty.")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return render_template("index.html", prediction="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", prediction="No selected file")
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocess the image
        img = load_img(filepath, target_size=(224, 224))  # Use the size your model expects
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image
        
        # Make a prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        
        # You need a mapping from the predicted class index to the actual vegetable names
        classes = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
        # Update this list with your actual classes
        prediction = classes[predicted_class[0]]
        
        return render_template("index.html", prediction=prediction)
    
    return render_template("index.html", prediction="Error in file upload")

if __name__ == '__main__':
    app.run(debug=False)
