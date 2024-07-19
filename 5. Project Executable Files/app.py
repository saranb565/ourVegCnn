from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model(r"C:\Users\DELL\Desktop\GoogleIntern\myProject\vegetable_classification_model.h5")

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
        classes = ['Bean','Bitter_Gourd','Bottle_Gourd','Brinjal','Broccoli','Cabbage','Capsicum','Carrot','Cauliflower','Cucumber','Papaya','Potato','Pumpkin','Radish','Tomato']
        # Update this list with your actual classes
        prediction = classes[predicted_class[0]]
        
        return render_template("index.html", prediction=prediction)
    
    return render_template("index.html", prediction="Error in file upload")

if __name__ == '__main__':
    app.run(debug=False)
