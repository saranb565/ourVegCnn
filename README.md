Vegetable Image Classification Project

Table of Contents:
Project Overview
Directory Structure
Installation
Usage
Model Information
Contact


Project Overview
This project involves the development of a vegetable image classification model. The dataset consists of 21,000 images categorized into 15 classes. The project aims to accurately classify these images using a machine learning model, which is deployed using a Flask application.


Directory Structure
The project is organized into the following directories:


Project Initialization and Planning Phase:

Initial project setup and planning documentation.
Status: Files added yesterday.


Data Collection and Preprocessing Phase:

Scripts and data for collecting and preprocessing the dataset.
Status: Files added yesterday.


Model Development Phase:

Code for developing the classification model.
Status: Files added 1 hour ago.


Model Optimization and Tuning Phase:

Scripts for optimizing and tuning the model.
Status: Files added yesterday.


Project Executable Files:

Executable files for running the project.
Status: Files added 2 days ago.


Documentation & Demonstration:

Documentation and demonstration files.
Status: Ongoing updates.


Installation
To run the project locally, follow these steps:

Clone the repository:
git clone https://github.com/saranb565/vegetable-image-classification.git
cd vegetable-image-classification

Install the required dependencies:
pip install -r requirements.txt

Run the Flask application:
python app.py


Usage:
Upload vegetable images through the Flask application to get classification results. The application uses a pre-trained model to classify the images into one of the 15 categories.

Model Information:
The classification model is developed using TensorFlow. The model file vegetable_classification_model.h5 is loaded in the Flask application to make predictions.
