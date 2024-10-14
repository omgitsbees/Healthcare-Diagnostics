Diagnostics App

This is a Diagnostics Application that integrates a user-friendly interface for managing doctors, patients, and their medical data, while also offering machine learning models for diagnostics, text analysis (NLP), and image processing. The app uses various libraries and tools, including SQLite for database management, scikit-learn for machine learning models, and TensorFlow for image classification.
Features

    Patient & Doctor Management:
        Register new patients and doctors.
        Secure login for both patients and doctors using password hashing.
        View patient medical history and assign patients to doctors.

    Machine Learning Models:
        Logistic Regression model for medical diagnosis based on patient data.
        Naive Bayes NLP model for text classification and analysis.
        CNN-based Image Classification model for medical image diagnostics.

Table of Contents

    Installation
    Usage
    Project Structure
    Technologies Used
    Models
    Future Enhancements

Installation

    Clone the Repository:

    bash

git clone https://github.com/yourusername/diagnostics-app.git
cd diagnostics-app

Install Dependencies: You need to install the following libraries:

bash

pip install pandas numpy scikit-learn nltk tensorflow opencv-python

Download NLTK Data (for NLP):

python

    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    Setup Database: The SQLite database will be created automatically when you run the app.

Usage

Run the main.py file to launch the GUI application:

bash

python main.py

The GUI provides options for:

    Registering new patients and doctors.
    Logging in as a patient or doctor.
    Viewing and adding patients to doctors.
    Training different machine learning models for diagnostics.

Project Structure

bash

diagnostics-app/
│
├── diagnostics.db            # SQLite database (auto-created)
├── diagnostics.csv           # Dataset for medical diagnostics
├── images/                   # Image data for training image model
│   ├── train/
│   └── test/
├── nlp_dataset.csv           # Dataset for NLP text classification
├── main.py                   # Main application script
├── requirements.txt          # Required Python libraries
└── README.md                 # Project readme file

Technologies Used

    Languages: Python, SQL (SQLite)
    Libraries:
        tkinter: GUI for application
        scikit-learn: Machine learning models
        nltk: Natural Language Processing
        tensorflow/keras: Image classification
        opencv: Image processing
        sqlite3: Database management

Models

    Logistic Regression:
        A logistic regression model is trained on medical data to predict diagnoses.
        Dataset: diagnostics.csv
        Accuracy: Evaluated using accuracy_score.

    NLP Model (Naive Bayes):
        Text classification using Naive Bayes for predicting medical outcomes based on patient records.
        Dataset: nlp_dataset.csv
        Metrics: Precision, Recall, F1-Score.

    Image Classification (CNN):
        A Convolutional Neural Network (CNN) model is trained on medical images for classification.
        Dataset: Images in images/train and images/test.
        Metrics: Accuracy after training and evaluation.

Future Enhancements

    Add more machine learning models such as decision trees or random forests.
    Improve the GUI design for a better user experience.
    Add more features to patient and doctor management, such as updating information.
    Expand the dataset and improve model accuracy for NLP and image classification tasks.
