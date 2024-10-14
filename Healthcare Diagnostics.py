import getpass
import hashlib
import tkinter as tk
from tkinter import messagebox
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a class for Patient
class Patient:
    def __init__(self, name, age, medical_history):
        self.name = name
        self.age = age
        self.medical_history = medical_history

# Define a class for Doctor
class Doctor:
    def __init__(self, name, specialty):
        self.name = name
        self.specialty = specialty
        self.patients = []

# Define a class for the Diagnostics App
class DiagnosticsApp:
    def __init__(self):
        self.patients = {}
        self.doctors = {}
        self.conn = sqlite3.connect('diagnostics.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients
            (name TEXT, age INTEGER, medical_history TEXT)
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctors
            (name TEXT, specialty TEXT)
        ''')
        self.conn.commit()

    def register_patient(self):
        name = input("Enter patient name: ")
        age = input("Enter patient age: ")
        medical_history = input("Enter patient medical history: ")
        patient = Patient(name, age, medical_history)
        self.patients[name] = patient
        self.cursor.execute('INSERT INTO patients VALUES (?, ?, ?)', (name, age, medical_history))
        self.conn.commit()
        print("Patient registered successfully!")

    def register_doctor(self):
        name = input("Enter doctor name: ")
        specialty = input("Enter doctor specialty: ")
        doctor = Doctor(name, specialty)
        self.doctors[name] = doctor
        self.cursor.execute('INSERT INTO doctors VALUES (?, ?)', (name, specialty))
        self.conn.commit()
        print("Doctor registered successfully!")

    def login_patient(self):
        name = input("Enter patient name: ")
        password = getpass.getpass("Enter patient password: ")
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.cursor.execute('SELECT * FROM patients WHERE name = ?', (name,))
        patient = self.cursor.fetchone()
        if patient and patient[1] == hashed_password:
            print("Login successful!")
            print("Medical History:", patient[2])
        else:
            print("Patient not found or incorrect password!")

    def login_doctor(self):
        name = input("Enter doctor name: ")
        password = getpass.getpass("Enter doctor password: ")
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.cursor.execute('SELECT * FROM doctors WHERE name = ?', (name,))
        doctor = self.cursor.fetchone()
        if doctor and doctor[1] == hashed_password:
            print("Login successful!")
            print("Patients:")
            for patient in self.patients.values():
                print(patient.name)
        else:
            print("Doctor not found or incorrect password!")

    def add_patient_to_doctor(self):
        doctor_name = input("Enter doctor name: ")
        patient_name = input("Enter patient name: ")
        if doctor_name in self.doctors and patient_name in self.patients:
            doctor = self.doctors[doctor_name]
            patient = self.patients[patient_name]
            doctor.patients.append(patient)
            print("Patient added to doctor's list successfully!")
        else:
            print("Doctor or patient not found!")

    def train_model(self):
        # Load dataset
        dataset = pd.read_csv('diagnostics.csv')
        X = dataset.drop(['diagnosis'], axis=1)
        y = dataset['diagnosis']
        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Model accuracy:", accuracy)

    def train_nlp_model(self):
        # Load dataset
        dataset = pd.read_csv('nlp_dataset.csv')
        X = dataset['text']
        y = dataset['label']
        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        # Train model
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)
        # Evaluate model
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print("NLP model accuracy:", accuracy)
        precision = precision_score(y_test, y_pred)
        print("NLP model precision:", precision)
        recall = recall_score(y_test, y_pred)
        print("NLP model recall:", recall)
        f1 = f1_score(y_test, y_pred)
        print("NLP model F1 score:", f1)

    def train_image_model(self):
        # Load dataset
        train_dir = 'images/train'
        test_dir = 'images/test'
        # Create data generators
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
        test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
        # Create model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Train model
        model.fit(train_generator, epochs=10, validation_data=test_generator)
        # Evaluate model
        loss, accuracy = model.evaluate(test_generator)
        print("Image model accuracy:", accuracy)

def main():
    app = DiagnosticsApp()
    root = tk.Tk()
    root.title("Diagnostics App")
    label = tk.Label(root, text="Welcome to the Diagnostics App!")
    label.pack()
    button = tk.Button(root, text="Register Patient", command=app.register_patient)
    button.pack()
    button = tk.Button(root, text="Register Doctor", command=app.register_doctor)
    button.pack()
    button = tk.Button(root, text="Login Patient", command=app.login_patient)
    button.pack()
    button = tk.Button(root, text="Login Doctor", command=app.login_doctor)
    button.pack()
    button = tk.Button(root, text="Add Patient to Doctor", command=app.add_patient_to_doctor)
    button.pack()
    button = tk.Button(root, text="Train Model", command=app.train_model)
    button.pack()
    button = tk.Button(root, text="Train NLP Model", command=app.train_nlp_model)
    button.pack()
    button = tk.Button(root, text="Train Image Model", command=app.train_image_model)
    button.pack()
    root.mainloop()

if __name__ == "__main__":
    main()