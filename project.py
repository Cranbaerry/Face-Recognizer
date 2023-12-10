# Python Footbal Face Recognition
# Naufal Hardiansyah / 2540117855
# Raymond Prasetio / 2501996846
# Repository: https://github.com/Cranbaerry/Face-Recognizer

import cv2
import os
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FaceRecognition:
    def __init__(self):
        # Initialize the parameters
        self.SCALE_FACTOR = 1.2
        self.MIN_NEIGHBORS = 5
        self.DATASET_PATH = 'Dataset/'
        self.GAUSSIAN_BLUR = lambda img: cv2.GaussianBlur(img, (13, 13), 0)
        self.BILATERAL_FILTER = lambda img: cv2.bilateralFilter(img, 10, 100, 100)
        self.LABELS = os.listdir(self.DATASET_PATH)
        
        # Initialize the face cascade classifier and face recognizer        
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer.create()

    def preprocess(self, img):
        # Apply Gaussian blur
        img = self.GAUSSIAN_BLUR(img)

        # Apply Bilateral filter
        img = self.BILATERAL_FILTER(img)
        return img

    def train_and_test(self):
        print("Training and Testing Model")
        face_list = []  # stores face
        class_list = []  # stores the name

        # Initialize a list to store the predicted labels
        for idx, name in enumerate(self.LABELS):
            full_path = self.DATASET_PATH + '/' + name
            for img_name in os.listdir(full_path):
                image_path = full_path + '/' + img_name
                img = cv2.imread(image_path, 0)
                img = self.preprocess(img)

                # Detect face
                detected_face = self.face_cascade.detectMultiScale(
                    img, 
                    scaleFactor=self.SCALE_FACTOR,
                    minNeighbors=self.MIN_NEIGHBORS
                )
                if len(detected_face) < 1:
                    continue

                for face in detected_face:
                    x, y, h, w = face
                    face_img = img[y:y + h, x:x + w]
                    face_list.append(face_img)
                    class_list.append(idx)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            face_list,
            class_list,
            test_size=0.25,
            random_state=0,
            stratify=class_list
        )

        # Train the recognizer on the training data
        self.face_recognizer.train(X_train, np.array(y_train))

        # Test the recognizer on the test data
        predicted_labels = []
        for test_img in X_test:
            res, confidence = self.face_recognizer.predict(test_img)
            predicted_labels.append(res)

        # Calculate accuracy of the model
        accuracy = accuracy_score(y_test, predicted_labels)
        print(f"Accuracy: {accuracy * 100:2f}%")
        
        # Save the model
        self.face_recognizer.save('model.xml')

    def predict(self):
        # Check if the model exists
        if not os.path.exists('model.xml'):
            print("Model not found. Please train the model first.")
            return
        
        # Load the model
        self.face_recognizer.read('model.xml')
        
        # Predict the image        
        predict_path = input("Input absolute path of the image to predict: ")
        predict_img = cv2.imread(predict_path, 0)
        # Preprocessing disabled for predicting
        # It gives 0% confidence despite predicting the correct result ¯\_(ツ)_/¯
        # predict_img = self.preprocess(predict_img)
        
        detected_face = self.face_cascade.detectMultiScale(
            predict_img, 
            scaleFactor=self.SCALE_FACTOR,
            minNeighbors=self.MIN_NEIGHBORS
        )
        
        if len(detected_face) < 1:
            print("No Face Detected")
            return

        for face in detected_face:
            x, y, h, w = face
            face_img = predict_img[y:y + h, x:x + w]
            res, confidence = self.face_recognizer.predict(face_img)
            print(res, confidence)
            confidence = math.floor(confidence * 100) / 100

            # Draws a rectangle at the face
            img_bgr = cv2.imread(predict_path)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)
            text = self.LABELS[res] + " : " + str(confidence) + "%"
            cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 0)
            cv2.imshow('Result', img_bgr)
            cv2.waitKey(0)

if __name__ == "__main__":
    face_recognition = FaceRecognition()
    while True:
        print("Python Football Face Recognition")
        print("1. Train and Test Model")
        print("2. Predict")
        print("3. Exit")
        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                face_recognition.train_and_test()
            elif choice == 2:
                face_recognition.predict()
            elif choice == 3:
                break
            else:
                print("Invalid Choice")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print("An error occurred:", str(e))
