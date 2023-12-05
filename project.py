# Python Footbal Face Recognition
# Naufal Hardiansyah / 2540117855
# Raymond Prasetio / 2501996846

import cv2
import os
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

# Constants
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5
 # https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
FACE_RECOGNIZER = cv2.face.LBPHFaceRecognizer.create()
DATASET_PATH = 'Dataset/'

def preprocess(img):
    # Apply image preprocessing techniques
    img = cv2.resize(img, (150, 150))  # Resize the image
    img = cv2.equalizeHist(img)  # Equalize histogram for improved contrast
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur
    img = cv2.bilateralFilter(img, 5, 150, 150)  # Apply bilateral filter
    return img

def trainAndTest():
    print("Training and Testing Model")
    face_list = [] #stores face
    class_list = [] #stores the name
    

    # Initialize a list to store the predicted labels
    names = os.listdir(DATASET_PATH)
    for idx, name in enumerate(names):
        full_path = DATASET_PATH + '/' + name
        for img_name in os.listdir(full_path):
            image_path = full_path + '/' + img_name
            img = cv2.imread(image_path, 0)
            img = preprocess(img)

            # Detect face
            # scaleFactor = everytime the image is scanned, its size will decrease by 20%
            # minNeighbors = the amount of neighbors that 
            detected_face = FACE_CASCADE.detectMultiScale(img, scaleFactor = SCALE_FACTOR, minNeighbors = MIN_NEIGHBORS)
            if len(detected_face) < 1:
                continue

            for face in detected_face:
                x, y, h, w = face
                face_img = img[y:y+h, x:x+w]            
                face_list.append(face_img)
                class_list.append(idx)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        face_list, 
        class_list, 
        test_size = 0.25, 
        random_state = 0,
        stratify = class_list
    )
    
    # Train the recognizer on the training data
    FACE_RECOGNIZER.train(X_train, np.array(y_train))
    
    # Test the recognizer on the test data
    predicted_labels = []
    for test_img in X_test:
        res, confidence = FACE_RECOGNIZER.predict(test_img)
        predicted_labels.append(res)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%") 

def predict():
    print("Predicting")
    predict_path = input("Input absolute path of the image to predict: ")
    predict_img = cv2.imread(predict_path, 0)
    # predict_img = preprocess(predict_img)
    detected_face = FACE_CASCADE.detectMultiScale(predict_img, scaleFactor = SCALE_FACTOR, minNeighbors = MIN_NEIGHBORS)
    names = os.listdir(DATASET_PATH)
        
    if len(detected_face) < 1:
        print("No Face Detected")
        return
    
    for face in detected_face:
        x, y, h, w = face
        face_img = predict_img[y:y+h, x:x+w]    
        res, confidence = FACE_RECOGNIZER.predict(face_img)
        confidence = math.floor(confidence * 100) / 100
        
        #Draws a rectangle at the face
        img_bgr = cv2.imread(predict_path)
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0,255,0), 1)
        text = names[res] + " : " + str(confidence) + "%"
        cv2.putText(img_bgr, text,(x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 0)
        cv2.imshow('Result', img_bgr)
        cv2.waitKey(0)        
        

# Main program
# https://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == "__main__":
    while True:
        print("Python Footbal Face Recognition")
        print("1. Train and Test Model")
        print("2. Predict")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            trainAndTest()
        elif choice == '2':
            predict()
        elif choice == '3':
            break
        else:
            print("Invalid Choice")
