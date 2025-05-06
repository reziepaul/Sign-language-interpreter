import cv2
import time
import numpy as np
import math
import mediapipe as mp
import os
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)
engine.setProperty('volume', 1.0)

# Initialize webcam with reduced resolution
cap = cv2.VideoCapture(0)
cap.set(3, 1000)   # Width
cap.set(4, 740)   # Height

# Ensure webcam opens
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2)

# Initialize hand detector and classifier
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/Keras_model.h5", "Model/labels.txt")

# Image processing parameters
imgSize = 600
offset = 20

# Labels for classification
labels = ["Hello", "I love you", "No" , "Sorry", "Thank You", "Yes"]

# Directory to save images
folder = 'Data/a'
os.makedirs(folder, exist_ok=True)
counter = 0
last_saved_time = time.time()

last_prediction = ""
last_spoken = ""

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from webcam.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgOutput = img.copy()

    hands_results = hands.process(img_rgb)

    if hands_results.multi_hand_landmarks:
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0

        for hand_landmarks in hands_results.multi_hand_landmarks:
            h, w, c = img.shape
            x_list = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
            y_list = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
            x_min, y_min, x_max, y_max = min(x_list), min(y_list), max(x_list), max(y_list)
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        x_min, y_min, x_max, y_max = max(0, x_min - offset), max(0, y_min - offset), min(img.shape[1], x_max + offset), min(img.shape[0], y_max + offset)

        if x_min < x_max and y_min < y_max:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y_min:y_max, x_min:x_max]

            if imgCrop.size > 0:
                aspectRatio = (y_max - y_min) / (x_max - x_min)
                if aspectRatio > 1:
                    k = imgSize / (y_max - y_min)
                    wCal = math.ceil(k * (x_max - x_min))
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / (x_max - x_min)
                    hCal = math.ceil(k * (y_max - y_min))
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                predicted_word = labels[index]

                if predicted_word != last_prediction:
                    last_prediction = predicted_word
                    if predicted_word != last_spoken:
                        engine.say(predicted_word)
                        engine.runAndWait()
                        last_spoken = predicted_word

                # Display prediction
                cv2.rectangle(imgOutput, (x_min, y_min - 50), (x_min + 150, y_min), (128, 0, 128), cv2.FILLED)
                cv2.putText(imgOutput, predicted_word, (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 1.7,
                            (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x_min, y_min), (x_max, y_max), (255, 0, 255), 4)

                # Save image every 2 seconds while hand is detected
                current_time = time.time()
                if current_time - last_saved_time > 0.5:
                    counter += 1
                    filename = f'{folder}/Image_{time.strftime("%Y%m%d_%H%M%S")}_{counter}.jpg'
                    cv2.imwrite(filename, imgWhite)
                    print(f'Detected: "{predicted_word}"')
                    last_saved_time = current_time

    # Show the webcam feed
    cv2.imshow('Webcam', imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
