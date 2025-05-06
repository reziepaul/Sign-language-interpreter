import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Image processing parameters
imgSize = 300
offset = 20

# Directory to save images
folder = 'Data/thank you'
if not os.path.exists(folder):
    os.makedirs(folder)

counter = 0  # Counter for saved images

while True:
    success, img = cap.read()  # Read frame from the webcam
    hands, img = detector.findHands(img)  # Detect hands in the frame

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create white image for processed output
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        # Resize and fit the cropped image into the white canvas
        if aspectRatio > 1:  # If height > width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize

        else:  # If width >= height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Display processed and cropped images
        cv2.imshow("ImageWhite", imgWhite)
        cv2.imshow("ImageCrop", imgCrop)

    # Display the webcam feed
    cv2.imshow('Webcam', img)

    # Keyboard controls
    key = cv2.waitKey(1)
    if key == ord('s'):  # Save the processed image when 's' is pressed
        counter += 1
        filename = f'{folder}/Image_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(filename, imgWhite)
        print(f"Saved: {filename} (Total: {counter})")

    elif key == ord('q'):  # Quit when 'q' is pressed
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
