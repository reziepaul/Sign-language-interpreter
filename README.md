# Real-Time Sign Language Interpreter
📌 Project Overview

This project implements a real-time sign language interpreter that recognizes hand gestures from a live webcam feed and translates them into text (or optionally speech). The solution leverages computer vision and deep learning to facilitate inclusive communication for individuals with hearing or speech impairments.

The system uses:

OpenCV for webcam input and real-time image preprocessing.

MediaPipe for robust hand detection and landmark tracking.

CVZone to simplify hand tracking operations.

TensorFlow/Keras to build and train a custom Convolutional Neural Network (CNN) for gesture classification.

pyttsx3 (optional) to convert predicted text into speech for auditory feedback.


⚙️ Key Features

Real-time hand tracking with MediaPipe.

Custom CNN model trained on a self-collected dataset.

Preprocessing pipeline: Cropping gestures, resizing while maintaining aspect ratio, and placing them on a white square background.

Gesture recognition for words such as:

"Hello" 👋

"Thank You" 🙏

"Yes" 👍

"No" 👎

"I Love You" 🤟

"Sorry" 🤲

Live translation: Predictions displayed on screen and optionally spoken aloud.

Accessible design: Bridges communication between sign language users and non-sign language speakers.




🏗️ System Architecture

Input: Webcam feed using OpenCV.

Hand Detection: MediaPipe detects hand landmarks and bounding boxes.

Preprocessing:

Crop region of interest (ROI) around the hand.

Resize image proportionally.

Place resized image on a white square canvas.

Normalize and prepare for model input.

Prediction: Preprocessed frame passed to the trained CNN model.

Output: Gesture label displayed on screen and optionally converted to speech.



🛠️ Tech Stack

Programming Language: Python 3.x

Computer Vision: OpenCV, CVZone, MediaPipe

Deep Learning: TensorFlow / Keras

Text-to-Speech (Optional): pyttsx3

Environment: Jupyter Notebook / PyCharm / VSCode



📦 Dependencies

Install the following packages before running the project:

pip install opencv-python
pip install mediapipe
pip install cvzone
pip install tensorflow
pip install numpy
pip install matplotlib
pip install pyttsx3   # optional, for text-to-speech



📂 Dataset Preparation

Collect images of hand gestures for each class (e.g., "Hello", "Thank You").

Preprocess each image:

Detect the hand using MediaPipe.

Crop the bounding box around the hand.

Resize while maintaining aspect ratio.

Place the resized hand image on a white square background.

Label each class appropriately and split into train/validation/test sets.




🧠 Model Training (CNN)

Architecture:

Convolutional Layers (feature extraction).

Pooling Layers (downsampling).

Dense Layers (classification).

Softmax activation for multi-class output.

Optimizer: Adam

Loss: Categorical Crossentropy

Evaluation Metrics: Accuracy
