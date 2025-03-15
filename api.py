from flask import Flask, request, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import threading
import pyttsx3  

app = Flask(__name__)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

detector = HandDetector(maxHands=2, detectionCon=0.8)  
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "Hi", "I", "K", "L", "Love", "M", "O", "P", "S", "T", "V", "W", "Z"]  
confidence_threshold = 0.7  

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return jsonify({"error": "No image provided"}), 400

    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    hands, _ = detector.findHands(img, draw=False)  

    if hands:   
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        x1, y1 = max(0, x - 20), max(0, y - 20)
        x2, y2 = min(img.shape[1], x + w + 20), min(img.shape[0], y + h + 20)

        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((300, 300, 3), np.uint8) * 255

        aspectRatio = h / w  
        if aspectRatio > 1:
            k = 300 / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, 300))
            wGap = math.ceil((300 - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = 300 / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (300, hCal))
            hGap = math.ceil((300 - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite)
        confidence = max(prediction)

        if confidence > confidence_threshold:
            predicted_text = labels[index]
            speak(predicted_text)
            return jsonify({"prediction": predicted_text, "confidence": confidence})
        else:
            return jsonify({"prediction": "Uncertain", "confidence": confidence})
    else:
        return jsonify({"error": "No hands detected"}), 400

if __name__ == '__main__':
    app.run(debug=True)