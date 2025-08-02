# Arch Technologies Internship Projects
Made by - **MOHAMMAD MUSA ALI** 

This repository contains four machine learning projects completed during my internship:

1. **Real-time Object Detection (YOLO)**
2. **Facial Emotion Recognition**
3. **Iris Flower Classification**
4. **House Price Prediction**

## 🚀 Project 1: Real-time Object Detection with YOLO

### Requirements
bash
pip install opencv-python numpy
How to Run
Download YOLO files:

bash
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
Run detection:

bash
python real_time_object_detection.py
😊 Project 2: Facial Emotion Recognition
Requirements
bash
pip install tensorflow opencv-python pandas matplotlib
Dataset Setup
Download FER-2013 dataset:

python
import pandas as pd
url = "https://storage.googleapis.com/kaggle-data-sets/1389035/2349265/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256"
pd.read_csv(url).to_csv('fer2013.csv')
Download Haar Cascade:

bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
How to Run
bash
python train_emotion_classifier.py  # Train model (30-50 mins)
python emotion_classification.py    # Real-time detection

🌸 Project 3: Iris Flower Classification
Requirements
bash
pip install scikit-learn pandas
How to Run
bash
python iris_classification.py
Note: Automatically downloads the iris dataset from sklearn

🏠 Project 4: House Price Prediction
Requirements
bash
pip install pandas scikit-learn matplotlib
How to Run
Add housing.csv to the project folder

Run:

bash
python house_price_prediction.py
📂 File Structure
text
Arch-Technologies/
├── Object-Detection/
│   ├── real_time_object_detection.py
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   └── coco.names
├── Emotion-Recognition/
│   ├── train_emotion_classifier.py
│   ├── emotion_classification.py
│   ├── fer2013.csv
│   └── haarcascade_frontalface.xml
├── Iris-Classification/
│   └── iris_classification.py
├── House-Prediction/
│   ├── house_price_prediction.py
│   └── housing.csv
└── README.md
⚠️ Troubleshooting
For YOLO:

Error: "weights file not found"

Verify all 3 files exist in the same folder:

yolov3.cfg

yolov3.weights

coco.names

For Emotion Recognition:

FER-2013 download fails:

python
pd.read_csv("https://github.com/christianversloot/fer2013-downloader/raw/master/fer2013.csv").to_csv('fer2013.csv')
DLL errors on Windows:
Install Visual C++ Redistributable

For House Price Prediction:

Dataset not found:

Ensure housing.csv is in the same folder as the script

📜 License
MIT License - Free for academic and commercial use
