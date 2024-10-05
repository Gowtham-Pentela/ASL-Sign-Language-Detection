# ASL Sign Language Recognition Project

This project aims to recognize American Sign Language (ASL) signs using a combination of machine learning and computer vision techniques. It consists of three main Jupyter notebooks: **Labeling**, **Training**, and **Detection**.

## Table of Contents
1. [Requirements](#requirements)
2. [Notebook Descriptions](#notebook-descriptions)
3. [Usage Instructions](#usage-instructions)
4. [Acknowledgments](#acknowledgments)

## Requirements
To run this project, you will need:
- Python 3.x
- Jupyter Notebook
- TensorFlow
- OpenCV
- Mediapipe
- NumPy
- Pickle

You can install the required packages using pip:

```bash
pip install tensorflow opencv-python mediapipe numpy
```

## Notebook Descriptions

### 1. Labeling Notebook (`labeling.ipynb`)
This notebook is responsible for creating a label mapping for the ASL signs. It defines the signs associated with each label (0-27), representing letters and special characters (like space and nothing). The output is a `label_dict.pickle` file that contains the label mapping, which is used in the training and detection stages.

### 2. Training Notebook (`training.ipynb`)
In this notebook, a Convolutional Neural Network (CNN) model is trained on a dataset of ASL sign images. The model learns to classify the signs based on the labeled data generated from the labeling notebook. The trained model is saved as `asl_cnn_model.h5` for later use in detection.

### 3. Detection Notebook (`detect_sign.ipynb`)
This notebook uses the trained model to detect ASL signs in real-time using a webcam. It leverages the MediaPipe library for hand landmark detection and the trained CNN model to predict the sign being performed. The predictions are displayed on the video feed from the webcam.

## Usage Instructions

1. **Labeling:**
   - Open `labeling.ipynb` and run the cells to generate the `label_dict.pickle` file.

2. **Training:**
   - Open `training.ipynb` and run the cells to train the CNN model on your dataset of ASL signs. After training, ensure that the `asl_cnn_model.h5` file is created in the same directory.

3. **Detection:**
   - Open `detect_sign.ipynb` and run the cells to start the webcam feed for real-time ASL sign detection. Ensure your webcam is properly connected and permissions are granted for camera access.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework.
- [OpenCV](https://opencv.org/) for image processing capabilities.
- [MediaPipe](https://google.github.io/mediapipe/) for hand landmark detection.
