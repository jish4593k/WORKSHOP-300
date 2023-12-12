import cv2
import os
import numpy as np
import torch
import tensorflow as tf
from ExtrinsicCalibration import ExCalibrator
from IntrinsicCalibration import InCalibrator, CalibMode
from SurroundBirdEyeView import BevGenerator
import tkinter as tk
from tkinter import filedialog

def runInCalib_3():
    print("Intrinsic Calibration with PyTorch ......")
  
    img_path = './IntrinsicCalibration/data/img_raw0.jpg'
    img = cv2.imread(img_path)
    img_tensor = torch.from_numpy(img)

    cv2.imshow("Processed Image (PyTorch)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def runNNInference():
    print("Neural Network Inference with Keras (TensorFlow) ......")
    # Use TensorFlow (Keras) for neural network inference
    model = tf.keras.models.load_model('your_keras_model.h5')

   
    img_path = ''
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224)) 
    img = np.expand_dims(img, axis=0) 
    img = img / 255.0  

    prediction = model.predict(img)

    
    print("Prediction:", prediction)

def runGUI():
    print("Creating GUI with Tkinter ......")
    root = tk.Tk()
    root.title("Computer Vision App")


    root.mainloop()

def main():
    runInCalib_3()
    runNNInference()
    runGUI()

if __name__ == '__main__':
    main()
