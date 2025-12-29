# ForgeryLens: Digital Image Forgery Detection

> **Detecting Digital Image Forgery using Error Level Analysis (ELA) and Transfer Learning with Lightweight CNN Architectures.**

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sarthakdey/visual-computing-25ai60r20-25ai60r11)

## 📌 Project Overview

**ForgeryLens** is a deep learning-based forensic tool designed to detect digital image manipulation (such as splicing and copy-move forgery). With the widespread availability of powerful image editing software, ensuring the authenticity of digital media has become critical.

This project combines **Error Level Analysis (ELA)** to highlight compression artifacts with a suite of **Convolutional Neural Networks (CNNs)** to classify images as "Authentic" or "Tampered". A key focus of this work is evaluating lightweight architectures (like MobileNetV2) for future deployment on low-power mobile and embedded devices.

📄 **Detailed Analysis:** For a comprehensive technical breakdown, methodology, and experimental data, please refer to the [Project Report (PDF)](Visual_Computing_project.pdf) included in this repository.

## 🚀 Key Features

* **Error Level Analysis (ELA):** Preprocessing technique to reveal compression anomalies invisible to the naked eye.
* **Transfer Learning:** Utilizes pre-trained state-of-the-art CNN architectures for robust feature extraction.
* **Lightweight Focus:** Specifically optimized for efficiency to enable edge/mobile deployment.
* **Balanced Dataset:** Trained on a balanced subset of the CASIA 2.0 Image Tampering Detection Dataset.

## 🧠 Methodology

The detection pipeline consists of three main stages:

1.  **Data Preparation:**
    * **Source:** CASIA 2.0 Image Tampering Detection Dataset.
    * **Distribution:** Balanced set of ~2,100 Authentic images and ~2,064 Tampered images.
2.  **ELA Preprocessing:** 
    * Input images are re-compressed at a known quality (90%).
    * The difference between the original and re-compressed image is calculated.
    * Differences are amplified to create an "ELA Map" that highlights spliced regions (which often exhibit different compression histories).
3.  **Classification (CNNs):** 

[Image of cnn architecture diagram]

    * The ELA maps are fed into CNN models to learn the distinction between authentic and forged patterns.
    * **Models Evaluated:**
        * ResNet18, ResNet34, ResNet50.
        * **MobileNetV2 (Best Trade-off)**.
        * EfficientNet-Lite0.

## 🛠️ Tech Stack

* **Language:** Python 3.7+
* **Deep Learning Framework:** TensorFlow / Keras.
* **Image Processing:** PIL (Pillow), OpenCV (cv2)
* **Data Manipulation:** NumPy, Scikit-learn
* **Environment:** Kaggle Kernels.

## 📊 Results

The project evaluated models based on accuracy, model size, FLOPs (floating-point operations), and inference latency.

* **Best Performer for Deployment:** **MobileNetV2**
    * It achieved the optimal balance between high detection accuracy and computational efficiency, making it the most suitable candidate for mobile or edge-based systems.

## 💻 Usage & Demo

You do not need to install the environment locally to test the project. The full training pipeline and inference code are hosted on Kaggle.

### 🚀 Run on Kaggle
Click the link below to access the notebook, where you can run the code using free GPU resources:

**[🔗 View Project Notebook on Kaggle](https://www.kaggle.com/code/sarthakdey/visual-computing-25ai60r20-25ai60r11)**

*Note: The dataset is automatically handled within the Kaggle environment.*

### Code Snippet: Error Level Analysis
Here is the core function used to generate ELA maps:

```python
from PIL import Image, ImageChops, ImageEnhance

def convert_to_ela_image(path, quality=90):
    image = Image.open(path).convert('RGB')
    
    # Save as temporary image at specific quality
    image.save('temp.jpg', 'JPEG', quality=quality)
    temp_image = Image.open('temp.jpg')
    
    # Calculate pixel difference
    ela_image = ImageChops.difference(image, temp_image)
    
    # Amplify the difference
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    
    return ImageEnhance.Brightness(ela_image).enhance(scale)
