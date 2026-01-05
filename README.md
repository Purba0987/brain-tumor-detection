# TumorAI – AI Brain Tumor Detection

## Project Overview
TumorAI is an AI-powered brain tumor detection system. It allows users to upload MRI scans and predicts the presence and type of brain tumor. Automatically highlights tumor region. The system also visualizes results, generates reports, and provides performance metrics for model evaluation.


---

## 
- **Brain Tumor Detection** using Convolutional Neural Networks (CNN) with VGG16 base
- **Image Augmentation**:
  - Random brightness and contrast adjustments
- **Data Visualization**:
  - Display random sample images with labels
  - Training history plots (accuracy and loss)
  - Confusion matrix visualization
  - ROC curve with AUC scores
- **Prediction Interface**:
  - Displays uploaded MRI image
  - Shows predicted tumor type and confidence score
- **Model Training** with batching and data generators
- **Multi-class Classification**:
  - Classes: `notumor`, `pituitary`, `meningioma`, `glioma`
- **Performance Metrics**:
  - Classification report
  - Confusion matrix
  - ROC curve for each class

---

## Tools and Technologies Used
- **Python 3.x**
- **Machine Learning / Deep Learning**:
  - TensorFlow / Keras
  - VGG16 pre-trained model (transfer learning)
- **Data Handling & Processing**:
  - NumPy
  - os, random
  - scikit-learn (shuffle, label encoding, classification metrics)
- **Image Processing & Augmentation**:
  - Pillow (PIL)
  - ImageEnhance (brightness, contrast)
  - keras.preprocessing.image (load_img, img_to_array)
- **Data Visualization**:
  - Matplotlib
  - Seaborn
- **Web / GUI (future)**:
  - Flask / Django (for web deployment)
  - PDF report generation (reportlab or fpdf)

---

## Features
- ⚡ **Instant MRI Analysis** – Results in seconds.
- 🛡 **High Accuracy Detection** – Deep learning-based tumor identification.
- 🧪 **Multi-Class Classification** – Detects Glioma, Meningioma & Pituitary tumors.
- 🟥 **MRI Image Annotation** – Automatically highlights tumor region.
- 📊 **Detailed Reports** – Professional medical report generation.
- 🔐 **Secure Upload** – Encrypted MRI data protection.
- 📱 **Multi-Device** – Works on all screen sizes.
- 🤖 **Chatbot & Voice Support** – Interactive AI explanations.
- **Data Augmentation** – Random brightness and contrast adjustments for better model generalization.
- **Visualization** – Training history, confusion matrix, ROC curves.
- **Prediction Interface** – Displays uploaded MRI image with predicted tumor type and confidence.



Watch demo video : https://youtu.be/0bDApW3KF_I?si=_8wWmLj-e4X8aGAW
