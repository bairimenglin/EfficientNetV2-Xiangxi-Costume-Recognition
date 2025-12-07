# EfficientNetV2 Xiangxi Minority Costume Recognition

Final project for EECE5644 Machine Learning class (Fall 2025).  
This project implements a image classification system for recognizing traditional Xiangxi minority costumes using EfficientNetV2 and PyTorch, with a PyQt5 graphical user interface for real-time prediction.

---

## 📦 Installation

Make sure Python is installed.

Install all required packages:

```bash
pip install -r requirements.txt

🚀 How to Run the Program

Run the PyQt5 GUI:
```bash

python gui.py

This will open the GUI window.
You can upload an image and get the predicted ethnic group.

Files in This Project
```bash
gui.py              → The PyQt5 GUI interface
predict.py          → Loads the trained model and performs predictions
model.py            → EfficientNetV2 model structure
train.py            → Training code (optional)
utils.py            → Helper functions
calculate_m_s.py    → Evaluation helper script
class_indices.json  → Mapping from class index to class name
requirements.txt    → Python dependencies

Details
```bash
Model: EfficientNetV2-S
Framework: PyTorch
Task: Fine-grained, multi-class classification
Input: 224×224 RGB images
Output: 5 Xiangxi minority costume categories
