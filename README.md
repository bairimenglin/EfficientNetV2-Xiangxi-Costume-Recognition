# EfficientNetV2 Xiangxi Minority Costume Recognition

Final project for EECE5644 Machine Learning class (Fall 2025).  
This project implements an image classification system for recognizing traditional Xiangxi minority costumes using EfficientNetV2 and PyTorch, with a PyQt5 graphical user interface for real-time prediction.

---

##  Installation

Make sure Python is installed.

Install all required packages:

```python
pip install -r requirements.txt
```

 How to Run the Program

Run the PyQt5 GUI:
```python
python gui.py
```
The dataset was split into 80% for training and 20% for testing. Upload last 20% image for prediction from data.zip

This will open the GUI window.
You can upload an image and get the predicted ethnic group.

Files in This Project：
```python
gui.py              - PyQt5 GUI interface
predict.py          - Loads trained model and performs predictions
model.py            - EfficientNetV2 model definition
train.py            - Training code (optional)
utils.py            - Helper functions
calculate_m_s.py    - Evaluation helper
class_indices.json  - Class index–to–name mapping
requirements.txt    - Python dependencies
```
GUI example
![GUI Example](gui%20example.png)

