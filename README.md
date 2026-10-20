# Smart Scan — Melanoma Detection

## Overview
Smart Scan is a deep learning web application designed to classify skin lesion images as **Melanoma** or **Not Melanoma** using a trained deep learning model.

## Features
- Upload a skin lesion image through a web interface.
- Predict whether the image is classified as melanoma or not melanoma.
- Display the prediction result.
- Provide random skin-health tips in Arabic.

## Technologies Used
- Python
- TensorFlow / Keras
- Flask
- NumPy
- HTML / CSS
- Jupyter Notebook

## Project Structure
```text
Smart-Scan-Melanoma-Detection/
├── app.py
├── melanoma_model_v4.h5
├── model code.ipynb
├── templates/
│   └── index.html
└── README.md
```

## How to Run
1. Install the required Python libraries.
2. Ensure `melanoma_model_v4.h5` is in the project root directory.
3. Run the application:

```bash
python app.py
```

4. Open the local URL shown in the terminal.

## Disclaimer
This project is for educational and research purposes only. Its predictions are not a substitute for professional medical diagnosis.
