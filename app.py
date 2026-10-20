from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model 
model = load_model("melanoma_model_v4.h5")

#Random tips
tips = [
    "استخدم واقي الشمس يوميًا لحماية بشرتك.",
    "افحص بشرتك شهريًا لاكتشاف أي تغييرات مبكرًا.",
    "تجنب التعرض لأشعة الشمس في أوقات الذروة.",
    "ارتدِ ملابس واقية عند الخروج في الشمس.",
    "تناول غذاءً صحيًا لدعم صحة بشرتك.",
    "تجنب أجهزة التسمير الصناعية (Tanning Beds)."
]

# Image processing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction
def predict_result(img_path):
    processed_image = preprocess_image(img_path)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction, axis=1)[0]
    class_labels = ['melanoma', 'not melanoma']  
    result = class_labels[class_index]
    confidence = float(np.max(prediction)) * 100
    return result, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    random_tip = random.choice(tips)  
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result, confidence = predict_result(filepath)

            return render_template('index.html', prediction=result, confidence=round(confidence, 2), image_path=filepath, tip=random_tip)

    return render_template('index.html', tip=random_tip)

if __name__ == '__main__':
    app.run(debug=True)

