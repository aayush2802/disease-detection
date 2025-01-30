from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import gc  # Garbage collection to free memory

app = Flask(__name__)

# Directories and configurations
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'Plant Disease Detection.h5')  # Replace with the correct model path
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model (consider quantization or smaller model if necessary)
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (updated according to your classes)
CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Remedies for each class
REMEDIES = {
    'Apple___Apple_scab': "Use fungicides containing captan or myclobutanil. Practice proper pruning and dispose of fallen leaves to reduce fungal spores.",
    'Apple___Black_rot': "Prune infected branches and remove fallen fruit. Use fungicides such as thiophanate-methyl or captan.",
    'Apple___Cedar_apple_rust': "Remove nearby juniper plants to break the disease cycle. Use fungicides like myclobutanil during early spring.",
    'Apple___healthy': "No issues detected. Ensure regular care, proper watering, and pest management.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use resistant varieties and apply fungicides like azoxystrobin. Rotate crops to minimize disease spread.",
    'Corn_(maize)___Common_rust_': "Plant resistant varieties and use fungicides such as mancozeb or propiconazole.",
    'Corn_(maize)___Northern_Leaf_Blight': "Plant resistant hybrids and apply fungicides during early disease stages.",
    'Corn_(maize)___healthy': "No issues detected. Continue proper crop care and pest management.",
    'Grape___Black_rot': "Apply fungicides like mancozeb or myclobutanil. Remove infected fruit and leaves promptly.",
    'Grape___Esca_(Black_Measles)': "Prune and remove infected vines. Minimize stress by providing proper irrigation and nutrients.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use fungicides containing captan or mancozeb. Prune infected areas and increase spacing for airflow.",
    'Grape___healthy': "No issues detected. Regularly prune and monitor for pests or diseases.",
    'Potato___Early_blight': "Apply fungicides like chlorothalonil or mancozeb. Rotate crops to reduce disease persistence.",
    'Potato___Late_blight': "Use fungicides such as mancozeb or chlorothalonil. Remove and destroy infected plants.",
    'Potato___healthy': "No issues detected. Continue regular watering and pest control.",
    'Tomato___Bacterial_spot': "Apply copper fungicides and avoid overhead watering. Remove infected plants promptly.",
    'Tomato___Early_blight': "Use fungicides like chlorothalonil and rotate crops annually.",
    'Tomato___Late_blight': "Apply fungicides and ensure proper plant spacing to improve airflow.",
    'Tomato___Leaf_Mold': "Increase ventilation and use fungicides like mancozeb or chlorothalonil.",
    'Tomato___Septoria_leaf_spot': "Remove infected leaves and apply fungicides early in the growing season.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Spray with insecticidal soap or neem oil. Increase humidity around plants.",
    'Tomato___Target_Spot': "Apply fungicides containing mancozeb or chlorothalonil. Remove infected leaves.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control whiteflies with insecticides or traps. Remove infected plants.",
    'Tomato___Tomato_mosaic_virus': "Avoid handling plants when wet and use resistant varieties.",
    'Tomato___healthy': "No issues detected. Provide consistent watering and fertilization."
}

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image for prediction
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Resize to 224x224 for MobileNetV2
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def home():
    return "Flask backend is running!"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        image = preprocess_image(filepath)

        # Make prediction
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_index]
        remedy = REMEDIES.get(predicted_class, "No remedy available for this class.")

        # Clean up: delete the uploaded file to free up memory
        os.remove(filepath)
        gc.collect()  # Manually trigger garbage collection to free up memory

        return jsonify({'prediction': predicted_class, 'remedy': remedy})

    return jsonify({'error': 'Invalid file type'}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
