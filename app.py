# Install required packages
# pip install Flask tensorflow pillow

# Import necessary libraries
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Create a Flask app
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
# Define a route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to handle image downloads
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# Load your trained model (replace 'your_model.h5' with your actual model file)
model = tf.keras.models.load_model('Functional_new20240101083121.h5')

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the main route for rendering the upload form
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about_me')
def about_me():
    return render_template('about_me.html')

@app.route('/predict')
def predict():
   
    images = [
        {'filename': 'osteo-1.jpg', 'description': 'Osteo 1'},
        {'filename': 'osteo-2.jpeg', 'description': 'Osteo 2'},
        {'filename': 'osteo-3.jpg', 'description': 'Osteo 3'},
        {'filename': 'normal-1.jpg', 'description': 'Normal 1'},
        {'filename': 'normal-2.jpeg', 'description': 'Normal 2'},
        {'filename': 'normal-3.png', 'description': 'Normal 3'}
    ]
    return render_template('predict.html', images=images)

# Define the route for handling image uploads and making predictions
@app.route('/prediction', methods=['POST'])
def prediction():
    if 'files' not in request.files:
        return render_template('prediction.html', error='No file part')

    files = request.files.getlist('files')

    results = []

    for file in files:
        if file.filename == '':
            continue

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = 'uploads/' + filename
            file.save(file_path)

            # Preprocess the image (adjust according to your model's input size)
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Make predictions
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            class_labels = ['normal', 'osteoporosis']
            class_name = class_labels[class_index]  # Replace with your actual class names
            accuracy = predictions[0, class_index]

            results.append({
                'filename': filename,
                'class_name': class_name,
                'accuracy': accuracy,
            })

        else:
            return render_template('prediction.html', error='Invalid file extension')

    return render_template('prediction.html', results=results)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
