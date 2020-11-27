from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

# Create Flask
app = Flask(__name__)

# Set max file size for the images as 10mb
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Only allow png, jpg, and jpeg files
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

def allowed_file_types(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Set global default
def init():
    global graph
    graph = tf.compat.v1.get_default_graph()

# Loading and processing the image input
def read_image(filename):
    # Load image
    img = load_img(filename, target_size = (64, 64))

    # Convert to array
    img = img_to_array(img)

    # Reshape for the model
    img = img.reshape(1, 64, 64, 3)

    # Rescale he image
    img = img.astype('float32')
    img = img / 255.

    return img

# Setting the homepage
@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('home.html')

# Predict the class of the image
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file_types(file.filename):
                filename = file.filename
                file_path = os.path.join('static/images', filename)
                file.save(file_path)
                img = read_image(file_path)

                # Predict the class of an image
                with graph.as_default():
                    model1 = load_model('updated_tuned_nn.h5')
                    class_prediction = model1.predict_classes(img)
                    #product = class_prediction
                    print(class_prediction)

                #Map apparel category with the numerical class
                if class_prediction[0] == 0:
                  product = "Deepfake Image"
                elif class_prediction[0] == 1:
                  product = "Real Image"

                return render_template('predict.html', product = product, user_image = file_path)
        except Exception:
            return "Unable to read the file.  Please make sure it is the correct file type."
    
    return render_template('predict.html')

if __name__ == '__main__':
    init()
    app.run(debug = True)