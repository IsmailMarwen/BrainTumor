import numpy as np
from flask import Flask, render_template, request, send_file
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import cv2
import os
from werkzeug.utils import secure_filename
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Define a Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
# Define a route for prediction
@app.route('/', methods=['POST'])
def predict():
    # Load the image from the request
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    img = cv2.imread(filepath)
    img1 = cv2.resize(img, (200, 200))
    img_flat = img1.reshape(-1) / 255.0

# pad the flattened image with zeros to make it have 40000 features
    if len(img_flat) < 40000:
       padded_img = np.pad(img_flat, (0, 40000 - len(img_flat)), 'constant')
    else:
       padded_img = img_flat[:40000]

# reshape the padded image to (1, 40000)
    padded_img = padded_img.reshape(1, -1)
    pred = model.predict(padded_img)
    # Create a plot of the prediction
    fig, ax = plt.subplots()
    if pred == 1:
        ax.imshow(img)
        ax.set_title('Brain tumor detected')
        ax.axis('off')
    else:
         ax.imshow(img)
         ax.set_title('No brain tumor detected')
         ax.axis('off')
    fig.savefig('prediction.png')
    
    # Return the HTML page with the prediction image
    return send_file(f'prediction.png', mimetype='image/png')
    


if __name__ == '__main__':
    app.run()