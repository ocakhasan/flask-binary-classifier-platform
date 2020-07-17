from flask import Flask, render_template, request, redirect, flash, url_for
from main import get_model
import urllib.request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

X = []
y = []

UPLOAD_FOLDER = "uploads"


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = get_model()

global graph
graph = tf.get_default_graph()


def add_data(filename, label):
    image = load_img('uploads/'+filename, target_size=(150, 150))
    image = img_to_array(image)
    image = preprocess_input(image)

    X.append(image)
    y.append(label)


def handle_data(X):
    X = np.array(X).reshape(-1, 150, 150, 3)


def getPrediction(model, filename):

    image = load_img('uploads/'+filename, target_size=(150, 150))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    with graph.as_default():
        yhat = model.predict(image)

    if yhat < 0.5:
        return 0
    else:
        return 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        """if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)"""

        files = request.files.getlist('file')

        print("files is {}".format(files))
        """if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)"""

        if files and ('predict' in request.form):
            for file in files:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                label = getPrediction(model, filename)
                flash(label)
                flash(filename)
            return redirect('/')

        elif files and ('data1' in request.form):
            for file in files:
                print("File {} is uploading".format(file))
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                add_data(filename, 0)
            return redirect('/')

        elif files and ('data2' in request.form):
            for file in files:
                print("File {} is uploading".format(file))
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                add_data(filename, 1)
            return redirect('/')


@app.route('/train', methods=['POST'])
def train():
    with graph.as_default():
        handle_data(X)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        model.fit(np.array(X), np.array(y), nb_epoch=5, batch_size=1)
        return redirect('/')


if __name__ == "__main__":

    app.run()
