import flask
from flask import Flask, render_template, url_for
# from flask import render_template
from flask import request
from flask import jsonify
from flask import send_from_directory
from flask import redirect
from flask import flash
import os.path

# import pandas as pd
# import time
# import pandas as pd
# import numpy as np
import datetime
import json

import numpy as np

from PIL import Image
import requests
from io import BytesIO

import pandas as pd
import numpy as np
import json
import time
import os

# Tensorflow imports
import tensorflow as tf

from .databaseconfig import user_name, password, local_host



# **** Place functions here from predict.py to avoid Heroku Error: **** 
# from predict import process_image, prediction, get_label_names
def process_image(image):
    # from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    '''
    The process_image function should take in an image (in the form of a NumPy array) and return an image in the form of a NumPy array with shape (224, 224, 3).
    '''
    # First, convert the image into a TensorFlow Tensor then resize it to the appropriate size using tf.image.resize
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224,224))
    # #Second, the pixel values of the input images are typically encoded as integers in the range 0-255, but the model expects the pixel values to be floats in the range 0-1. 
    # # #Therefore, you'll also need to normalize the pixel values
    image /= 255
    image = image.numpy()
    return image
    
def prediction(image_path,model,top_k):
    import os
    # model_path = os.path.join(app.root_path)
    model = 'my_model2'
    model_path = os.path.join(app.root_path, model)

    reloaded_keras_model = tf.saved_model.load(model_path)

    processed_test_image = process_image(image_path)

    #The image returned by the process_image function is a NumPy array with shape (224, 224, 3) but the model expects the input images to be of shape (1, 224, 224, 3). This extra dimension represents the batch size.
    # Use  np.expand_dims() function to add the extra dimension.
    processed_test_image = np.expand_dims(processed_test_image, axis=0)
    
    # Get the predictions by using the probability model to predict the input image 
    # predictions = probability_model.predict(processed_test_image)
    predictions = reloaded_keras_model(processed_test_image)
    predictions = np.array(predictions)
    # Get the index of the top 10 probabilities
    top_idxs = predictions[0].argsort()[-top_k:][::-1]
    
    # Get the top 10 probabilities
    top_probabilities = predictions[0][top_idxs]
    probs = top_probabilities
    
    # Get the labels (the index of the probabilities)
    labels_nums = [str(idx) for idx in top_idxs]
    classes = labels_nums
    return probs, classes


def get_label_names(json_file, labels):
    '''
    Given json_file that contains the label names for the label numbers, return the correct label names from array 'label'
    '''
    class_names = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}
    new_class_names = {}
    for key in class_names:
    #     print(key)
        new_class_names[int(key)-1]=class_names[key]
    new_class_names
    label_names = [new_class_names[int(i)] for i in labels]
    return label_names

# **** End functions from predict.py  **** 


from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'tmp/uploads'
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS = {'jpg','jpeg'}


app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# t_host =  f'postgresql://{user_name}:{password}@{local_host}/flower-image-db'
# from flask_sqlalchemy import SQLAlchemy
# t_host =  f'postgresql://{user_name}:{password}@{local_host}/flower-image-db'

# # app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '')
# app.config['SQLALCHEMY_DATABASE_URI'] =t_host 
# # # Remove tracking modifications
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# # 
# db = SQLAlchemy(app)

# class ImageDB(db.Model):
#     __tablename__ = 'tbl_files_images'

#     id_image = db.Column(db.Integer, primary_key=True)
#     blob_image_data = db.Column(db.LargeBinary)

#     def __repr__(self):
#         return '<ImageDB %r>' % (self.name)

        
import psycopg2 


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        print(request.url_root)
        return flask.render_template('index.html')
    if request.method == 'POST':
        print(request.url_root)
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            # return redirect(request.url)
            return flask.render_template('index.html', 
                                    noFile = True)
        file = request.files['file']
        # id_image = 1
        # SaveFileToPG(id_image, file)
        # print('my FIle')
        # print(file)
        my_file = request.files['file'].read()
        # print('my FIle')
        # print(my_file)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # flash('No selected file')
            # return redirect(request.url)
            return flask.render_template('index.html', 
                                    noSelectedFile = True)
        if allowed_file(file.filename) == False:
            return flask.render_template('index.html', 
                                    noFile = True)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
 

            im = Image.open(BytesIO(my_file))

            test_image = np.asarray(im)
                

            print(filename)

            top_k = str(request.form['month_day'])
            if top_k.isdigit() == False:
                return flask.render_template('index.html', 
                                    topKError = True)
            else:

                temp = 'temp'

                top_k = int(top_k)
                

                probs, classes = prediction(test_image, 'my_model2', top_k)


                # # # # Print the probabilities and class numbers to the console
                print(f'Proabilities: {probs}')
                print(probs[0])
                print(f'Label Numbers: {classes}')
                # # Call the get_label_names function with the JSON file with class names and the classes returned from prediction
                label_names = get_label_names('label_map.json', classes)
                # # Print the class names to the console
                print(f'Label Names: {label_names}')
                print(probs[0])
                print(classes[0])
                print(label_names[0])
                top_flower_name = label_names[0].title()
                top_prob = round(probs[0]*100,2)

                if top_k == 1:
                    return flask.render_template('index.html', 
                                            probs = probs,
                                            classes=classes,
                                            labelNames=label_names,
                                            topFlowerName = top_flower_name,
                                            onlyOneK = True,
                                            topProb = top_prob)
                else:
                    temp = 'temp'
                    return flask.render_template('index.html', 
                                            probs = probs,
                                            classes=classes,
                                            labelNames=label_names,
                                            temp=temp,
                                            topFlowerName = top_flower_name,
                                            topProb = top_prob,
                                            filename=filename,
                                            top_k = top_k)


def SaveFileToPG(id_image, fileData):
    insert_stmt = (
        "INSERT INTO tbl_files_images (id_image, blob_image_data) "
        "VALUES (%s, %s)"
        )
    data = (id_image, fileData)
    db_cursor.execute(insert_stmt, data)

            


# @app.route('/uploads/<filename>/<top_k>')
# @app.route('/tmp/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
                               # filename)

# @app.route('/test/<filename>')
# def test(filename):
#     myIMG = url_for('uploaded_file',filename=filename)
#     temp='temp'
#     return flask.render_template('index.html',
#                                     filename=filename,
#                                         myURL = myIMG,
#                                             temp=temp)




if __name__ == "__main__":
    app.run(debug=True) 