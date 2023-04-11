from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

model = load_model('MobileNetV2.h5') #loading the model
class_names=['Blue Oyster Mushroom',
 'Caterpillar Fungus',
 'King Oyster Mushroom',
 "Lion's Mane Mushroom",
 'Pink Oyster mushrooms',
 'Shiitake Mushroom',
 'White Oyster Mushroom']


def predict(image):
    img = Image.open(image)     #reading the image
    img = img.resize((180,180))     #resizing the image
    img = np.array(img)     #converting the image to numpy array
    img = np.expand_dims(img, axis=0)   #expanding the dimension
    img = img[:,:,:,0:3]     #converting 4 channel to 3 channel
    prediction = class_names [np.argmax(model.predict(img))]    #predicting the image
    return prediction


# index route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')    #returning the index.html file

# @app.route('/index1.html', methods=['GET'])
# def index1():
#     return render_template('index1.html') 

@app.route('/home.html', methods=['GET'])
def home():
    return render_template('home.html') 


# predict route
@app.route('/predict', methods=['POST'])
def predictt():
    img = request.files['img']    #getting the image from the user
    #img.save('static\img.jpg')     #saving the image
    prediction, confidence = predict(img)     #predicting the image from previously defined predict function and sending the image as an argument
    img = Image.open(img) 
    data = io.BytesIO()
    img.save(data, "JPEG")
    encoded_img = base64.b64encode(data.getvalue())
    decoded_img = encoded_img.decode('utf-8')
    img_data = f"data:image/jpeg;base64,{decoded_img}"
    return render_template('predict.html', data=[prediction])

"""
# index route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')    #returning the index.html file

# @app.route('/index1.html', methods=['GET'])
# def index1():
#     return render_template('index1.html') 

"""
@app.route('/home.html', methods=['GET'])
def home():
    return render_template('home.html') 


# predict route
@app.route('/predict', methods=['POST'])
def predictt():
    img = request.files['img']    #getting the image from the user
    #img.save('static\img.jpg')     #saving the image
    prediction = predict(img)     #predicting the image from previously defined predict function and sending the image as an argument
    img = Image.open(img) 
    data = io.BytesIO()
    img.save(data, "JPEG")
    encoded_img = base64.b64encode(data.getvalue())
    decoded_img = encoded_img.decode('utf-8')
    img_data = f"data:image/jpeg;base64,{decoded_img}"
    return render_template('predict.html', data=[prediction])

if __name__ == '__main__':
    app.run(debug=True)
