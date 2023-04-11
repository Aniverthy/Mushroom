from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

model = load_model('MobileNetV2.h5')

app = Flask(__name__)
def predict(image):
    img = Image.open(image)     #reading the image
    img = img.resize((180,180))     #resizing the image
    img = np.array(img)     #converting the image to numpy array
    img = np.expand_dims(img, axis=0)   #expanding the dimension
    img = img[:,:,:,0:3]     #converting 4 channel to 3 channel
    prediction = class_names [np.argmax(model.predict(img))]    #predicting the image
    return prediction


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    image = request.form['myimage']
    pred = predict(image)
    img = Image.open(image) 
    data = io.BytesIO()
    img.save(data, "JPEG")
    encoded_img = base64.b64encode(data.getvalue())
    decoded_img = encoded_img.decode('utf-8')
    img_data = f"data:image/jpeg;base64,{decoded_img}"
    return render_template('afterhome.html', data=[prediction,img_data])
    #return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)