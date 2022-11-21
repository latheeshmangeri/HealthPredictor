import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

global graph
graph = tf.compat.v1.get_default_graph()
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
try:
    os.remove('D:\zBin\Py\Health\static\img.png')
except:
    pass
app = Flask(__name__, template_folder='template')
model1 = load_model("models\Malaria_model111.h5")
model2 = load_model("models\Pneumonia_Prediction_model.h5")
model3 = load_model("models\Breast_model.h5")
model4 = load_model("models\Brain_Tumor_VGG_model.h5")
model5 = load_model("models\TB_model.h5")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/malaria', methods=['GET', 'POST'])
def malaria():
    return render_template('malaria.html')

@app.route('/predict_malaria', methods=['GET', 'POST'])
def predict_malaria():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, f.filename)
        print("filepath is ", filepath)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(50,50,3))
        img.save('D:\\zBin\\Py\\Health\\static\\img.png')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        output = model1.predict(x)
        print("prediction", output)
        per = output[0][0] if output[0][0] > output[0][1] else output[0][1]
        res_val = "INFECTED" if output[0][1] == 0 else "UNINFECTED"

        return render_template('malaria.html', imgfile='img.png', prediction_text='Result : {} {}%'.format(res_val,per*100))
    return None

@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/predict_pneumonia', methods=['GET', 'POST'])
def predict_pneumonia():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(64, 64))
        img.save('D:\\zBin\\Py\\Health\\static\\img.png')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        #with graph.as_default():
        output = model2.predict_classes(x)

        result = model2.predict(x)
        print(result)
        result = float("{:.2f}".format(float(result[0][0])))
        print("prediction", output)

        res_val = "Normal" if result == 0 else "Pneumonia"

        return render_template('pneumonia.html', imgfile='img.png', prediction_text='Result : {} {}%'.format(res_val,result*100))
    return None
@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    return render_template('breast_cancer.html')

@app.route('/predict_breast_cancer', methods=['GET', 'POST'])
def predict_breast_cancer():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, f.filename)
        print("filepath is ", filepath)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(50,50))
        img.save('D:\\zBin\\Py\\Health\\static\\img.png')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        result = model3.predict(x)
        print("prediction", result)
        cell = result[0][0] if result[0][0] > result[0][1] else result[0][1]
        cell = float("{:.2f}".format(float(cell)))
        res_val = "IDC(+ve)" if result[0][1] < result[0][0] else "IDC(-ve)"

        return render_template('breast_cancer.html', imgfile='img.png', prediction_text='Result : {} {}%'.format(res_val, cell*100))
    return None
@app.route('/brain_tumor', methods=['GET', 'POST'])
def brain_tumor():
    return render_template('brain_tumor.html')

@app.route('/predict_brain_tumor', methods=['GET', 'POST'])
def predict_brain_tumor():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, f.filename)
        print("filepath is ", filepath)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        img.save('D:\\zBin\\Py\\Health\\static\\img.png')
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)
        result = model4.predict_classes(test_image)
        print(result)
        result = model4.predict(test_image)
        print(result)
        result = float("{:.2f}".format(float(result[0][0])))
        print(result)
        output = "Tumor Detected" if result > 0 else "No Tumor Detected"

        return render_template('brain_tumor.html', imgfile='img.png', prediction_text='Result : {}'.format(output))
    return None
@app.route('/tuberculosis', methods=['GET', 'POST'])
def tuberculosis():
    return render_template('tuberculosis.html')

@app.route('/predict_tuberculosis', methods=['GET', 'POST'])
def predict_tuberculosis():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath, f.filename)
        print("filepath is ", filepath)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(96,96))
        img.save('D:\\zBin\\Py\\Health\\static\\img.png')
        test_image = image.img_to_array(img)
        test_image = np.expand_dims(test_image, axis=0)
        result = model5.predict(test_image)
        print(result)
        per = result[0][0] if result[0][0] > result[0][1] else result[0][1]
        cell = "TB(+ve)" if result[0][0] == 0 else "TB(-ve)"
        print(cell, per * 100, "%")

        return render_template('tuberculosis.html', imgfile='img.png', prediction_text='Result : {}'.format(cell))
    return None

if __name__ == '__main__':
    app.run(debug=True, threaded=False)