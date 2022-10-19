from cProfile import run
from tensorflow.python.keras.models import load_model
from keras.utils import image_utils
import numpy as np
from flask import Flask, request, render_template,Response
import cv2
from cvzone.ClassificationModule import Classifier
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os


UPLOAD_FOLDER = 'D:/Pycharm/final/static/uploads'
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])


app = Flask(__name__)


model_path = "./soil.h5"

SoilNet = load_model(model_path)

classes = {0:"Black Soil:-{ Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute }",1:"Clay Soil:-{ Virginia, Wheat , Jowar,Millets,Linseed,Castor,Sunflower} ",2:"Loam Soil:-{ Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans }",3:"Red Soil:{ Cotton,Wheat,Pilses,Millets,OilSeeds,Potatoes }"}

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/ripe',methods=['GET'])
def ripe():
    return render_template('ripe.html')

@app.route('/soil',methods=['GET'])
def soil():
    return render_template('soil.html')



def ripe():
    cap = cv2.VideoCapture(0)
    myClassifier = Classifier('./keras_model.h5', './labels.txt')
    while True:
        # _, img = cap.read()
        success,frame=cap.read()
        predictions = myClassifier.getPrediction(frame)
        # cv2.imshow("Image",frame)
        # cv2.waitKey(1)
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        if cv2.waitKey(1) & 0xFF == ord('a'):
            cap.release()
            cv2.destroyAllWindows() 
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    

@app.route('/video',methods=['GET'])
def video():
    return Response(ripe(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/count', methods=['GET', 'POST'])
def count():
    output_count = ""
    imgs=""
    if request.method == 'POST':
        f = request.files['image']
        file_path = "./static/uploads/"+f.filename
        print(file_path)
        img = cv2.imread(file_path)
        img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        box,label,count = cv.detect_common_objects(img)
        output = draw_bbox(img,box,label,count)
        output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
        output_count = str(len(label))
        imgs = "./static/uploads/"+f.filename
    return render_template('count.html',count = output_count,image_path=imgs)

def model_predict(image_path,model):
    print("Predicted")
    image = image_utils.load_img(image_path,target_size=(100,100))
    image = image_utils.img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    prediction = classes[result]
    
    
    if result == 0:
        print("black.html")
        
        return "black","black.html"
    elif result == 1:
        print("Clay.html")
        
        return "Clay", "clay.html"
    elif result == 2:
        print("Loam.html")
        
        return "Loam" , "loam.html"
    elif result == 3:
        print("Red.html")
        
        return "Red" , "red.html"


@app.route('/predict',methods=['GET','POST'])
def predict():
    print("Entered")
    if request.method == 'POST':
        print("Entered here")
        file = request.files['image'] 
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('./static/uploads', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = model_predict(file_path,SoilNet)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)           

if __name__ == '__main__':
    app.run(debug=True,threaded=False)