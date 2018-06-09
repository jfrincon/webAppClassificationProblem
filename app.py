import cv2
import numpy as np

from sklearn.cluster import KMeans
from sklearn.externals import joblib
import pickle

from sklearn import svm

from flask import Flask, render_template


app = Flask(__name__)
@app.route("/")

def index():
        kmeans_path = 'models/kmeans_model2'
        kmeans = pickle.load(open(kmeans_path, 'rb'))
        clusterNum = 50

        svm_path = 'models/SVM_Model2'
        svc = pickle.load(open(svm_path, 'rb'))
                                                                                                                                                        

        names = ['19','auditorio','18','38','biblioteca','26','idiomas','dogger','agora','admisiones']

        img_path = 'static/photo.jpg'

        img = cv2.imread(img_path)
        img = cv2.resize(img, (250,250))
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv2.drawKeypoints(gray,kp,img)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)

        histogram = [0]*clusterNum
        for elem in des:
                histogram[kmeans.predict([elem])[0]]+= 1
        maxElem = max(histogram)

        for elem in range(len(histogram)):
                histogram[elem] = histogram[elem]/maxElem

        prediction = svc.predict([histogram])[0]
        htmlStr = "<h1>"+str(names[int(prediction)-1])+"</h1>" + "\n" + "<img src=\"static/photo.jpg\">"
#        return (names[int(prediction)-1])




        return htmlStr

if __name__ == "__main__":
        app.run(host='0.0.0.0', debug=True)
