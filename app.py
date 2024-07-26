from flask import Flask, render_template, Response
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

app = Flask(__name__)

model=YOLO(r'yolov8s.pt')  #path to model

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap=cv2.VideoCapture(0)
    my_file=open(r"coco.txt") #path to coco file
    data=my_file.read()
    class_list=data.split("\n")

    count=0
    tracker=Tracker()

    while True:
        ret,frame=cap.read()
        if not ret:
            break
        count+=1
        if count%3!=0:
            continue
        
        frame=cv2.resize(frame,(1020,500))

        results=model.predict(frame)
        a=results[0].boxes.data
        px=pd.DataFrame(a).astype("float")
        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            c=class_list[d]
            if 'person' in c:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)
                cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

        _,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
