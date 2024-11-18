from flask import Flask, render_template, request, jsonify
import cv2
import datetime
import os
from ultralytics import YOLO
from folium import Map, Circle, Marker
from folium.plugins import MiniMap
import geopy.distance

app = Flask(__name__)
model = YOLO("yolov8n.pt")


detected_animals_data = []
last_detection_times = {}


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def dashboard():

    return render_template('dashboard.html', animals=detected_animals_data)


@app.route('/map', methods=['GET'])
def render_map():

    lat = float(request.args.get('lat', 0))
    lon = float(request.args.get('lon', 0))

   
    m = Map(location=[lat, lon], zoom_start=100)
    Circle(location=(lat, lon), radius=10, color='red', fill=True).add_to(m)

    MiniMap().add_to(m)


    map_html = m._repr_html_()
    return jsonify({"map_html": map_html})
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
   
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

        
            detections = process_video_live(file_path)
            return jsonify(detections)
    return render_template('upload.html')


WILDLIFE_ANIMALS = ["lion", "tiger", "elephant", "bear", "deer", "fox", "wolf", "cheetah", "leopard", "giraffe"]

def process_video_live(video_path):
    video = cv2.VideoCapture(video_path)
    live_detections = []

    global last_detection_times
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            for box in result.boxes:
                if hasattr(box, "cls"):
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]

                
                    if class_name.lower() in WILDLIFE_ANIMALS:
                        current_time = datetime.datetime.now()

                        if class_name not in last_detection_times or \
                                (current_time - last_detection_times[class_name]).total_seconds() > 60:
                            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                            live_detections.append({"animal": class_name, "time": timestamp})
                            detected_animals_data.append({"animal": class_name, "time": timestamp})
                            last_detection_times[class_name] = current_time

    video.release()
    return live_detections

   


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
