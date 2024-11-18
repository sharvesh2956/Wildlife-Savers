import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from geopy.geocoders import Nominatim
from PIL import Image
import tempfile

model = YOLO('yolov8n.pt') 


st.title("Wild Animal Detection in Video")


tab1, tab2 = st.tabs(["Detection", "Alerts"])

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])


def get_map_iframe():
    geolocator = Nominatim(user_agent="wild_animal_detection_app")
    location = geolocator.geocode("Your City, Your State, Your Country")
    if location:
        latitude, longitude = location.latitude, location.longitude
        map_url = f"https://www.google.com/maps/embed/v1/view?key=YOUR_GOOGLE_MAPS_API_KEY&center={latitude},{longitude}&zoom=15"
        return f'<iframe width="100%" height="400" src="{map_url}" allowfullscreen></iframe>'
    return None


with tab2:
    st.markdown("## Map Location of Alert Area")
    map_iframe = get_map_iframe()
    if map_iframe:
        st.markdown(map_iframe, unsafe_allow_html=True)


if tab1.button("Show Location on Map", key="map_button_1"):
    if map_iframe:
        st.markdown(map_iframe, unsafe_allow_html=True)


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name


    video = cv2.VideoCapture(temp_video_path)

    detected_animals = set()
    stframe_detection = tab1.empty() 
    stframe_alerts = tab2.empty()     

    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

      
        results = model(frame)
        frame_detections = set()

      
        frame_alert = frame.copy()

    
        for result in results:
            for box in result.boxes:
               
                if hasattr(box, 'xyxy') and hasattr(box, 'conf') and hasattr(box, 'cls'):
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]
                    frame_detections.add(class_name)

   
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.rectangle(frame_alert, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_alert, f"ALERT: {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        detected_animals.update(frame_detections)

   
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_alert_rgb = cv2.cvtColor(frame_alert, cv2.COLOR_BGR2RGB)

        stframe_detection.image(frame_rgb, caption=f"Detected Animals: {', '.join(frame_detections) if frame_detections else 'None'}", use_column_width=True)
        stframe_alerts.image(frame_alert_rgb, caption="Alert - Red Boxes Indicate Animal Presence", use_column_width=True)

    video.release()

    st.write("All Detected Animals:", ", ".join(detected_animals) if detected_animals else "None")
