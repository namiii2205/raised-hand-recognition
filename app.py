import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title='Hand Raising Recognition', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

# Load YOLO model
yolo_model = YOLO("hand_raising_detection.pt")

class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect objects in the frame
        results = yolo_model(img, conf=0.4, iou=0)
        
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "Raising Hand", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img

st.title("Real-time Person Detection with YOLO and Streamlit")

webrtc_streamer(key="example", video_transformer_factory=YOLOVideoTransformer)
