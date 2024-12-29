import streamlit as st
from utils import load_pickle, extract_features, labels
from camera_input_live import camera_input_live
import numpy as np
import cv2
from streamlit import session_state as state

st.title("Machine Learning")

svm_pickle = load_pickle('svm_model.pkl')
# rf_pickle = load_pickle('rf_model.pkl')

selection = st.selectbox("Choose model", [
    "K-Nearest Neighbour",
    "Support Vector Machine",
    "Random Forest"
])

image = camera_input_live(200)

if image:
  image = np.asarray(bytearray(image.read()), dtype="uint8") 
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (512, 512))
  features = extract_features(image)
  # st.write(features)
  # st.write(selection)
  model = state['models'][selection]
  pred = model.predict([features])
  st.write(labels[int(pred)])
  st.image(image)
