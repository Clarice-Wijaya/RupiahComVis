import streamlit as st
from streamlit import session_state as state
from utils import load_pickle, load_pt

if 'models' not in state:
  state['models'] = {
    "ResNet50": load_pt("ResNet50"),
    "YoLoV11" : load_pt("YoLoV11"),
    "K-Nearest Neighbour": load_pickle('knn_model.pkl'),
    "Support Vector Machine": load_pickle('svm_model.pkl'),
    "Random Forest": load_pickle('rf_model.pkl')
  }

pg = st.navigation([st.Page("machine_learning.py", title="Machine Learning")
                    , st.Page("deep_learning.py", title="Deep Learning")])
pg.run()