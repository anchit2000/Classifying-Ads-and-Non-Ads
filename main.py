import streamlit as st
import pandas as pd 
import numpy as np
from model import building_text_model
from model import building_photo_model
from model import building_photo_modelcnn

st.title("Advertisement detection sing ML")

"""
This project is divided into two parts.
"""
"""
1. AD detection from texts
"""
"""
2. AD detection from photos
"""
st.info("Open Sidebar")
text = st.sidebar.button("Classify using text")
photo = st.sidebar.button("Classify photos using Logistic Regression")
photo1 = st.sidebar.button("Classify photos using CNN (VGG_19)")
text_user_input = st.text_input("Enter some text to classify")
photo_user_input = st.file_uploader("Upload a photo",type = 'jpg')

if text:
	result = building_text_model(text_user_input)
	# st.write(text_user_input)
	if result:
		st.write(text_user_input + " was clasified as an ad")
	else:
		st.write(text_user_input + " was clasified as a non-ad")
		st.balloons()
if photo:
	st.image(photo_user_input)
	result = building_photo_model(photo_user_input)
	if result:
		st.write(text_user_input + " was clasified as an ad using logistic regression")
	else:
		st.write(text_user_input + " was clasified as a non-ad using logistic regression")
		st.balloons()
		
if photo1:
	st.image(photo_user_input)
	result = building_photo_modelcnn(photo_user_input)
	if result == 0:
		st.write(text_user_input + " was clasified as an ad using CNN")
	else:
		st.write(text_user_input + " was clasified as a non-ad using CNN")
		st.balloons()







