from turtle import width
import streamlit as st
import moviepy.editor as mp
from moviepy.editor import *

st.header('What is this project about ')

st.write('Hello this is a little projet for our deep learning class')
st.write('We were asked to design a model that is able to detect car in a video tape')

st.sidebar.image("./image/logo.png")
st.sidebar.write(" Eliel KATCHE \t Sven LOTHE")

st.markdown('# Which model we will use ')
st.write('Yolov4')

clip = "./videos/car_1.mp4"
st.video(clip)

st.write(" we will apply the algorithm to detect car (see below)")

st.image('./image/car_detected.png')
st.markdown('## Why this model?  ')

st.markdown('## The parameters  ')

st.markdown('# Our data  ')

