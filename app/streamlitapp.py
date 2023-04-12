#import all the dependencies
import streamlit as st
import os
import imageio

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

#set the layout to the streamlit app as wide
st.set_page_config(layout='wide')


#setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('this application is originally developed from the lipnet deep learning model')


st.title('LipNet full stack app')
#generating a list of options or videos
options = os.listdir(os.path.join('..','data','s1'))
selected_video = st.selectbox('Choose Video',options)

col1, col2 = st.columns(2)

if options:
    # Rendering the video 
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        output_path = 'test_video.mp4'
        cmd = f'ffmpeg -i {file_path} -vcodec libx264 {output_path} -y'
        status = os.system(cmd)
        if status == 0 and os.path.exists(output_path):
            # Rendering inside of the app
            video = open(output_path, 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)
        else:
            st.error(f'Error: Failed to convert video. Status code: {status}')


    with col2:
        st.info('this is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('Animation.gif',video,fps=10)
        st.image('Animation.gif',width=400)
        
        #this is the output of the machine learning model as 
        st.info('this is output of the machine learning as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        #convert prediction to text 
        st.info('decode the raw tokens into words')
        converted_preds = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_preds)




