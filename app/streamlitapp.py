# import necessary dependencies
import streamlit as st
import os
import imageio

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# set the layout of the Streamlit app to wide
st.set_page_config(layout='wide')

# set up the sidebar
with st.sidebar:
    # display logo and app title
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model')

# display app title
st.title('LipNet full stack app')

# generate a list of options for videos
options = os.listdir(os.path.join('..','data','s1'))
selected_video = st.selectbox('Choose Video',options)

# set up two columns for displaying the video and model output
col1, col2 = st.columns(2)

if options:
    # render the selected video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        # set up file paths for input and output videos
        file_path = os.path.join('..','data','s1', selected_video)
        output_path = 'test_video.mp4'
        # convert video to mp4 format using ffmpeg
        cmd = f'ffmpeg -i {file_path} -vcodec libx264 {output_path} -y'
        status = os.system(cmd)
        if status == 0 and os.path.exists(output_path):
            # display the converted video inside the app
            video = open(output_path, 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)
        else:
            # display an error message if video conversion fails
            st.error(f'Error: Failed to convert video. Status code: {status}')

    # display the video as input to the model
    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        # load the video and its annotations
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        # convert the video to a GIF for display
        imageio.mimsave('Animation.gif',video,fps=10)
        st.image('Animation.gif',width=400)

        # display the output of the model as tokens
        st.info('This is the output of the machine learning as tokens')
        model = load_model()
        # predict the text from the video
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # convert the model output from tokens to text
        st.info('This is the decoded text')
        converted_preds = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_preds)
