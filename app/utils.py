import tensorflow as tf
from typing import List
import cv2
import os 

# Define the vocabulary for the speech recognition model
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create a mapping from characters to integer ids
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# Create a mapping from integer ids back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]: 
    # Open the video at the specified path
    cap = cv2.VideoCapture(path)
    
    # Initialize a list to hold the video frames
    frames = []
    
    # Loop through each frame in the video
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        
        # Convert the RGB frame to grayscale
        frame = tf.image.rgb_to_grayscale(frame)
        
        # Crop the frame to a specific size
        frames.append(frame[190:236,80:220,:])
    
    # Release the video object
    cap.release()
    
    # Calculate the mean and standard deviation of the frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    
    # Normalize the frames by subtracting the mean and dividing by the standard deviation
    return tf.cast((frames - mean), tf.float32) / std
    
def load_alignments(path:str) -> List[str]: 
    # Open the alignment file at the specified path
    with open(path, 'r') as f: 
        lines = f.readlines() 
    
    # Initialize a list to hold the speech tokens
    tokens = []
    
    # Loop through each line in the alignment file
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    
    # Convert the speech tokens to integer ids using the character-to-integer mapping
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    # Convert the input path from bytes to string
    path = bytes.decode(path.numpy())
    
    # Extract the file name from the input path
    file_name = path.split('/')[-1].split('.')[0]
    
    # Create paths to the video file and alignment file based on the file name
    video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
    
    # Load the video frames and speech alignments
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    # Return the frames and alignments as a tuple
    return frames, alignments
