import numpy as np
import pickle
#import cv2
from PIL import Image

# Saves and loads the data as a pickle file
def save_pickle(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file):
    if not file[-3:] == "pkl" and not file[-3:] == "kle":
        file = file + "pkl"

    with open(file, "rb") as f:
        data = pickle.load(f)

    return data


# Concatenates two consecutive frames along the channel dimension
def stack_frames(prev_frame, frame, size1=84, size2=84):
    prev_frame = np.array(Image.fromarray(prev_frame).resize((size1, size2)))
    frame = np.array(Image.fromarray(frame).resize((size1, size2)))
    stacked_frames = np.concatenate((prev_frame, frame), axis=-1)
    return stacked_frames


# Returns a np.array random batch of dimension batch_size
def sample_batch(data, batch_size=32):
    idxs = np.random.randint(0, len(data), batch_size)
    batch = [data[i] for i in idxs]
    return np.array(batch)
