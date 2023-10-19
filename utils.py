import numpy as np
import pickle
import cv2


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file):
    if not file[-3:] == 'pkl' and not file[-3:] == 'kle':
        file = file+'pkl'

    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data


def stack_frames(prev_frame, frame, size1=84, size2=84):
    prev_frame = cv2.resize(prev_frame, (size1, size2))
    frame = cv2.resize(frame, (size1, size2))
    stacked_frames = np.concatenate((prev_frame, frame), axis=-1)
    return stacked_frames


def sample_batch(data, batch_size=32):
    idxs = np.random.randint(0, len(data), batch_size)
    return data[idxs]