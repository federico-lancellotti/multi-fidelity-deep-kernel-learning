import numpy as np
import pickle
#import cv2
from PIL import Image
import matplotlib.pyplot as plt


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
def stack_frames(prev_frame, frame, size1=84, size2=84, crop={'portion':1, 'pos':1}, occlusion={'portion':0, 'pos':1}):
    if crop['portion'] < 1 and crop['portion'] >= 0:
        prev_frame = crop_frame(prev_frame, crop['portion'], crop['pos'])
        frame = crop_frame(frame, crop['portion'], crop['pos'])

    if occlusion['portion'] > 0 and occlusion['portion'] <= 1:
        prev_frame = add_occlusion(prev_frame, occlusion['portion'], occlusion['pos'])
        frame = add_occlusion(frame, occlusion['portion'], occlusion['pos'])
    
    prev_frame = np.array(Image.fromarray(prev_frame).resize((size1, size2)))
    frame = np.array(Image.fromarray(frame).resize((size1, size2)))
    stacked_frames = np.concatenate((prev_frame, frame), axis=-1)
    return stacked_frames


# Crops a frame. The function returns a new frame which is the
# portion% corner of the original frame, with coordinates:
#  - pos=1: top-right corner
#  - pos=2: top-left corner
#  - pos=3: bottom-left corner
#  - pos=4: bottom-right corner
def crop_frame(frame, portion=0.75, pos=1):
    size = frame.shape
    newsize = (portion * np.array(size)).astype(int)

    if pos==1:
        return frame[:newsize[0], size[1]-newsize[1]:, :]
    elif pos==2:
        return frame[:newsize[0], :newsize[1], :]
    elif pos==3:
        return frame[size[0]-newsize[0]:, :newsize[1], :]
    elif pos==4:
        return frame[size[0]-newsize[0]:, size[1]-newsize[1]:, :]
    else:
        print("Wrong choice of pos.")
        return frame

# Adds a black occlusion to the frame. The function returns 
# the original frame with a black square imposed to the:
#  - pos=1: top-right corner
#  - pos=2: top-left corner
#  - pos=3: bottom-left corner
#  - pos=4: bottom-right corner
def add_occlusion(frame, portion=0.25, pos=1):
    size = frame.shape
    newsize = (portion * np.array(size)).astype(int)

    if pos==1:
        frame[:newsize[0], size[1]-newsize[1]:, :] = .0
    elif pos==2:
        frame[:newsize[0], :newsize[1], :] = .0
    elif pos==3:
        frame[size[0]-newsize[0]:, :newsize[1], :] = .0
    elif pos==4:
        frame[size[0]-newsize[0]:, size[1]-newsize[1]:, :] = .0
    else:
        print("Wrong choice of pos.")

    return frame


# Returns a np.array random batch of dimension batch_size
def sample_batch(data, batch_size=32):
    idxs = np.random.randint(0, len(data), batch_size)
    batch = [data[i] for i in idxs]
    return np.array(batch)

# Plots a frame
def plot_frame(frame, show=False, filename=""):
    plt.imshow(frame)
    plt.axis("off")  # hide the axis

    # Save the image locally
    if filename:
        filename = filename + ".png"
        plt.savefig(filename)

    # Print to screen
    if show:
        plt.show()

    plt.close()

# Plots the latent dimensions as functions of time
def plot_latent_dims(z, dims=3, T=200, show=False, filename=""):
    # Loop over each latent dimension
    for i in range(dims):
        theta_i = z[:,i] # extract the current dimension (state variable) theta_i
        plt.figure(figsize=(15, 5))

        # Loop over each episode
        #for j in range(int(len(theta_i)/T)):
        for j in range(2):
            pos = j*200
            l = T
            plt.plot(range(l), theta_i[pos:pos+l], linewidth=1.5, alpha=0.7)

        plt.xlabel('t')
        plt.ylabel('theta_' + str(i+1))
        plt.title('Latent dimension ' + str(i+1) + '-th')
        plt.grid(True)

        # Save the image locally
        if filename:
            plt.savefig(filename + "x_" + str(i+1) + ".png")

        # Print to screen
        if show:
            plt.show()
        
        plt.close()