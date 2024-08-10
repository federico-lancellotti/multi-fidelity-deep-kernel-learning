import numpy as np
import torch
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio


def save_pickle(filename, data):
    """
    Save data as a pickle file.

    Parameters:
        filename (str): The name of the file to save.
        data: The data to be saved.

    Returns:
        None
    """

    with open(filename, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file):
    """
    Load data from a pickle file.

    Parameters:
        file (str): The path to the pickle file.

    Returns:
        data: The loaded data from the pickle file.
    """

    if not file[-3:] == "pkl" and not file[-3:] == "kle":
        file = file + "pkl"

    with open(file, "rb") as f:
        data = pickle.load(f)

    return data


def stack_frames(prev_frame, frame, size1=84, size2=84, crop={'portion':1, 'pos':1}, occlusion={'portion':0, 'pos':1}):
    """
    Stack two frames together along the channel dimension and perform optional cropping and occlusion.

    Args:
        prev_frame (numpy.ndarray): The previous frame.
        frame (numpy.ndarray): The current frame.
        size1 (int, optional): The desired width of the stacked frames. Defaults to 84.
        size2 (int, optional): The desired height of the stacked frames. Defaults to 84.
        crop (dict, optional): The cropping parameters. Defaults to {'portion':1, 'pos':1} (complete image).
        occlusion (dict, optional): The occlusion parameters. Defaults to {'portion':0, 'pos':1} (no occlusion).

    Returns:
        numpy.ndarray: The stacked frames.
    """

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


def crop_frame(frame, portion=0.75, pos=1):
    """
    Crop a frame based on the specified portion and position.

    Parameters:
        - frame: numpy.ndarray
            The input frame to be cropped.
        - portion: float, optional
            The portion of the frame to be retained after cropping. Default is 0.75.
        - pos: int, optional
            The position of the cropped portion within the frame. 
            1: top-right, 2: top-left, 3: bottom-left, 4: bottom-right. Default is 1.

    Returns:
        - cropped_frame: numpy.ndarray
            The cropped frame.
    """

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


def add_occlusion(frame, portion=0.25, pos=1):
    """
    Add occlusion to a given frame.

    Parameters:
        - frame: numpy.ndarray
            The input frame to which occlusion will be added.
        - portion: float, optional
            The portion of the frame to be occluded. Default is 0.25.
        - pos: int, optional
            The position of the occlusion. Valid values are 1, 2, 3, and 4.
            1: Top right corner
            2: Top left corner
            3: Bottom left corner
            4: Bottom right corner
            Default is 1.

    Returns:
        - frame_with_occlusion: numpy.ndarray
            The frame with occlusion added.

    Note:
        - The occluded portion of the frame will be set to 0, namely black.
    """

    size = frame.shape
    newsize = (portion * np.array(size)).astype(int)
    frame_with_occlusion = frame.copy()

    if pos==1:
        frame_with_occlusion[:newsize[0], size[1]-newsize[1]:, :] = .0
    elif pos==2:
        frame_with_occlusion[:newsize[0], :newsize[1], :] = .0
    elif pos==3:
        frame_with_occlusion[size[0]-newsize[0]:, :newsize[1], :] = .0
    elif pos==4:
        frame_with_occlusion[size[0]-newsize[0]:, size[1]-newsize[1]:, :] = .0
    else:
        print("Wrong choice of pos.")

    return frame_with_occlusion


def plot_frame(frame, show=False, filename="", pause=0):
    """
    Plot a frame (image) and optionally save it to a file and/or display it.

    Parameters:
        - frame: The frame (image) to be plotted.
        - show (optional): If True, display the plotted frame on the screen. Default is False.
        - filename (optional): If provided, save the plotted frame to a .svg file with the given filename. Default is an empty string.
        - pause (optional): The time to pause before closing the plot. Default is 0.
        
    Returns:
        None
    """

    plt.imshow(frame)
    plt.axis("on")  # hide the axis
    plt.xticks([])
    plt.yticks([])

    # Save the image locally
    if filename:
        filename = filename + ".svg"
        plt.savefig(filename)

    # Print to screen
    if show:
        if pause:
            plt.pause(pause)
        else:
            plt.show()


def plot_latent_dims(z, var, T_start, T, dt, dims=3, episodes=3, show=False, filename=None):
    """
    Plot the latent dimensions of the given data, as functions of time, 
    with two standard deviation upper and lower bounds, for uncertainty quantification.

    Args:
        z (numpy.ndarray): The input data with shape (N, D), where N is the number of samples and D is the number of dimensions.
        var (numpy.ndarray): The variance of the latent dimensions.
        T_start (int): The starting time.
        T (int): The length of each episode.
        dt (float): The time step.
        dims (int, optional): The number of latent dimensions to plot. Defaults to 3.
        episodes (int, optional): The number of episodes to plot. Defaults to 3.
        show (bool, optional): Whether to display the plot. Defaults to False.
        filename (str, optional): The filename to save the plot, as .svg file. Defaults to None.

    Returns:
        None
    """
    # Loop over each latent dimension
    for i in range(dims):
        theta_i = z[:,i] # extract the current dimension (state variable) theta_i
        var_i = var[:,i]
        plt.figure(figsize=(15, 5))

        # Loop over each episode
        episodes = min(episodes, int(len(theta_i)/T))
        time = T_start + np.arange(T)*dt
        for j in range(episodes):
            pos = j*T
            l = T
            plt.plot(time, theta_i[pos:pos+l], linewidth=1.5, alpha=0.7)

            # Plot the variance
            upper_bound = theta_i[pos:pos+l] + 2 * np.sqrt(var_i[pos:pos+l])
            lower_bound = theta_i[pos:pos+l] - 2 * np.sqrt(var_i[pos:pos+l])
            plt.fill_between(time, lower_bound, upper_bound, color='gray', alpha=0.3, label='2σ Band')


        plt.xlabel('t')
        plt.ylabel(r"$\theta_{" + str(i+1) + r"}$")
        plt.title('Latent dimension ' + str(i+1))
        plt.grid(True)

        # Save the image locally
        if filename:
            plt.savefig(filename + "x_" + str(i+1) + ".svg")

        # Print to screen
        if show:
            plt.show()
        
        plt.close()


def len_of_episode(env_name):
    """
    Return the length of an episode for the given environment.

    Args:
        env_name (str): The name of the environment. Supported values are "Pendulum", "Acrobot", and "MountainCarContinuous".

    Returns:
        int: The length of an episode for the given environment.
    """
    if env_name == "Pendulum":
        return 200  # Pendulum-v1 episode truncates at 200 time steps.
    elif env_name == "Acrobot":
        return 500  # Acrobot-v1 episode truncates at 500 time steps.
    elif env_name == "MountainCarContinuous":
        return 400  # MountainCarContinuous-v0 episode truncates at 999 time steps.
    else:
        assert False, "Invalid environment name. Test case not supported."
        

def heatmap_to_image(u):
    """
    Convert a matrix to its heatmap as image.

    Args:
        u (numpy.ndarray): The matrix to be converted to a heatmap.

    Returns:
        Image: The heatmap image.
    """

    u = (u - u.min()) / (u.max() - u.min())
    u = cm.viridis(u)[:,:,:3]
    u = (u * 255).astype(np.uint8)

    return u


def align_pde(len_0, len_1, mu0, mu1):
    """
    Compute indices to align the levels of fidelity in time.

    Args:
        len_0 (int): The length of observation of one episode at level of fidelity 0.
        len_1 (int): The length of observation of one episode at level of fidelity 1.
        mu0 (list): The list of mu values for level 0.
        mu1 (list): The list of mu values for level 1.

    Returns:
        numpy.ndarray: The indices to align the levels of fidelity in time.
    """

    mu_positions = {mu_val: i for i, mu_val in enumerate(mu0)}

    idx = []
    for mu_val in mu1:
        i = mu_positions[mu_val]
        start = i*len_0
        new_idx = range(start, start+len_1)
        idx.append(new_idx)

    idx = np.array(idx).flatten()

    return idx


def get_length(x):
    """
    Get the length of a variable.

    Args:
        x: The variable to get the length of.

    Returns:
        int: The length of the variable.
    """

    if isinstance(x, list):
        return len(x)
    elif isinstance(x, (int, float)):
        return 1
    else:
        raise TypeError("The variable is not a list or a number.")


def check_indices(tensor, indices):
    assert torch.all(indices >= 0) and torch.all(indices < tensor.size(0)), "Index out of bounds"


def generate_gif(filepath, start, end, step=1, filename="movie.gif"):
    """
    Generate a gif from a sequence of images.

    Args:
        filepath (str): The path to the images. Warning: .png files are expected.
        start (int): The starting index of the images.
        end (int): The ending index of the images.
        filename (str, optional): The name of the gif file. Defaults to "movie.gif".

    Returns:
        None
    """

    filenames = [filepath + str(i) + ".png" for i in range(start, end, step)]
    images = []
    
    for f in filenames:
        images.append(imageio.imread(f))

    result_filename = filepath + filename
    imageio.mimsave(result_filename, images)


def plot_error(error, T_start, dt, labels, filepath):
    """
    Plot the error.

    Args:
        error (list): The list of errors.
        T_start (int): The starting time.
        dt (float): The time step.
        labels (list): The list of labels.
        filepath (str): The path to save the plot.

    Returns:
        None
    """

    time = T_start + np.arange(len(error[0])) * dt

    for i in range(len(error)):
        plt.plot(time, error[i], label=labels[i])

    plt.legend()
    plt.title("Reconstruction error")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.savefig(filepath + "error.svg", format="svg")
    plt.close()