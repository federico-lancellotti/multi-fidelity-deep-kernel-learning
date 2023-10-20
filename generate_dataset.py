import numpy as np
import os
import gymnasium as gym

from logger import Logger
from utils import stack_frames

# Gymnasium is using np.bool8, which is deprecated.
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')


# Config inputs (to migrate in a YAML file)
env_name = 'Pendulum-v1'
test = 0
if test:
    data_file_name = 'pendulum_test.pkl'
else:
    data_file_name = 'pendulum_train.pkl'
frame_dim1 = 84
frame_dim2 = 84
num_episodes = 5
seed = 2
random_policy = True

# Set Gym environment
env = gym.make(env_name, g=9.81, render_mode="rgb_array")

# Set seeds
env.reset(seed=seed) # it should be enough to set 'np.random.seed(seed)', but just in cases...
env.action_space.seed(seed)
np.random.seed(seed)

# Set logger
directory = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(directory + '/Data/')
logger = Logger(folder)

max_step = 200 # Pendulum-v1 episode truncates at 200 time steps.

# First step (with random action)
action = env.action_space.sample() # only at the first step, we take a random action
observation, reward, terminated, truncated, info = env.step(action) # run one timestep of the environment’s dynamics using the agent actions
frame = np.array(env.render()) # compute the render frames

action = np.array([0.0]) # null action
for step_index in range(max_step):
    prev_frame = frame

    # Render new frame
    observation, reward, terminated, truncated, info = env.step(action)
    frame = np.array(env.render())
    frame = frame.astype(np.float32) / 255

    # Stack and log two consecutive frames
    obs = stack_frames(prev_frame=prev_frame, frame=frame, size1=frame_dim1, size2=frame_dim2)
    logger.obslog(obs)

# Save the dataset
logger.save_obslog(filename=data_file_name)

# Close the Gym environment
env.close()
