import numpy as np
import os
import gymnasium as gym
import yaml

from logger import Logger
from utils import stack_frames

# Gymnasium is using np.bool8, which is deprecated.
from warnings import filterwarnings

filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias",
)


def generate_dataset(test):
    # Config inputs
    with open("config.yaml", "r") as file:
        args = yaml.safe_load(file)

    env_name = args["env_name"]
    if test:
        data_file_name = ["pendulum_test_0.pkl", "pendulum_test_1.pkl"]
        print("Generating test set...")
    else:
        data_file_name = ["pendulum_train_0.pkl", "pendulum_train_1.pkl"]
        print("Generating training set...")
    frame_dim1 = args["obs_dim_1"]
    frame_dim2 = args["obs_dim_2"]
    num_episodes = args["num_episodes"]
    seed = args["seed"]

    # Set Gym environment
    env = gym.make(env_name, g=9.81, render_mode="rgb_array")

    # Set seeds
    env.reset(
        seed=seed
    )  # it should be enough to set 'np.random.seed(seed)', but just in cases...
    env.action_space.seed(seed)
    np.random.seed(seed)

    # Set logger
    directory = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(directory + "/Data/")
    logger = [Logger(folder), Logger(folder)]

    max_step = 200  # Pendulum-v1 episode truncates at 200 time steps.
    action = np.array([0.0])  # null action

    for episode in range(num_episodes[0]):
        state = (
            env.reset()
        )  # reset the environment with new (random) initial conditions
        frame = np.array(env.render())  # renders the frame

        print("Episode: ", episode)
        for step_index in range(0, max_step):
            prev_frame = frame

            # Render new frame
            observation, reward, terminated, truncated, info = env.step(
                action
            )  # run one timestep of the environment’s dynamics using the agent actions
            frame = np.array(env.render())

            # Stack and log two consecutive frames
            obs0 = stack_frames(
                prev_frame=prev_frame,
                frame=frame,
                size1=frame_dim1[0],
                size2=frame_dim2[0],
            )
            logger[0].obslog((obs0, terminated))

            if episode < num_episodes[1]:
                obs1 = stack_frames(
                    prev_frame=prev_frame,
                    frame=frame,
                    size1=frame_dim1[1],
                    size2=frame_dim2[1],
                )
                logger[1].obslog((obs1, terminated))

    # Save the dataset
    logger[0].save_obslog(filename=data_file_name[0])
    logger[1].save_obslog(filename=data_file_name[1])

    # Close the Gym environment
    env.close()
    print("Done.")


if __name__ == "__main__":
    generate_dataset(test=0)
    generate_dataset(test=1)
