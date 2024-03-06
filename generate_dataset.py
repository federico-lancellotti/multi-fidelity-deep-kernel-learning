import numpy as np
import os
import gymnasium as gym
import yaml
import matplotlib.pyplot as plt

from logger import Logger
from utils import stack_frames

# Gymnasium is using np.bool8, which is deprecated.
from warnings import filterwarnings

filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias",
)


class GenerateDataset:
    def __init__(self, levels=2, test=False, png=False):
        super(GenerateDataset, self).__init__()

        self.levels = levels
        self.png = png

        # Config inputs
        with open("config.yaml", "r") as file:
            args = yaml.safe_load(file)

        self.env_name = args["env_name"]
        if test:
            self.data_filename = [f"pendulum_test_{i}.pkl" for i in range(self.levels)]
            print("Generating test set...")
        else:
            self.data_filename = [f"pendulum_train_{i}.pkl" for i in range(self.levels)]
            print("Generating training set...")

        self.frame_dim1 = args["obs_dim_1"]
        self.frame_dim2 = args["obs_dim_2"]
        self.num_episodes = args["num_episodes"]
        self.max_steps = 200  # Pendulum-v1 episode truncates at 200 time steps.
        self.seed = args["seed"]

        self.set_environment()

        # Set logger
        directory = os.path.dirname(os.path.abspath(__file__))
        self.folder = os.path.join(directory + "/Data/")
        self.png_folder = os.path.join(self.folder + "png/")
        self.logger = [Logger(self.folder) for i in range(levels)]

    # Generate and set the Gymnasium environment
    def set_environment(self):
        # Set Gym environment
        self.env = gym.make(self.env_name, g=9.81, render_mode="rgb_array")

        # Set seeds
        self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)

    # Generate the dataset, frame by frame.
    def generate_dataset(self):
        np.random.seed(self.seed)
        action = np.array([0.0])  # null action

        for episode in range(self.num_episodes[0]):
            # Reset the environment with new (random) initial conditions
            # and render the first frames.
            state = self.env.reset()
            frame0 = np.array(self.env.render())
            frame1 = np.array(self.env.render())

            print("Episode: ", episode)
            for step_index in range(self.max_steps):
                obs = self.new_obs(frame0, frame1)

                # Render new frame
                # run one timestep of the environment’s dynamics using the agent actions
                next_state, _, terminated, _, _ = self.env.step(action)
                frame2 = np.array(self.env.render())
                next_obs = self.new_obs(frame1, frame2)

                if step_index == self.max_steps - 1:
                    terminated = True

                # Log the observations
                self.log_obs(episode, obs, next_obs, terminated)

                # Print png
                if self.png:
                    frame_name = self.png_folder + str(episode) + "_" + str(step_index)
                    self.save_image(frame1, frame_name, show=False)

                # Prepare the next iteration
                frame0 = frame1
                frame1 = frame2
                state = next_state

        # Save the dataset
        self.save_log()

        # Close the Gym environment
        self.env.close()
        print("Done.")

    # Produce the new observation, stacking two consecutive frames.
    def new_obs(self, frame1, frame2):
        obs = []

        for l in range(self.levels):
            obs_l = stack_frames(
                prev_frame=frame1,
                frame=frame2,
                size1=self.frame_dim1[l],
                size2=self.frame_dim2[l],
            )
            obs.append(obs_l)

        return obs

    # Log the observation in the logger, as a tuple with the current frame,
    # the next one and the terminated status.
    def log_obs(self, episode, obs, next_obs, terminated):
        self.logger[0].obslog(dict(obs=obs[0], next_obs=next_obs[0], terminated=terminated))

        for l in range(1, self.levels):
            if episode < self.num_episodes[l]:
                self.logger[l].obslog(dict(obs=obs[l], next_obs=next_obs[l], terminated=terminated))

    # Save locally the complete log as file.
    def save_log(self):
        # Save the dataset
        for l in range(self.levels):
            self.logger[l].save_obslog(filename=self.data_filename[l])

    # Save locally the frame as png.
    def save_image(self, frame, filename, show=False):
        # Print the frame as image with Matplotlib
        plt.imshow(frame)
        plt.axis("off")  # hide the axis

        # Save the image locally
        filename = filename + ".png"
        plt.savefig(filename)

        # Show the image (if show==1)
        if show:
            plt.show()


if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        args = yaml.safe_load(file)
    levels = len(args["training_dataset"])

    # Training set
    train_set = GenerateDataset(levels=levels, test=False, png=False)
    train_set.generate_dataset()

    # Test set
    test_set = GenerateDataset(levels=levels, test=True, png=False)
    test_set.generate_dataset()
