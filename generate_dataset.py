import numpy as np
import os
import gymnasium as gym
import yaml
import matplotlib.pyplot as plt

from logger import Logger
from utils import stack_frames, len_of_episode

# Gymnasium is using np.bool8, which is deprecated.
from warnings import filterwarnings

filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias",
)


class GenerateDataset:
    """
    Class for generating datasets for the multi-fidelity deep kernel learning model.
    It generates a dataset of frames from a Gymnasium environment at multiple levels of fidelity.

    The input parameters are stored in a dictionary with the following keys:
    - env_name: the name of the Gym environment;
    - obs_dim_1: the first dimension of the frame;
    - obs_dim_2: the second dimension of the frame;
    - crop: whether to crop the frame;
    - occlusion: whether to add an occlusion to the frame;
    - num_episodes: the number of episodes to generate.

    The dataset is saved in the Data folder as a pickle file for each level of fidelity, 
    containing a list of dictionaries with:
    - obs: the frames at time t-1 and t, stacked;
    - next_obs: the frames at time t and t+1, stacked;
    - terminated: whether the episode is terminated;
    - state: the current state of the environment;
    - next_state: the next state of the environment.
    
    The class can also save the frames as PNG images in the png folder.

    Args:
        args (dict): The dictionary of input parameters.
        seed (int): The seed for the random number generator. Default is 1.
        test (bool): Whether to generate a test set. Default is False.
        png (bool): Whether to save frames as PNG images. Default is False.

    Attributes:
        levels (int): The number of levels of fidelity for the dataset.
        seed (int): The seed for the random number generator.
        png (bool): Whether to save frames as PNG images.
        env_name (str): The name of the Gym environment.
        data_filename (list): The list of filenames for the dataset.
        frame_dim1 (list): The list of the first dimension of the frame at each level.
        frame_dim2 (list): The list of the second dimension of the frame at each level.
        crop (list): The list of flags indicating whether to crop the frame at each level.
        occlusion (list): The list of flags indicating whether to add an occlusion to the frame at each level.
        num_episodes (list): The list of the number of episodes to generate at each level.
        max_steps (int): The maximum number of steps in an episode.
        env (gym.Env): The Gym environment.
        folder (str): The path to the Data folder.
        png_folder (str): The path to the png folder.
        logger (list): The list of loggers for each level of fidelity.

    Methods:
        set_environment: Sets up the Gym environment based on the specified environment name.
        generate_dataset: Generates a dataset by running episodes in the environment and logging observations.
        new_obs: Generates a new observation by stacking frames at different levels of fidelity.
        log_obs: Logs the observation as a tuple with the current frame, the next one, the terminated status, the current state, and the next state.
        save_log: Saves the dataset as a pickle file for each level of fidelity.
        save_image: Saves the given frame as an image.
    """

    def __init__(self, args, seed=1, test=False, png=False):
        """
        Initializes the GenerateDataset class.

        Args:
            args (dict): The dictionary of input parameters.
            seed (int): The seed for the random number generator. Default is 1.
            test (bool): Flag indicating whether to generate a test set or not. Default is False.
            png (bool): Flag indicating whether to generate PNG images or not. Default is False.
        """

        super(GenerateDataset, self).__init__()

        self.levels = len(args["training_dataset"])
        self.seed = seed
        self.png = png

        self.env_name = args["env_name"]
        if test:
            self.data_filename = [f"data_test_{i}.pkl" for i in range(self.levels)]
            self.num_episodes = args["num_episodes_test"]
            print("Generating test set...")
        else:
            self.data_filename = [f"data_train_{i}.pkl" for i in range(self.levels)]
            self.num_episodes = args["num_episodes"]
            print("Generating training set...")

        self.frame_dim1 = args["obs_dim_1"]
        self.frame_dim2 = args["obs_dim_2"]
        self.crop = args["crop"]
        self.occlusion = args["occlusion"]

        # Set the maximum number of steps in an episode, according to the chosen environment
        self.max_steps = len_of_episode(self.env_name)

        # Call the set_environment method
        self.set_environment()

        # Set logger
        directory = os.path.dirname(os.path.abspath(__file__))
        self.folder = os.path.join(directory + "/Data/")
        self.png_folder = os.path.join(self.folder + "png/")
        self.logger = [Logger(self.folder) for i in range(levels)]


    def set_environment(self):
        """
        Sets up the Gym environment based on the specified environment name.

        Parameters:
        - self.env_name (str): The name of the Gym environment.
        - self.seed (int): The seed value for random number generation.

        Returns:
        - None
        """

        # Set Gym environment
        if self.env_name == "Pendulum":
            self.env = gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
        elif self.env_name == "Acrobot":
            self.env = gym.make("Acrobot-v1", render_mode="rgb_array")
        elif self.env_name == "MountainCarContinuous":
            self.env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
        else:
            assert False, "Invalid environment name. Test case not supported."

        # Set seeds
        self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)


    def generate_dataset(self):
        """
        Generates a dataset by running episodes in the environment and logging observations.
        By default, the action taken is a null action: 
        - (0.0) for Pendulum;
        - 1 for Acrobot;
        - (0.0) for MountainCarContinuous.

        Returns:
            None
        """

        np.random.seed(self.seed)
        if self.env_name == "Pendulum":
            action = np.array([0.0])  # null action
        elif self.env_name == "Acrobot":
            action = 1  # null action
        elif self.env_name == "MountainCarContinuous":
            action = np.array([0.0])  # null action
        else:
            assert False, "Invalid environment name. Test case not supported."

        for episode in range(self.num_episodes[0]):
            # Reset the environment with new (random) initial conditions
            # and render the first frames.
            state = self.env.reset()[0]
            frame0 = np.array(self.env.render())
            frame1 = np.array(self.env.render())

            print("Episode: ", episode)
            for step_index in range(self.max_steps):
                # Generate the observations
                obs = self.new_obs(frame0, frame1)

                # Render new frame: 
                # run one timestep of the environment’s dynamics using the agent actions
                next_state, _, terminated, _, _ = self.env.step(action)
                frame2 = np.array(self.env.render())
                next_obs = self.new_obs(frame1, frame2)

                # Set the terminated flag if the episode is over
                if step_index == self.max_steps - 1:
                    terminated = True

                # Log the observations
                self.log_obs(episode, obs, next_obs, terminated, state, next_state)

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


    def new_obs(self, frame1, frame2):
        """
        Generate a new observation by stacking frames at different levels of fidelity.

        Args:
            frame1: The previous frame.
            frame2: The current frame.

        Returns:
            A list of stacked frames at different levels of fidelity.
        """

        obs = []

        for l in range(self.levels):
            obs_l = stack_frames(
                prev_frame=frame1,
                frame=frame2,
                size1=self.frame_dim1[l],
                size2=self.frame_dim2[l],
                crop=self.crop[l],
                occlusion=self.occlusion[l],
            )
            obs.append(obs_l)

        return obs


    def log_obs(self, episode, obs, next_obs, terminated, state, next_state):
        """
        Logs the observation as a tuple with: 
        - the current frame;
        - the next one;
        - the terminated status;
        - the current state;
        - the next state.

        Args:
            episode (int): The episode number.
            obs (list): A list of observations at each level.
            next_obs (list): A list of next observations at each level.
            terminated (bool): Indicates whether the episode is terminated.
            state (object): The current state.
            next_state (object): The next state.

        Returns:
            None
        """

        self.logger[0].obslog(dict(obs=obs[0], next_obs=next_obs[0], terminated=terminated, state=state, next_state=next_state))

        for l in range(1, self.levels):
            if episode < self.num_episodes[l]:
                self.logger[l].obslog(dict(obs=obs[l], next_obs=next_obs[l], terminated=terminated, state=state, next_state=next_state))


    def save_log(self):
        """
        Save the dataset as a pickle file for each level of fidelity.
        This method saves the dataset by iterating over the levels and calling the `save_obslog` method of each logger.

        Args:
            None

        Returns:
            None
        """
        
        for l in range(self.levels):
            self.logger[l].save_obslog(filename=self.data_filename[l])


    def save_image(self, frame, filename, show=False):
        """
        Save the given frame as an image.

        Args:
            frame: The frame to be saved as an image.
            filename: The name of the file to save the image as.
            show: A boolean indicating whether to display the image after saving (default is False).
        """

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
    seed = args["seed"]

    # Training set
    train_set = GenerateDataset(args=args, seed=seed, test=False, png=False)
    train_set.generate_dataset()

    # Test set
    test_set = GenerateDataset(args=args, seed=seed+1, test=True, png=False)
    test_set.generate_dataset()
