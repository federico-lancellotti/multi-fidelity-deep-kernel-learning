import numpy as np
import os
import gymnasium as gym
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp

from .logger import Logger
from .utils import stack_frames, len_of_episode, heatmap_to_image

# Gymnasium is using np.bool8, which is deprecated.
from warnings import filterwarnings

filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message="`np.bool8` is a deprecated alias",
)


class GenerateDataset(ABC):
    """
    Abstract class for generating datasets for the multi-fidelity deep kernel learning model.
    It generates a dataset at multiple levels of fidelity.
    The class is inherited by the GenerateGym and GenerateReactionDiffusion classes, 
    which implement the abstract method generate_dataset for the Gymnasium environments 
    and the reaction-diffusion system, respectively.

    Args:
        args (dict): The dictionary of input parameters.
    
    Attributes:
        env_name (str): The name of the Gym environment.
        levels (int): The number of levels of fidelity for the dataset.
        directory (str): The path to the directory of the project.

    Methods:
        generate_dataset: Abstract method to generate the dataset.
    """

    def __init__(self, args):
        """
        Constructor for the GenerateDataset class.

        Args:
            args (dict): The dictionary of input parameters.
        """
        
        self.env_name = args["env_name"]
        self.levels = len(args["training_dataset"])
        self.directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    

    @abstractmethod
    def generate_dataset(self):
        """
        Abstract method to generate the dataset.

        Returns:
            None
        """

        pass



class GenerateGym(GenerateDataset):
    """
    Class for generating datasets for the multi-fidelity deep kernel learning model using Gymnasium environments.
    It generates a dataset of frames from a Gymnasium environment at multiple levels of fidelity.
    Inheriting from the GenerateDataset class, it implements the abstract method generate_dataset.

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

        super().__init__(args)

        self.seed = seed
        self.test = test
        self.png = png

        if self.test:
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
        self.folder = os.path.join(self.directory + "/Data/")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.png_folder = os.path.join(self.folder + "png/")
        self.logger = [Logger(self.folder) for i in range(self.levels)]


    def set_environment(self):
        """
        Sets up the Gym environment based on the specified environment name.

        Parameters:
        - self.env_name (str): The name of the Gym environment.
        - self.seed (int): The seed value for random number generation.

        Returns:
            None
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


class GenerateReactionDiffusion(GenerateDataset):
    """
    Class for generating the reaction-diffusion dataset for the multi-fidelity deep kernel learning model.
    It solves the reaction-diffusion system for two levels of fidelity and logs the data as images.
    Inheriting from the GenerateDataset class, it implements the abstract method generate_dataset.

    Args:
        args (dict): The dictionary of input parameters.

    Attributes:
        - T (dict): The time horizon for the two levels of fidelity.
        - dt (float): The time step.
        - d (dict): The diffusion coefficient for the two levels of fidelity.
        - mu (dict): The reaction coefficient for the two levels of fidelity.
        - L (float): The size of the domain.
        - n (dict): The number of grid points in one dimension for the two levels of fidelity.
        - levels (int): The number of levels of fidelity.
        - folder (str): The path to the folder where the data is saved.
        - train_data_filename (list): The filenames for the training data.
        - test_data_filename (list): The filenames for the test data.
        - Logger (list): The logger objects for the training data.
        - Logger_test (list): The logger objects for the test data.

    Methods:
        - generate_dataset: Generates the reaction-diffusion dataset for the multi-fidelity deep kernel learning model.
        - solve_reaction_diffusion: Solves the reaction-diffusion system for the given level of fidelity and parameter mu.
        - reaction_diffusion_rhs: Computes the right-hand side of the reaction-diffusion system.
    """
    
    def __init__(self, args):
        super().__init__(args)

        # Set the parameters
        self.T = args["T"]
        self.T_test = args["T_test"]
        self.dt = args["dt"]
        self.d = args["d"]
        self.mu = args["mu"]
        self.L = args["L"]
        self.n = args["n"]

        # Add a test time to the high-fidelity time horizon
        # NOTE: All the other levels of fidelity should have a equal or larger time horizon
        self.T[self.levels-1] = self.T[self.levels-1] + self.T_test 

        # Convert mu to list, if it is not already (we need to iterate over it later...)
        for level in range(self.levels):
            self.mu[level] = [self.mu[level]] if not isinstance(self.mu[level], list) else self.mu[level]

        # Set the folder to save the data
        self.folder = os.path.join(self.directory + "/Data/reaction-diffusion/")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Set the filenames and the logger objects
        self.train_data_filename = [f"data_train_{i}.pkl" for i in range(self.levels)]
        self.test_data_filename = [f"data_test_{i}.pkl" for i in range(self.levels)]
        self.Logger = [Logger(self.folder) for i in range(self.levels)]
        self.Logger_test = [Logger(self.folder) for i in range(self.levels)]


    def generate_dataset(self):
        """
        Generates the reaction-diffusion dataset for the multi-fidelity deep kernel learning model.
        It solves the reaction-diffusion system for two levels of fidelity and logs the data as images.
        In particular, it stacks three consecutive frames as two observations and logs them in a pickle file.
        It saves a pickle file for each level of fidelity for the training data and the test data, each containing:
        - obs: The observation (two consecutive frames, at time t and t+1).
        - next_obs: The next observation (two consecutive frames, at time t+1 and t+2).
        - t: The time index.
        - terminated: Whether the episode is terminated.

        Returns:
            None
        """

        print("Generating the reaction-diffusion dataset...")

        u = []
        for level in range(self.levels):
            u_l = []
            for mu in self.mu[level]:
                u_l.append(self.solve_reaction_diffusion(level, mu)[0])
            u.append(np.concatenate(u_l, axis=-1))

        # NOTE: The following assumes two levels of fidelity
                
        # Log the low-fidelity train data
        time_range = int(self.T[0]/self.dt)
        for index in range(len(self.mu[0])):
            t_start = index * time_range
            t_end = (index + 1) * time_range

            for t in range(time_range - 2):
                terminated = True if t == time_range - 3 else False
                self.log_obs(u=u[0][:,:,t_start:t_end], 
                             t=t, 
                             size=self.n[0], 
                             Logger=self.Logger[0], 
                             terminated=terminated)
        
        self.Logger[0].save_obslog(filename=self.train_data_filename[0])

        # Log the high-fidelity train data
        time_range = int((self.T[1] - self.T_test)/self.dt)
        for index in range(len(self.mu[1])):
            t_start = index * int(self.T[1]/self.dt) # exclude the test timeframe
            t_end = t_start + time_range

            for t in range(time_range - 2):
                terminated = True if t == time_range - 3 else False
                self.log_obs(u=u[1][:,:,t_start:t_end], 
                             t=t, 
                             size=self.n[1], 
                             Logger=self.Logger[1], 
                             terminated=terminated)
                
        self.Logger[1].save_obslog(filename=self.train_data_filename[1])

        # Log the test data
        start = int((self.T[self.levels-1] - self.T_test) / self.dt)
        time_range = int(self.T_test/self.dt)
        for level in range(self.levels):
            for index in range(len(self.mu[level])):
                t_start = index*int(self.T[level]/self.dt) + start
                t_end = t_start + time_range

                for t in range(time_range - 2):
                    terminated = True if t == t_end - 3 else False
                    self.log_obs(u=u[level][:,:,t_start:t_end], 
                                t=t, 
                                size=self.n[level], 
                                Logger=self.Logger_test[level], 
                                terminated=terminated)
                
            self.Logger_test[level].save_obslog(filename=self.test_data_filename[level])

        print("End.")


    def log_obs(self, u, t, size, Logger, terminated):
        """
        Logs the observation and the next observation for the given time index t.
        In particular, it logs:
        - obs: The observation (two consecutive frames, at time t and t+1).
        - next_obs: The next observation (two consecutive frames, at time t+1 and t+2).
        - t: The time index.
        - terminated: Whether the episode is terminated.

        Args:
            u (numpy.ndarray): The solution of the reaction-diffusion system.
            t (int): The time index.
            size (int): The size of the observation.
            Logger (Logger): The logger object.
            terminated (bool): Whether the episode is terminated.

        Returns:
            None
        """

        frame0 = heatmap_to_image(u[:,:,t])
        frame1 = heatmap_to_image(u[:,:,t+1])
        frame2 = heatmap_to_image(u[:,:,t+2])

        obs = stack_frames(frame0, frame1, size, size)
        next_obs = stack_frames(frame1, frame2, size, size)

        Logger.obslog(dict(obs=obs, next_obs=next_obs, t=t, terminated=terminated))


    def solve_reaction_diffusion(self, level, mu):
        """
        Solves the reaction-diffusion system for the given level of fidelity and parameter mu.

        Args:
            level (int): The level of fidelity.
            mu (float): The parameter mu.

        Returns:
            The solution of the reaction-diffusion system.
        """

        # Parameters and initial conditions
        d = self.d[level]
        mu = mu

        T = self.T[level]
        t = np.arange(0, T, self.dt)
        L = self.L
        n = self.n[level]
        N = n*n

        x2 = np.linspace(-L/2, L/2, n+1)
        x = x2[:-1]
        y = x
        kx = (2 * np.pi / L) * np.concatenate([np.arange(0, n//2), np.arange(-n//2, 0)])
        ky = kx

        X, Y = np.meshgrid(x, y)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K22 = K2.flatten()

        f = np.exp(-0.01 * (X**2 + Y**2))

        u_ini = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
        v_ini = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))

        uvt = np.concatenate([fft2(u_ini).flatten(), fft2(v_ini).flatten()])

        # Solve the reaction-diffusion system
        sol = solve_ivp(self.reaction_diffusion_rhs, [t[0], t[-1]], uvt, t_eval=t, args=(K22, d, mu, n, N), method='RK45')

        # Initialize the tensors to store the results
        u = np.zeros((n, n, len(sol.t)))
        v = np.zeros((n, n, len(sol.t)))

        # Store the results in the tensors
        for t in range(len(sol.t)):
            u_t = np.real(ifft2(np.reshape(sol.y[:N, t], (n, n))))
            v_t = np.real(ifft2(np.reshape(sol.y[N:, t], (n, n))))
            u[:, :, t] = u_t
            u[:, :, t] = v_t

        return u, v


    def reaction_diffusion_rhs(self, t, uvt, K22, d, mu, n, N):
        """
        Computes the right-hand side of the reaction-diffusion system.

        Args:
            t: The current time.
            uvt: The current state of the system.
            K22: The squared wavenumber.
            d: The diffusion coefficient.
            mu: The reaction coefficient.
            n: The number of grid points in one dimension.
            N: The total number of grid points.

        Returns:
            The right-hand side of the reaction-diffusion system.
        """

        ut = np.reshape(uvt[:N], (n, n))
        vt = np.reshape(uvt[N:], (n, n))
        u = np.real(ifft2(ut))
        v = np.real(ifft2(vt))

        u3 = u**3
        v3 = v**3
        u2v = (u**2) * v
        uv2 = u * (v**2)
        utrhs = np.reshape(fft2(u - u3 - uv2 + mu * u2v + mu * v3), N)
        vtrhs = np.reshape(fft2(v - u2v - v3 - mu * u3 - mu * uv2), N)

        rhs = np.concatenate([
            -d * K22 * uvt[:N] + utrhs,
            -d * K22 * uvt[N:] + vtrhs
        ])
        
        return rhs