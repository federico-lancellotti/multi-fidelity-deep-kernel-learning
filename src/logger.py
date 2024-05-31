import pickle
import os


class Logger:
    """
    A class for logging the generated data and saving it to a file, as a pickle object.

    Attributes:
        directory (str): The directory where the log file will be saved.
        filename (str): The name of the log file.
        datalog (list): A list to store the logged data.

    Methods:
        obslog(data): Logs the given data.
        save_obslog(filename, folder): Saves the logged data to a file, as a pickle object.
    """

    def __init__(self, folder, filename="dataset.pkl"):
        """
        Initializes a Logger object.

        Args:
            folder (str): The directory where the log file will be saved.
            filename (str, optional): The name of the log file. Defaults to "dataset.pkl".
        """

        self.directory = folder
        self.filename = filename
        # make sure the folder exists
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.datalog = []


    def obslog(self, data):
        """
        Logs the given data.

        Args:
            data: The data to be logged.
        """

        self.datalog.append(data)


    def save_obslog(self, filename="dataset.pkl", folder=""):
        """
        Saves the logged data to a file, as a pickle object.

        Args:
            filename (str, optional): The name of the log file. Defaults to "dataset.pkl".
            folder (str, optional): The directory where the log file will be saved. Defaults to "" (the directory where the log file was initialized).
        """

        if folder == "":
            folder = self.directory
        with open(folder + filename, "wb") as f:
            pickle.dump(self.datalog, f, pickle.HIGHEST_PROTOCOL)
