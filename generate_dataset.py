import yaml
from src.GenerateDataset import GenerateGym, GeneratePDE

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        args = yaml.safe_load(file)

    env_name = args["env_name"]

    # Generate Gym dataset (either Pendulum, Acrobot, or MountainCarContinuous)
    if env_name in ("Pendulum", "Acrobot", "MountainCarContinuous"):
        seed = args["seed"]
        
        # Training set
        NewTrainSet = GenerateGym(args=args, seed=seed, test=False, png=False)
        NewTrainSet.generate_dataset()

        # Test set
        NewTestSet = GenerateGym(args=args, seed=seed+1, test=True, png=False)
        NewTestSet.generate_dataset()

    # Generate PDE dataset (reaction-diffusion)
    elif env_name in ("reaction-diffusion", "diffusion-advection"):
        NewDataset = GeneratePDE(args)
        NewDataset.generate_dataset()

    # No other test case supported (yet)
    else:
        raise ValueError("Invalid environment name. Test case not supported.")
    
