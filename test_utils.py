import os
import numpy as np
import matplotlib.pyplot as plt


def plot_reconstruction(real_frames, images, i, T_start, dt, mu, filepath):
    t = T_start + round(i*dt, 2)
    t_next = T_start + round((i+1)*dt, 2)

    # Generate a grid of 1x4 subplots
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    # Compute MSE
    mse = np.mean((images[0] - real_frames[0]) ** 2)
    mse_next = np.mean((images[2] - real_frames[1]) ** 2)

    # Reference frame
    axs[0].imshow(real_frames[0])
    axs[0].set_title('t = ' + str(t))

    # First subplot
    axs[1].imshow(images[0])
    axs[1].set_title('t = ' + str(t))

    # Second subplot
    axs[2].imshow(images[1])
    axs[2].set_title('(MSE = ' + str(round(mse, 4)) + ')')

    # Third subplot
    axs[3].imshow(images[2])
    axs[3].set_title('t+dt = ' + str(t_next))

    # Fourth subplot
    axs[4].imshow(images[3])
    axs[4].set_title('(MSE = ' + str(round(mse_next, 4)) + ')')

    # Add a title
    # fig.suptitle(r'$\mu$ = ' + str(mu), fontsize=16)
    # fig.text(0.25, 0.95, 'Reconstruction', ha='center', va='center', fontsize=14)
    # fig.text(0.75, 0.95, 'Error', ha='center', va='center', fontsize=14)

    # Save the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    filename = filepath + str(i) + ".svg"
    plt.savefig(filename, format="svg")
    plt.close()

    return mse, mse_next


def reconstruct(model, input_data, mu_x, mu_next, z_fwd_LF, L_test, T_start, dt, start, end, mu_value, filepath):
    mse = []
    mse_next = []

    for i in range(start,end):
        mu_x_rec, _, _, _ = model.predict_dynamics_mean(mu_next[i].unsqueeze(dim=0), z_fwd_LF[i])
        mu_x_rec = mu_x_rec.permute(0, 2, 3, 1) # move color channel to the end

        # Set the images
        frame = input_data[i, :, :, 0:3].detach().numpy()
        frame_next = input_data[i+1, :, :, 0:3].detach().numpy()
        image1 = mu_x[i, :, :, 0:3]
        image2 = 1 - np.abs(image1 - frame)
        image3 = mu_x_rec[0, :, :, 0:3].detach().numpy()
        image4 = 1 - np.abs(image3 - frame_next)

        mse_new, mse_next_new = plot_reconstruction([frame, frame_next], 
                                                    [image1, image2, image3, image4], 
                                                    i%L_test, T_start, dt, mu_value, filepath)
        mse.append(mse_new)
        mse_next.append(mse_next_new)

    return mse, mse_next


def extrapolate(model, input_data, mu_fwd_init, z_fwd_LF, L_test, T_start, dt, filepath, N_iter=20, start=0):
    mse_extra = []
    mu_fwd = mu_fwd_init.unsqueeze(dim=0)
    if not os.path.exists(filepath + "extrapolation/"):
        os.makedirs(filepath + "extrapolation/")
    
    for i in range(start, start+N_iter):
        mu_x_rec, _, mu_fwd, _ = model.predict_dynamics_mean(mu_fwd, z_fwd_LF[i])
        mu_x_rec = mu_x_rec.permute(0, 2, 3, 1)
        mu_x_rec = mu_x_rec.detach().numpy()

        frame = input_data[i+1, :, :, 0:3].detach().numpy()
        image1 = mu_x_rec[:, :, :, 0:3].squeeze(0)
        image2 = 1 - np.abs(image1 - frame)

        # Generate a grid of 1x2 subplots
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        t = T_start + round((i%L_test)*dt, 2)

        # Compute MSE
        mse = np.mean((image1 - frame) ** 2)
        mse_extra.append(mse)

        # Reference frame
        axs[0].imshow(frame)
        axs[0].set_title('t = ' + str(t))

        # First subplot
        axs[1].imshow(image1)
        axs[1].set_title('t = ' + str(t))

        # Second subplot
        axs[2].imshow(image2)
        axs[2].set_title('(MSE = ' + str(round(mse, 4)) + ')')

        # Save the image
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        filename = filepath + "extrapolation/" + str(i%L_test) + ".svg"
        plt.savefig(filename, format="svg")
        plt.close()

    return mse_extra
