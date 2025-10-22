import torch
import h5py
import pandas as pd
import PIL
import IPython
import matplotlib.pyplot as plt
import random
import os
import torch.nn.functional as F


def set_device(device="cuda", idx=0):
    """
    (For use in Google Colab)
    Automatically selects the best available device (GPU/CPU).

    Parameters:
    ----------
    device : str, optional
        Desired device ("cuda" or "cpu"), default is "cuda".
    idx : int, optional
        GPU index to use (default: 0).

    Returns:
    ----------
    str
        The best available device string (e.g., "cuda:0" or "cpu").
    """
    if device != "cpu":
        if torch.cuda.is_available() and torch.cuda.device_count() > idx:
            print(f"CUDA installed! Running on GPU {idx}: {torch.cuda.get_device_name(idx)}")
            device = f"cuda:{idx}"
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print(f"CUDA installed but only {torch.cuda.device_count()} GPU(s) available! Running on GPU 0: {torch.cuda.get_device_name(0)}")
            device = "cuda:0"
        else:
            print("No GPU available! Running on CPU.")
            device = "cpu"
    return device


def load_event(id, project_path) :
    """
    Loads image data for a specific event from an HDF5 file.

    Parameters
    ----------
    id : str
        The identifier of the event to load. This is used to access the specific event within the HDF5 file.
    project_path : str
        The path to the project directory.

    Returns
    -------
    event : dict
        A dictionary mapping image types to their corresponding image data for the given event.
        The keys are:
        - 'vis' : Visible spectrum image data.
        - 'ir069' : Infrared image data at wavelength 069.
        - 'ir107' : Infrared image data at wavelength 107.
        - 'vil' : Visible infrared loop image data.
        - 'lght' : Lightning data.
    """
    with h5py.File(project_path + f'/data/train.h5','r') as f:
        event = {img_type: f[id][img_type][:] for img_type in ['vis', 'ir069', 'ir107', 'vil', 'lght']}
    return event


# Function to calculate basic statistics about the values present in each of the image types
def calculate_image_stats(dict_list, image_types):
    """
    Calculate statistics for the spatial dimensions (192x192 or 384x384) across all channels.

    Parameters
    ----------
        dict_list : list
            List of dictionaries containing tensors.
        image_types : list of str
            Names of tensor dictionaries (e.g., ['ir069', 'ir107', ...]).

    Returns:
    ----------
        pd.DataFrame:
            A DataFrame with spatial statistics for each tensor dictionary.
    """
    # initializing a list to store statistic values in
    stats_values = []

    for dict_name, tensor_dict in zip(image_types, dict_list):
        # Initialize cumulative statistics
        global_min = float('inf')
        global_max = float('-inf')
        total_mean = 0
        total_median = 0
        count = 0
        total_std = 0

        for tensor in tensor_dict.values():
            # Remove the batch dimension (shape becomes [192, 192, 36] or [384, 384, 36])
            tensor = tensor.squeeze(0)

            # Reshaping the 3D tensor into a 2D channel -> appending channel dimension values along the 2nd dimension
            if tensor.shape == (192, 192, 36):
                flattened_tensor = tensor.permute(0, 2, 1).reshape(192, 192 * 36)
            else:
                flattened_tensor = tensor.permute(0, 2, 1).reshape(384, 384 * 36)

            # Update global statistics
            global_min = min(global_min, torch.min(flattened_tensor).item())
            global_max = max(global_max, torch.max(flattened_tensor).item())
            total_mean += torch.mean(flattened_tensor).item()
            total_median += torch.median(flattened_tensor).item()
            total_std += torch.std(flattened_tensor).item()

            count += 1

        # Average the statistics over all tensors
        row = {'image type': dict_name}
        row['XY Value Min'] = global_min
        row['XY Value Max'] = global_max
        row['XY Value Mean'] = total_mean / count if count > 0 else float('nan')
        row['XY Value Median'] = total_median / count if count > 0 else float('nan')
        row['XY Value Std dev'] = total_std / count if count > 0 else float('nan')
        stats_values.append(row)

    return pd.DataFrame(stats_values)

# these functions will plot an event
def make_gif(outfile, files, fps=10, loop=0):
    """
    Helper function for saving GIFs

    Parameters
    ----------
    outfile : str
        The name of the output file.
    files : list
        List of file names. 
    fps : int, optional
        Frames per second (default is 10).
    loop : int, optional
        Number of loops (0 for infinite loop, default is 0).
    
    Returns
    -------
    IPython.display.Image
        An IPython display object representing the GIF.

    """
    imgs = [PIL.Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='gif', append_images=imgs[1:],
                 save_all=True, duration=int(1000/fps), loop=loop)
    im = IPython.display.Image(filename=outfile)
    im.reload()
    return im

def plot_event(id, project_path, output_gif=False, save_gif=False):
    """
    Helper function for plotting an event

    Parameters
    ----------
    id : str
        The identifier of the event to plot.
    project_path : str
        The path to the project directory.
    output_gif : bool, optional
        Whether to output a GIF of the event (default is False).
    save_gif : bool, optional
        Whether to save the GIF file (default is False).

    Returns
    -------
    None
    """
    event = load_event(id, project_path)
    t = event["lght"][:,0]# time of lightning strike (in seconds relative to first frame)
    def plot_frame(ti):
        f = (t >= ti*5*60 - 2.5*60) & (t < ti*5*60 + 2.5*60)# find which lightning strikes fall in current frame
        fig,axs = plt.subplots(1,4,figsize=(16,4))
        fig.suptitle(f"Event: {id}, Frame: {ti}, Time: {ti*5} min")
        axs[0].imshow(event["vis"][:,:,ti], vmin=0, vmax=10000, cmap="grey"), axs[0].set_title('Visible')
        axs[1].imshow(event["ir069"][:,:,ti], vmin=-8000, vmax=-1000, cmap="viridis"), axs[1].set_title('Infrared (Water Vapor)')
        axs[2].imshow(event["ir107"][:,:,ti], vmin=-7000, vmax=2000, cmap="inferno"), axs[2].set_title('Infrared (Cloud/Surface Temperature)')
        axs[3].imshow(event["vil"][:,:,ti], vmin=0, vmax=255, cmap="turbo"), axs[3].set_title('Radar (Vertically Integrated Liquid)')
        axs[3].scatter(event["lght"][f,3], event["lght"][f,4], marker="x", s=30, c="tab:red")
        axs[3].set_xlim(0,384), axs[3].set_ylim(384,0)
        if output_gif:
            file = f"_temp_{id}_{ti}.png"
            fig.savefig(file, bbox_inches="tight", dpi=150, pad_inches=0.02, facecolor="white")
            plt.close()
        else:
            plt.show()
    if output_gif:
        for ti in range(36): plot_frame(ti)
        im = make_gif(f"{id}.gif", [f"_temp_{id}_{ti}.png" for ti in range(36)])
        for ti in range(36): os.remove(f"_temp_{id}_{ti}.png")
        IPython.display.display(im)
        if not save_gif: os.remove(f"{id}.gif")
    else:
        plot_frame(0)
        plot_frame(17)
        plot_frame(34)