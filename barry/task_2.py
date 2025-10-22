import os
import PIL
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from IPython import display
from torch.utils.data import Dataset, DataLoader


class Task2DataPreparer:
    """
    A class for preparing event-based image datasets for machine learning models.
    
    This class provides functionality to:
    - Load event IDs and event data from an HDF5 file.
    - Compute normalization parameters (min-max scaling) for images.
    - Resize and normalize image data.
    - Split the dataset into training, validation, and test sets.
    - Create PyTorch data loaders.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file containing the event data.
    test_size : float, optional
        Proportion of the dataset to be used for testing. Default is 0.2.
    val_size : float, optional
        Proportion of the dataset to be used for validation. Default is 0.1.
    sample_ratio : float, optional
        Ratio of events to be sampled from the dataset. Default is 0.5.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    is_real_data : bool, optional
        Flag to indicate if the data is real data (no VIL channel). Default is False.

    Attributes
    ----------
    norm_params : dict or None
        Dictionary storing min-max normalization parameters for each image type.
        Computed using the training dataset.
    """

    def __init__(self, file_path, test_size=0.2, val_size=0.1, sample_ratio=0.5, random_state=42, is_real_data=False):
        self.file_path = file_path
        self.test_size = test_size
        self.val_size = val_size
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.is_real_data = is_real_data
        self.norm_params = None  # Will be computed later

    def load_event_ids(self):
        """
        Load all event IDs from the HDF5 file.

        Returns
        -------
        list of str
            A list of event IDs available in the dataset.
        """
        with h5py.File(self.file_path, 'r') as f:
            event_ids = list(f.keys())  # Extract event IDs from the file
        return event_ids

    def load_event(self, event_id):
        """
        Load the image data of a specific event.

        Parameters
        ----------
        event_id : str
            The ID of the event to be loaded.

        Returns
        -------
        dict
            Dictionary containing image arrays for different channels:
            {'vis': array, 'ir069': array, 'ir107': array, 'vil': array}.
        """
        with h5py.File(self.file_path, 'r') as f:
            if self.is_real_data:
                event = {img_type: f[event_id][img_type][:] for img_type in ['vis', 'ir069', 'ir107']}
            else:
                event = {img_type: f[event_id][img_type][:] for img_type in ['vis', 'ir069', 'ir107', 'vil']}
        return event

    def compute_normalization_params(self, event_ids):
        """
        Compute min-max normalization parameters for each image type.

        Parameters
        ----------
        event_ids : list of str
            List of event IDs used to compute normalization parameters.
        """
        # Initialize min and max values for each image type
        vis_min, vis_max = float('inf'), float('-inf')
        ir069_min, ir069_max = float('inf'), float('-inf')
        ir107_min, ir107_max = float('inf'), float('-inf')
        vil_min, vil_max = float('inf'), float('-inf')

        # Iterate over events to compute min and max values
        for event_id in event_ids:
            event = self.load_event(event_id)
            vis, ir069, ir107 = event['vis'], event['ir069'], event['ir107']

            vis_min, vis_max = min(vis_min, vis.min()), max(vis_max, vis.max())
            ir069_min, ir069_max = min(ir069_min, ir069.min()), max(ir069_max, ir069.max())
            ir107_min, ir107_max = min(ir107_min, ir107.min()), max(ir107_max, ir107.max())

            if not self.is_real_data:
                vil = event['vil']
                vil_min, vil_max = min(vil_min, vil.min()), max(vil_max, vil.max())

        # Store normalization parameters
        self.norm_params = {
            'vis': (vis_min, vis_max),
            'ir069': (ir069_min, ir069_max),
            'ir107': (ir107_min, ir107_max)
        }

        if not self.is_real_data:
            self.norm_params['vil'] = (vil_min, vil_max)

    def adjust_image_size(self, event):
        """
        Resize and normalize the input images.

        Parameters
        ----------
        event : dict
            Dictionary containing the original image data for an event.

        Returns
        -------
        tuple of numpy.ndarray
            Resized and normalized images for VIS, IR069, IR107, and VIL (if applicable).
        """
        target_shape = (192, 192)  # Target image size

        # Extract image channels
        X_vis, X_ir069, X_ir107 = event['vis'], event['ir069'], event['ir107']

        # Resize VIS images if needed
        if X_vis.shape[:2] != target_shape:
            X_vis_resized = np.stack([resize(X_vis[:, :, t], target_shape, mode='reflect',
                                             preserve_range=True, anti_aliasing=True) for t in range(X_vis.shape[2])], axis=-1)
        else:
            X_vis_resized = X_vis

        # Normalize images using precomputed min-max values
        X_vis_resized = (X_vis_resized - self.norm_params['vis'][0]) / (self.norm_params['vis'][1] - self.norm_params['vis'][0])
        X_ir069 = (X_ir069 - self.norm_params['ir069'][0]) / (self.norm_params['ir069'][1] - self.norm_params['ir069'][0])
        X_ir107 = (X_ir107 - self.norm_params['ir107'][0]) / (self.norm_params['ir107'][1] - self.norm_params['ir107'][0])

        if self.is_real_data:
            return X_vis_resized, X_ir069, X_ir107
        else:
            y_vil = event['vil']
            # Resize VIL images if needed
            if y_vil.shape[:2] != target_shape:
                y_vil_resized = np.stack([resize(y_vil[:, :, t], target_shape, mode='reflect',
                                                  preserve_range=True, anti_aliasing=True) for t in range(y_vil.shape[2])], axis=-1)
            else:
                y_vil_resized = y_vil

            # Normalize VIL
            y_vil_resized = (y_vil_resized - self.norm_params['vil'][0]) / (self.norm_params['vil'][1] - self.norm_params['vil'][0])

            return X_vis_resized, X_ir069, X_ir107, y_vil_resized

    def prepare_datasets(self):
        """
        Prepare training, validation, and test datasets.

        Returns
        -------
        tuple
            Numpy arrays for training, validation, and test sets, along with normalization parameters.
        """
        event_ids = self.load_event_ids()

        # Randomly sample event IDs
        selected_ids = np.random.choice(event_ids, size=int(len(event_ids) * self.sample_ratio), replace=False)

        # Split dataset into train, validation, and test sets
        train_ids, test_ids = train_test_split(selected_ids, test_size=self.test_size, random_state=self.random_state)
        train_ids, val_ids = train_test_split(train_ids, test_size=self.val_size / (1 - self.test_size),
                                              random_state=self.random_state)

        # Compute normalization parameters using training data
        self.compute_normalization_params(train_ids)

        X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

        # Load and preprocess data
        for dataset, event_list in zip([(X_train, y_train), (X_val, y_val), (X_test, y_test)], 
                                        [train_ids, val_ids, test_ids]):
            for event_id in event_list:
                event = self.load_event(event_id)
                if self.is_real_data:
                    X_vis, X_ir069, X_ir107 = self.adjust_image_size(event)
                    for t in range(X_vis.shape[2]):
                        dataset[0].append(np.stack([X_vis[:, :, t], X_ir069[:, :, t], X_ir107[:, :, t]], axis=-1))
                else:
                    X_vis, X_ir069, X_ir107, y_vil = self.adjust_image_size(event)
                    for t in range(X_vis.shape[2]):
                        dataset[0].append(np.stack([X_vis[:, :, t], X_ir069[:, :, t], X_ir107[:, :, t]], axis=-1))
                        dataset[1].append(y_vil[:, :, t])

        if self.is_real_data:
            return np.array(X_train), np.array(X_val), np.array(X_test), self.norm_params
        else:
            return (np.array(X_train), np.array(y_train),
                    np.array(X_val), np.array(y_val),
                    np.array(X_test), np.array(y_test),
                    self.norm_params)
    
    def prepare_surprise_datasets(self):
        """
        Prepare the surprise dataset.

        Returns
        -------
        tuple
            Numpy arrays for the full dataset along with normalization parameters.
        """
        event_ids = self.load_event_ids()

        # Compute normalization parameters using the full dataset
        self.compute_normalization_params(event_ids)

        X_data = []

        # Load and preprocess data
        for event_id in event_ids:
            event = self.load_event(event_id)
            X_vis, X_ir069, X_ir107 = self.adjust_image_size(event)
            for t in range(X_vis.shape[2]):
                X_data.append(np.stack([X_vis[:, :, t], X_ir069[:, :, t], X_ir107[:, :, t]], axis=-1))

        return np.array(X_data), self.norm_params

    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128, num_workers=8):
        """
        Create PyTorch data loaders for training, validation, and test datasets.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray or None
            Training labels.
        X_val : numpy.ndarray
            Validation data.
        y_val : numpy.ndarray or None
            Validation labels.
        X_test : numpy.ndarray
            Test data.
        y_test : numpy.ndarray or None
            Test labels.
        batch_size : int, optional
            Batch size for data loaders. Default is 128.
        num_workers : int, optional
            Number of workers for data loaders. Default is 8.

        Returns
        -------
        tuple
            Data loaders for training, validation, and test datasets.
        """
        train_dataset = self.WeatherDataset(X_train, y_train)
        val_dataset = self.WeatherDataset(X_val, y_val)
        test_dataset = self.WeatherDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    class WeatherDataset(Dataset):
        def __init__(self, X, y=None):
            self.X = np.array(X, dtype=np.float32)  # Ensure X is a numpy array of type float32
            self.y = np.array(y, dtype=np.float32) if y is not None else None  # Ensure y is a numpy array of type float32

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            # Change the order of dimension (H, W, C) -> (C, H, W)
            x = torch.tensor(self.X[idx], dtype=torch.float32).permute(2, 0, 1)
            if self.y is not None:
                y = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)  # added channel dimension
                return x, y
            else:
                return x


class Task2UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Task2UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )

        # encoder
        self.enc1 = conv_block(input_channels, 64)
        self.enc2 = conv_block(64, 128)

        # bottleneck
        self.bottleneck = conv_block(128, 256)

        # decoder
        self.up2 = up_block(256, 128)
        self.dec2 = conv_block(128 + 128, 128)
        self.up1 = up_block(128, 64)
        self.dec1 = conv_block(64 + 64, 64)

        # final Output
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))

        dec2 = self.dec2(torch.cat([self.up2(bottleneck), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return self.final(dec1)


class Task2DataVisualization:
    """
    A class containing helper functions for visualizing and processing data.
    """

    @staticmethod
    def make_gif(outfile, files, fps=10, loop=0):
        """
        Create and save a GIF from a list of image files.

        Parameters
        ----------
        outfile : str
            Path to the output GIF file.
        files : list
            List of image file paths.
        fps : int
            Frames per second (default: 10).
        loop : int
            Number of times to loop the GIF (default: 0 for infinite loop).

        Returns
        -------
        IPython.display.Image
            The generated GIF for display in a Jupyter notebook.
        """
        imgs = [PIL.Image.open(file) for file in files]
        imgs[0].save(fp=outfile, format='gif', append_images=imgs[1:],
                     save_all=True, duration=int(1000/fps), loop=loop)
        im = display.Image(filename=outfile)
        im.reload()
        return im

    @staticmethod
    def plot_event(data_loader, model, output_gif=False, save_gif=False):
        """
        Plot event frames and optionally save them as a GIF.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            PyTorch DataLoader containing the dataset.
        model: nn.Module
            PyTorch model used for prediction.
        output_gif : bool
            Whether to generate GIFs (default: False).
        save_gif : bool
            Whether to save GIFs to disk (default: False).
        """
        def plot_frame(index):
            channels = data_loader.dataset.__getitem__(index)
            channels = channels.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                result = model(channels).squeeze(0).squeeze(0)

            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(f"Frame: {index}")

            axs[0].imshow(channels[0, 0], cmap="gray"), axs[0].set_title('Channel 1')
            axs[1].imshow(channels[0, 1], cmap="viridis"), axs[1].set_title('Channel 2')
            axs[2].imshow(channels[0, 2], cmap="inferno"), axs[2].set_title('Channel 3')
            axs[3].imshow(result, cmap="turbo"), axs[3].set_title('Predicted Result')

            if output_gif:
                file = f"_temp_{index}.png"
                fig.savefig(file, bbox_inches="tight", dpi=150, pad_inches=0.02, facecolor="white")
                plt.close()
            else:
                plt.show()

        if output_gif:
            num_frames = len(data_loader.dataset)
            num_gifs = num_frames // 36

            for gif_index in range(num_gifs):
                files = []
                for frame_index in range(36):
                    index = gif_index * 36 + frame_index
                    plot_frame(index)
                    files.append(f"_temp_{index}.png")
                im = Task2DataVisualization.make_gif(f"output_{gif_index}.gif", files)
                for file in files:
                    os.remove(file)
                display.display(im)
                if not save_gif:
                    os.remove(f"output_{gif_index}.gif")
        else:
            for index in range(0, len(data_loader.dataset), 36):
                plot_frame(index)

    @staticmethod
    def resize_image(image, size=(384, 384)):
        """
        Resize an image to a target size using bilinear interpolation.

        Parameters
        ----------
        image : numpy.ndarray or tensor
            Input image.
        size : tuple
            Target size (default: (384, 384)).

        Returns
        -------
        numpy.ndarray
            Resized image.
        """
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
        resized_image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
        return resized_image.squeeze().numpy()

    @staticmethod
    def denormalize(image, norm_params):
        """
        Denormalize an image using given normalization parameters.

        Parameters
        ----------
        image : numpy.ndarray
            Normalized image.
        norm_params : tuple
            (min_val, max_val) used for normalization.

        Returns
        -------
        numpy.ndarray
            Denormalized image.
        """
        min_val, max_val = norm_params
        return image * (max_val - min_val) + min_val

    @staticmethod
    def render_npy_file(npy_file_path):
        """
        Load and visualize frames from a .npy file.

        Parameters
        ----------
        npy_file_path : str
            Path to the .npy file.
        """
        predictions = np.load(npy_file_path)
        height, width, num_frames = predictions.shape
        num_frames_to_plot = min(num_frames, 6)

        fig, axs = plt.subplots(1, num_frames_to_plot, figsize=(15, 5))
        fig.suptitle(f"Predictions from {npy_file_path}", fontsize=16)

        for i in range(num_frames_to_plot):
            axs[i].imshow(predictions[:, :, i], cmap='turbo')
            axs[i].set_title(f"Frame {i+1}")
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def render_first_channel_from_h5(h5_file_path, key):
        """
        Load and visualize the first channel of an image stored in an HDF5 file.

        Parameters
        ----------
        h5_file_path : str
            Path to the HDF5 file.
        key : str
            Dataset key to access.
        """
        with h5py.File(h5_file_path, 'r') as f:
            data = f[key]
            first_channel = data['vis']  # Assuming 'vis' is the first channel
            height, width, num_frames = first_channel.shape
            num_frames_to_plot = min(num_frames, 6)

            fig, axs = plt.subplots(1, num_frames_to_plot, figsize=(15, 5))
            fig.suptitle(f"First Channel from {key} in {h5_file_path}", fontsize=16)

            for i in range(num_frames_to_plot):
                axs[i].imshow(first_channel[:, :, i], cmap='viridis')
                axs[i].set_title(f"Frame {i+1}")
                axs[i].axis('off')

            plt.tight_layout()
            plt.show()
