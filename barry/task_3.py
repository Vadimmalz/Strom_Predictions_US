#Placeholder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from common_functions import load_event
import pandas as pd
from torchvision.transforms import Resize
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from IPython.display import display, Image

def get_frame_number(t):
    '''
    This function outputs dpending on the time, 
    which frame does the lightning occur given the 36 frames 
    '''
    if t < 150:
        return 0
    elif t >= 150 and t < 10650:
        return 1 + (t - 150) // 300
    else:
        return 35
    

def safe_normalize(x, eps=1e-6):
    """
    Normalize an image to the range [0, 1].
    """
    x_min = x.min()
    x_max = x.max()

    if x_max == x_min:
        return torch.zeros_like(x) if x_max == 0 else torch.ones_like(x)

    return (x - x_min) / (x_max - x_min + eps)  # Avoid division by very small values


train_transform = Compose([
    Lambda(lambda x: safe_normalize(x))
])


class LightningDataset(Dataset):
    def __init__(self, event_list,project_path, transform=None, target_size=(384, 384), num_frames=3):
        self.event_list = event_list
        self.transform = transform
        self.project_path = project_path
        self.target_size = target_size
        self.data = self._create_dataset()

    def _create_dataset(self):
        dataset = {'ir069': [], 'ir107': [], 'lght': [], 'vis': [], 'vil': []}
        project_path = "/content/drive/My Drive/ACDS-Barry"

        for event_id in self.event_list:
            event = load_event(event_id,project_path) 
            dataset['vis'].append(event['vis'])
            dataset['ir069'].append(event['ir069'])
            dataset['vil'].append(event['vil'])
            dataset['ir107'].append(event['ir107'])
            dataset['lght'].append(event['lght'])
        return dataset

    def __getitem__(self, idx):
        # Select random frame number
        frame_idx = random.randint(1, 36)


        # Load data for the given index
        ir069 = torch.from_numpy(self.data['ir069'][idx])[:, :, frame_idx-1]
        ir107 = torch.from_numpy(self.data['ir107'][idx])[:, :, frame_idx-1]
        vis = torch.from_numpy(self.data['vis'][idx])[:, :, frame_idx-1]
        vil = torch.from_numpy(self.data['vil'][idx])[:, :, frame_idx-1]

        # Upsizes both infra red bands
        resize_transform = Resize(self.target_size)
        ir069_resized = resize_transform(ir069.unsqueeze(0)).squeeze(0)
        ir107_resized = resize_transform(ir107.unsqueeze(0)).squeeze(0)

        if self.transform:
            ir069_resized = self.transform(ir069_resized)
            ir107_resized = self.transform(ir107_resized)
            vis = self.transform(vis)
            vil = self.transform(vil)

        combined_bands = torch.stack([ir069_resized, ir107_resized, vis, vil], dim=0) 

        # Process lightning data
        df = pd.DataFrame(data=self.data["lght"][idx], columns=["t", "lat (deg)", "lon (deg)", "vil pixel x", "vil pixel y"]).sort_values("t")
        df['vil pixel x'] = df['vil pixel x'].astype(int)
        df['vil pixel y'] = df['vil pixel y'].astype(int)
        df['frame_number'] = df['t'].apply(get_frame_number)

        # Filter lightning data for selected frame
        df_frame = df[df['frame_number'] == frame_idx]
        grid_size = 384
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Convert floating points to integer grid indices
        x_indices = df_frame['vil pixel x'].astype(int)
        y_indices = df_frame['vil pixel y'].astype(int)

        # Accumulate counts
        for x, y in zip(x_indices, y_indices):
            grid[y, x] += 1

        lightning_grid = torch.tensor(grid, dtype=torch.float32)

        return combined_bands, lightning_grid

    def __len__(self):
        return len(self.event_list)

def plot_storm_images(images, targets, index=0):
    """
    Plots the four input bands (Visible, Infrared, Radar) and overlays storm locations.

    Parameters:
    - images (tensor or np.array): Input satellite images of shape (batch, channels, height, width).
    - targets (tensor or np.array): Ground truth storm grid of shape (batch, height, width).
    - index (int): Index of the sample to visualize (default: 0).

    Returns:
    - None (displays the figure).
    """

    # Extract storm locations
    storm_y, storm_x = np.where(targets[index] > 0)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    # Visible band
    axs[0].imshow(images[index, 2], cmap="gray")
    axs[0].set_title("Visible")

    # Infrared (Water Vapor)
    axs[1].imshow(images[index, 0], cmap="viridis")
    axs[1].set_title("Infrared (Water Vapor)")

    # Infrared (Cloud/Surface Temperature)
    axs[2].imshow(images[index, 1], cmap="inferno")
    axs[2].set_title("Infrared (Cloud/Surface Temperature)")

    # VIL with storm locations
    axs[3].imshow(images[index, 3], cmap="turbo")
    axs[3].scatter(storm_x, storm_y, color="red", s=2, label="Storm")
    axs[3].set_title("Radar (VIL)")
    axs[3].legend()

    plt.show()

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),  # Upgrading the kernel size yielded to better results
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), 
                nn.ReLU(inplace=True)
            )

        def downsample():
            return nn.MaxPool2d(kernel_size=2, stride=2)

        def upsample(in_c, out_c):
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1)  

        # Encoder (Downsampling)
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = downsample()
        self.enc2 = conv_block(64, 128)
        self.pool2 = downsample()
        self.enc3 = conv_block(128, 256)
        self.pool3 = downsample()
        self.enc4 = conv_block(256, 512)
        self.pool4 = downsample()

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder (Upsampling)
        self.up4 = upsample(1024, 512)
        self.dec4 = conv_block(1024, 512)
        self.up3 = upsample(512, 256)
        self.dec3 = conv_block(512, 256)
        self.up2 = upsample(256, 128)
        self.dec2 = conv_block(256, 128)
        self.up1 = upsample(128, 64)
        self.dec1 = conv_block(128, 64)

        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        # Output layer
        output = F.relu(self.final(d1))  # Relu to assure non-negative values for grid
        return output
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred) 
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice
    
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Trains a UNet model using the given data loaders, loss function, and optimizer.

    Parameters:
    - model (torch.nn.Module): The UNet model.
    - train_loader (DataLoader): Training dataset loader.
    - val_loader (DataLoader): Validation dataset loader.
    - criterion (torch.nn.Module): Loss function (e.g., DiceLoss, MSELoss).
    - optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam, SGD).
    - num_epochs (int): Number of training epochs (default: 5).
    - device (str): Device to use ('cuda' or 'cpu').

    Returns:
    - model (torch.nn.Module): Trained model.
    - train_losses (list): Training loss history.
    - val_losses (list): Validation loss history.
    """

    model.to(device) 
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  
        train_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(device).float(), targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, targets.unsqueeze(1)) 
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device).float(), targets.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, targets.unsqueeze(1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return model, train_losses, val_losses

def plot_model_predictions(model, data_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Plots model predictions vs. ground truth with Radar VIL as the background.

    Parameters:
    - model (torch.nn.Module): Trained model.
    - data_loader (DataLoader): Validation or test dataset loader.
    - threshold (float): Threshold value to binarize predictions.
    - device (str): 'cuda' or 'cpu' (default: auto-detect GPU).

    Returns:
    - None (Displays the plots).
    """
    
    model.eval()  
    model.to(device)
    
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device).float(), targets.to(device).float()

            outputs = model(images)

            predicted_grid = outputs[0, 0].cpu().numpy()
            ground_truth_grid = targets[0].cpu().numpy()

            # Here i used a technique where based on the max value of a grid i multplied by 0.9 in order to get a threshold.
            # Because my model did not predit in the same way as the target grid therefore 1 could not always be the delimeter
            threshold_value = predicted_grid.max() * 0.9
            pred_y, pred_x = np.where(predicted_grid >= threshold_value)  
            gt_y, gt_x = np.where(ground_truth_grid >= 1) 

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # Plot Predicted Lightning on VIL Background
            axs[0].imshow(images[0, 3].cpu().numpy(), cmap="turbo") 
            axs[0].scatter(pred_x, pred_y, color="red", s=80, marker="x", linewidth=2, label="Predicted Storms")
            axs[0].set_title(f"Predicted Lightning Grid (Threshold: {threshold_value:.2f})")
            axs[0].legend()
            axs[0].axis("off")

            # Plot Ground Truth Lightning on VIL Background
            axs[1].imshow(images[0, 3].cpu().numpy(), cmap="turbo")
            axs[1].scatter(gt_x, gt_y, color="blue", s=80, marker="x", linewidth=2, label="Ground Truth Storms")
            axs[1].set_title("Ground Truth Lightning Grid")
            axs[1].legend()
            axs[1].axis("off")

            plt.show()
            break

def get_data_loaders(project_path, batch_size=16, transform=None):
    """
    Returns PyTorch DataLoaders for training, validation, and testing 

    Parameters:
    - project_path (str): Path to the dataset.
    - batch_size (int): Batch size for training and validation.
    - transform: Transformations to apply to input data.

    Returns:
    - train_loader (DataLoader): DataLoader for training set.
    - val_loader (DataLoader): DataLoader for validation set.
    - test_loader (DataLoader): DataLoader for testing set.
    """

    # Load event IDs
    events_df_path = os.path.join(project_path, "events.csv")
    if not os.path.exists(events_df_path):
        raise FileNotFoundError(f"Dataset file not found: {events_df_path}")

    events_df = pd.read_csv(events_df_path)
    list_id = events_df.id.unique().tolist()

    # Arbritary Slicing
    train_ids = list_id[:16]
    val_ids = list_id[150:155]
    test_ids = list_id[-3:]

    train_dataset = LightningDataset(train_ids, project_path, transform=transform)
    val_dataset = LightningDataset(val_ids, project_path, transform=transform)
    test_dataset = LightningDataset(test_ids, project_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



class LightningDatasetClass(Dataset):
    '''
    Dataclass set use for the test -> surprise storms
    '''
    def __init__(self, event,frame_idx, project_path, transform=None, target_size=(384, 384)):
        self.event = event
        self.transform = transform
        self.project_path = project_path
        self.target_size = target_size
        self.frame_idx = frame_idx

    def __getitem__(self, idx):
        ir069 = torch.from_numpy(self.event['ir069'])[:, :, self.frame_idx-1]
        ir107 = torch.from_numpy(self.event['ir107'])[:, :, self.frame_idx-1]
        vis = torch.from_numpy(self.event['vis'])[:, :, self.frame_idx-1]
        vil = torch.from_numpy(self.event['vil'])[:, :, self.frame_idx-1]

        resize_transform = Resize(self.target_size)
        ir069_resized = resize_transform(ir069.unsqueeze(0)).squeeze(0)
        ir107_resized = resize_transform(ir107.unsqueeze(0)).squeeze(0)

        if self.transform:
            ir069_resized = self.transform(ir069_resized)
            ir107_resized = self.transform(ir107_resized)
            vis = self.transform(vis)
            vil = self.transform(vil)

        combined_bands = torch.stack([ir069_resized, ir107_resized, vis, vil], dim=0)
        return combined_bands

    def __len__(self):
        return 1
    

def get_data_loader(project_path, event, frame_idx, batch_size=1, transform=None):
    """
    Returns PyTorch DataLoaders for training, validation, and testing 
    with the same dataset slicing as originally used.

    Parameters:
    - project_path (str): Path to the dataset.
    - batch_size (int): Batch size for training and validation.
    - transform: Transformations to apply to input data.

    Returns:
    - train_loader (DataLoader): DataLoader for training set.
    - val_loader (DataLoader): DataLoader for validation set.
    - test_loader (DataLoader): DataLoader for testing set.
    """

    # Create dataset instances
    dataset = LightningDatasetClass(event, project_path, transform=train_transform , frame_idx=frame_idx)

    # Create data loaders
    loader = DataLoader(dataset, batch_size= batch_size, shuffle=True)


    return loader

def plot_model_predictions_and_extract_events(model, event, project_path, transform=None):
    """
    Runs inference for all frame indices (0 to 35), plots predictions in a 6x6 grid 
    with Radar (VIL) as the background and detected lightning events overlaid as large crosses,
    and extracts lightning events into an (N,3) DataFrame.

    Parameters:
    - model (torch.nn.Module): The trained UNet model.
    - event (str): The event ID to load.
    - project_path (str): Path to the dataset.
    - transform: Transformations to apply.

    Returns:
    - df (pd.DataFrame): DataFrame containing detected lightning events with columns (t, x, y).
    """
    
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fig, axs = plt.subplots(6, 6, figsize=(18, 18))  # 6x6 grid for 36 frames
    lightning_data = []  # Store extracted (t, x, y) points

    for frame_idx in range(36):  
        dataset = LightningDatasetClass(event, frame_idx=frame_idx, project_path=project_path, transform=transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)  
        images = next(iter(loader)).to(device).float()

        with torch.no_grad():
            outputs = model(images)

        predicted_grid = outputs[0, 0].cpu().numpy()

        # Thresholding: same technique as above
        threshold = predicted_grid.max() * 0.9  
        y_coords, x_coords = np.where(predicted_grid >= threshold) 

        for y, x in zip(y_coords, x_coords):
            lightning_data.append([frame_idx, x, y])  # Frame index, x-coord, y-coord

        #  VIL 
        ax = axs[frame_idx // 6, frame_idx % 6]  
        ax.imshow(images[0, 3].cpu().numpy(), cmap="turbo")  
        
        # Lightning Strikes
        ax.scatter(
            x_coords, y_coords, color="red", s=80, marker="x", linewidth=2, label="Storm"
        )  

        ax.set_title(f"Frame {frame_idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(lightning_data, columns=["t", "x", "y"])

    return df


def interpolate_lightning_times(df):
    """
    Interpolates time values for lightning events within each frame's time window.

    Parameters:
    - df (pd.DataFrame): DataFrame containing detected lightning events with columns (t, x, y).

    Returns:
    - df (pd.DataFrame): DataFrame with interpolated time values in a new column `time_sec`.
    """

    # Define constants
    FRAME_DURATION = 300 
    HALF_WINDOW = 150  
    TOTAL_DURATION = 10800 
    MAX_FRAME = 35 

    df["time_sec"] = 0
    for t in df["t"].unique():
        mask = df["t"] == t
        num_events = mask.sum()  

        if t == 0:  
            start_time = 0
            end_time = 150
        elif t == MAX_FRAME:  
            start_time = 10650
            end_time = 10800
        else:  
            start_time = t * FRAME_DURATION - HALF_WINDOW
            end_time = t * FRAME_DURATION + HALF_WINDOW

        if num_events > 0:
            df.loc[mask, "time_sec"] = np.linspace(start_time, end_time, num_events)

    return df[['time_sec','x','y']]


def create_gif_model_predictions(model, event, project_path, transform=None):
    """
    Runs inference for all frame indices (0 to 35) and creates an animated GIF that loops indefinitely (No Save).

    Parameters:
    - model (torch.nn.Module): The trained UNet model.
    - event (str): The event ID to load.
    - project_path (str): Path to the dataset.
    - transform: Transformations to apply.

    Returns:
    - None (Displays looping GIF without saving).
    """

    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    frames = []  # Store generated images for GIF

    for frame_idx in range(36):  
        dataset = LightningDatasetClass(event, frame_idx=frame_idx, project_path=project_path, transform=transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)  

        images = next(iter(loader)).to(device).float()

        with torch.no_grad():
            outputs = model(images)

        predicted_grid = outputs[0, 0].cpu().numpy()

        threshold = predicted_grid.max() * 0.9 
        y_coords, x_coords = np.where(predicted_grid >= threshold)  

        fig, ax = plt.subplots(figsize=(6,6))

        ax.imshow(images[0, 3].cpu().numpy(), cmap="turbo") 
        ax.scatter(
            x_coords, y_coords, color="red", s=80, marker="x", linewidth=2, label="Storm"
        )  

        ax.set_title(f"Frame {frame_idx}")
        ax.axis("off")

        # Save frame as an image (temporarily)
        frame_path = f"frame_{frame_idx}.png"
        plt.savefig(frame_path, bbox_inches="tight")
        plt.close(fig)
        
        # Append to frames list
        frames.append(imageio.imread(frame_path))

    # Create GIF in memory (NO SAVE)
    gif_path = "/tmp/storm_animation.gif"
    imageio.mimsave(gif_path, frames, duration=0.3, loop=0)  # GIF loops forever

    # Cleanup temporary files
    for frame_idx in range(36):
        os.remove(f"frame_{frame_idx}.png")

    display(Image(filename=gif_path))


