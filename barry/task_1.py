import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from keras.models import Sequential
import tensorflow as tf
from keras.layers import ConvLSTM2D, Conv3D, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras.saving
import glob
import h5py
import IPython
import PIL
import matplotlib.pyplot as plt
import os
import io
import cv2
import IPython.display
import PIL.Image

def task1_normalize_data(data):
  """
    Normalizes the input data by scaling values to the range [0, 1].

    Parameters:
    ----------
        data : np.ndarray
            The input array to be normalized.

    Returns:
    -------
        np.ndarray
            The normalized array where values are scaled to [0,1].
  """
  return (data - data.min()) / (data.max() - data.min() + 1e-6)

def task1_load_multichannel_data(file_path, event_id):
    """
    Reads storm data from an HDF5 file and processes Task 1A and Task 1B.

    Parameters:
    ----------
        file_path : str
            The path to the HDF5 file containing storm event data.
        event_id : str
            The event identifier used to extract relevant data from the file.

    Returns:
    -------
        tuple of np.ndarray
            - resized_vis : np.ndarray
                Resized visible spectrum data with shape (192, 192, T).
            - ir069 : np.ndarray
                Infrared 6.9μm channel data.
            - ir107 : np.ndarray
                Infrared 10.7μm channel data.
            - resized_vil : np.ndarray
                Resized vertical integrated liquid (VIL) data with shape
                (192, 192, T).
    """
    with h5py.File(file_path, "r") as f:
        vil = f[event_id]["vil"][:]
        vis = f[event_id]["vis"][:]
        ir069 = f[event_id]["ir069"][:]
        ir107 = f[event_id]["ir107"][:]

        vil = task1_normalize_data(vil)
        vis = task1_normalize_data(vis)
        ir069 = task1_normalize_data(ir069)
        ir107 = task1_normalize_data(ir107)

        T = vis.shape[2]

        # Resize vis, vil to (192, 192)
        resized_vis = np.array([cv2.resize(vis[:, :, i],
                                             (192, 192)) \
                                 for i in range(T)]).transpose(1, 2, 0)
        resized_vil = np.array([cv2.resize(vil[:, :, i],
                                             (192, 192)) \
                                 for i in range(T)]).transpose(1, 2, 0)

        return resized_vis, ir069, ir107, resized_vil

def task1_preprocess_batch(file_path, event_ids,
                                 save_filename, batch_size=40):
    """
    Processes storm events in batches and saves them in compressed files to
    optimize RAM usage.

    Parameters:
    ----------
        file_path : str
            The path to the HDF5 file containing storm event data.
        event_ids : list of str
            A list of event identifiers to process.
        save_filename : str
            The base filename for saving the processed data.
        batch_size : int, optional
            The number of storms processed and saved in a single batch
            (default: 40).

    Returns:
    -------
        None
    """

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    start_frames = [0, 2, 4, 6, 8, 10, 12]  # 800 * 7
    all_X, all_Y = [], []
    batch_idx = 0

    for i, event_id in enumerate(event_ids):
        try:
            vis, ir069, ir107, vil = \
            task1_load_multichannel_data(file_path, event_id)

            for start in start_frames:
                all_Y.append(vil[:, :, start+12:start+24])  # Output VIL
                all_X.append(np.stack([ir069[:, :, start:start+12], #1
                                       ir107[:, :, start:start+12], #2
                                       vil[:, :, start:start+12], #3
                                       vis[:, :, start:start+12]],  #4
                                       axis=-1))  # 4-Channel Input

        except Exception as e:
            print(f"Error processing {event_id}: {e}")

        # Each 'batch_size' storm is saved once, freeing RAM
        if (i + 1) % batch_size == 0 or (i + 1) == len(event_ids):
          # local save
          np.savez_compressed(f"data/{save_filename}/{save_filename}_batch{batch_idx}.npz",
                              X=np.array(all_X), Y=np.array(all_Y))
          all_X, all_Y = [], []  # Clear RAM and continue processing
          batch_idx += 1

def task1_load_npz_generator(data_dir):
    """
    Lazily loads and yields data samples from compressed `.npz` files
    in a specified directory.

    This approach prevents excessive memory usage by reading data incrementally.

    Parameters:
    ----------
        data_dir : str
            The directory containing `.npz` files with preprocessed storm event
            data.

    Yields:
    -------
        tuple (np.ndarray, np.ndarray)
            - X : np.ndarray
                The input sample with multiple channels.
            - Y : np.ndarray
                The corresponding output sample with an additional
                singleton dimension.
    """

    files = sorted(glob.glob(f"{data_dir}/*.npz"))

    for file in files:
        data = np.load(file)
        X, Y = data["X"], data["Y"]

        for i in range(len(X)):  # Each '.npz 'may contain multiple samples
            yield X[i], np.expand_dims(Y[i], axis=-1)
            # Return samples one by one to avoid full RAM

def task1_get_tf_dataset(task, data_dir, batch_size=1):
    """
    Creates a TensorFlow dataset from a generator, processing storm data for
    Task 1A or Task 1B.

    Parameters:
    ----------
        task : str
            The task identifier ("1A" or "1B").
        data_dir : str
            The directory containing `.npz` files with preprocessed
            storm event data.
        batch_size : int, optional
            The number of samples per batch (default: 1).

    Returns:
    -------
        tf.data.Dataset
            A TensorFlow dataset that yields batches of (X, Y)
            tensors for training or evaluation.
    """

    dataset = tf.data.Dataset.from_generator(
        lambda: task1_load_npz_generator(data_dir),
        output_signature=(
            tf.TensorSpec(shape=(192, 192, 12, 4), dtype=tf.float32),  # X
            tf.TensorSpec(shape=(192, 192, 12, 1), dtype=tf.float32),  # Y
        )
    )

    # only vil
    if task == "1A":
        dataset = dataset.map(lambda X, Y: (X[..., 2:3], Y),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif task == "1B":
        dataset = dataset.map(lambda X, Y: (X, Y),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size).shuffle(1000) \
              .prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

# RMSE
@keras.saving.register_keras_serializable()
def t1_rmse(y_true, y_pred):
    return tf.math.sqrt(tf.reduce_mean(tf.math.square(y_true - y_pred)))

# MAE
@keras.saving.register_keras_serializable()
def t1_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

# KL divergence
@keras.saving.register_keras_serializable()
def t1_kl_divergence(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 1e-10, 1)  # avoid log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1)

    # Make sure that y_true and y_pred are normalized
    y_true = y_true / tf.reduce_sum(y_true, axis=-1, keepdims=True)
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    return tf.reduce_mean(y_true * tf.math.log(y_true / (y_pred + 1e-10)))

# reconstruction loss
@keras.saving.register_keras_serializable()
def t1_reconstruction_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_true - y_pred))

# total loss = reconstruction loss + KL divergence
@keras.saving.register_keras_serializable()
def t1_total_loss(y_true, y_pred):
    return t1_reconstruction_loss(y_true, y_pred) + t1_kl_divergence(y_true, y_pred)

def task1_build_model(input_shape):
    """
    Builds a ConvLSTM2D-based deep learning model for Task 1.

    Parameters:
    ----------
        input_shape : tuple
            The shape of the input tensor (height, width, time steps, channels).

    Returns:
    -------
        model : tf.keras.Model
            A compiled ConvLSTM2D model for storm prediction.
    """

    model = Sequential()

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         input_shape=input_shape))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                         padding='same', return_sequences=True))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                         padding='same', return_sequences=True))

    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid',
                     padding='same'))

    return model

def task1_plot_training_history(history):
    """
    Plots the training loss curves for various metrics.

    Parameters:
    ----------
        history : tf.keras.callbacks.History
            The history object returned by model.fit(),
            containing loss values and metrics.

    Returns:
    -------
        None
    """

    metrics = ['loss', 'mae', 'rmse', 'kl_divergence', 'reconstruction_loss']
    titles = ['Total Loss', 'Mean Absolute Error (MAE)',
              'Root Mean Squared Error (RMSE)',
              'KL Divergence', 'Reconstruction Loss']

    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i + 1)
        plt.plot(history.history[metric], label=f'Train {metric}', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Training {titles[i]}')
        plt.legend()

    plt.tight_layout()
    plt.show()

def task1_evaluate_test_set(model, test_dataset):
    """
    Evaluates the model on a test dataset and computes RMSE, MAE,
    KL-Divergence, Reconstruction Loss, and Total Loss.

    Parameters:
    ----------
        model : tf.keras.Model
            The trained model to be evaluated.
        test_dataset : tf.data.Dataset
            The test dataset containing input-output pairs.

    Returns:
    -------
        None
    """

    total_rmse, total_mae, total_kl, total_recon, total_loss_sum = 0, 0, 0, 0, 0
    num_samples = 0

    for X_batch, Y_batch in test_dataset:
        Y_pred = model.predict(X_batch, verbose=0)

        batch_rmse = tf.sqrt(tf.reduce_mean(tf.square(Y_batch - Y_pred))).numpy()
        batch_mae = tf.reduce_mean(tf.abs(Y_batch - Y_pred)).numpy()

        Y_batch_safe = tf.clip_by_value(Y_batch, 1e-10, 1)
        Y_pred_safe = tf.clip_by_value(Y_pred, 1e-10, 1)
        batch_kl = tf.reduce_mean(Y_batch_safe * tf.math.log(Y_batch_safe / Y_pred_safe)).numpy()

        batch_recon = tf.reduce_mean(tf.square(Y_batch - Y_pred)).numpy()
        batch_total_loss = batch_recon + batch_kl

        total_rmse += batch_rmse
        total_mae += batch_mae
        total_kl += batch_kl
        total_recon += batch_recon
        total_loss_sum += batch_total_loss
        num_samples += 1

    print(f"Test RMSE: {total_rmse / num_samples:.5f}")
    print(f"Test MAE: {total_mae / num_samples:.5f}")
    print(f"Test KL Divergence: {total_kl / num_samples:.5f}")
    print(f"Test Reconstruction Loss: {total_recon / num_samples:.5f}")
    print(f"Test Total Loss: {total_loss_sum / num_samples:.5f}")

def task1_predict_frames(model, X_input):
    """
      Uses the trained model to predict future frames based on input sequences.

      Parameters:
      ----------
          model : tf.keras.Model
              The trained model used for prediction.
          X_input : np.ndarray
              The input sequence of frames.

      Returns:
      -------
          np.ndarray
              The predicted future frames.
    """
    return model.predict(X_input)

def task1_plot_predicted_frames(initial_frames, predicted_frames, actual_frames):
    """
    Plots the input, predicted, and actual frames for visualization.

    Parameters:
    ----------
        initial_frames : np.ndarray
            The input frames used as context for prediction.
        predicted_frames : np.ndarray
            The frames predicted by the model.
        actual_frames : np.ndarray
            The ground truth future frames.

    Returns:
    -------
        None
    """

    num_frames = initial_frames.shape[3]
    plt.figure(figsize=(15, 5))
    for i in range(num_frames):
        plt.subplot(3, num_frames, i+1)
        plt.imshow(initial_frames[0, :, :, i, 0], cmap="turbo")
        plt.title(f"Input {i+1}")
        plt.axis('off')

    for i in range(num_frames):
        plt.subplot(3, num_frames, num_frames+i+1)
        plt.imshow(predicted_frames[0, :, :, i, 0], cmap="turbo")
        plt.title(f"Predicted {i+13}")
        plt.axis('off')

    for i in range(num_frames):
        plt.subplot(3, num_frames, 2*num_frames+i+1)
        plt.imshow(actual_frames[0, :, :, i, 0], cmap="turbo")
        plt.title(f"Actual {i+13}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Prediction
def task1_plot_predicted_frames_b(initial_frames,
                                  predicted_frames, actual_frames):
    num_frames = initial_frames.shape[3]
    plt.figure(figsize=(15, 5))
    for i in range(num_frames):
        plt.subplot(3, num_frames, i+1)
        plt.imshow(initial_frames[0, :, :, i, 2], cmap="turbo")
        plt.title(f"Input {i+1}")
        plt.axis('off')

    for i in range(num_frames):
        plt.subplot(3, num_frames, num_frames+i+1)
        plt.imshow(predicted_frames[0, :, :, i, 0], cmap="turbo")
        plt.title(f"Predicted {i+13}")
        plt.axis('off')

    for i in range(num_frames):
        plt.subplot(3, num_frames, 2*num_frames+i+1)
        plt.imshow(actual_frames[0, :, :, i, 0], cmap="turbo")
        plt.title(f"Actual {i+13}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

