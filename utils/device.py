import torch


def get_device():
    """
    Returns the device to be used for training.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")