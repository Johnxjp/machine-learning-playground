import pickle
import matplotlib.pyplot as plt
import numpy as np

from typing import Sequence


def unpickle(file):
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    data = {bytes.decode(k): v for k, v in data.items()}
    return data


def load_images(filename: str) -> tuple[Sequence[np.uint8], Sequence[np.uint8]]:
    data = unpickle(filename)
    
    # keys are bytes
    images = np.array(data["data"], dtype=np.uint8)
    images = images.reshape(len(images), 3, 32, 32)
    labels = np.array(data["labels"], dtype=np.uint8)
    return images, labels


def display_image(image: Sequence[np.uint8], label: int, label_names: list[str]) -> None:   
    """ Given an image (C, H, W) plot with title """
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.title(label_names[label])
    plt.plot()
