"""Script that converts original data into single numpy file for training and test"""

import numpy as np

from utils import load_images


def repackage_files(dirname: str, filenames: list[str], savefile: str):
    """Loads and joins data from save files into single numpy file"""

    all_images, all_labels = [], []
    for f in filenames:
        images, labels = load_images(dirname + "/" + f)
        all_images.extend(images)
        all_labels.extend(labels)

    all_images = np.array(all_images, dtype=np.uint8).reshape(-1, 3, 32, 32)
    all_labels = np.array(all_labels, dtype=np.uint8)
    assert all_images.shape == (10000 * len(filenames), 3, 32, 32)
    assert all_labels.shape == (10000 * len(filenames),)

    np.save(dirname + "/" + savefile + "_images.npy", all_images)
    np.save(dirname + "/" + savefile + "_labels.npy", all_labels)


def main():
    base_dir = "../datasets/cifar-10-batches-py"
    training_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_files = ["test_batch"]

    repackage_files(base_dir, training_files, "train")
    repackage_files(base_dir, test_files, "test")


if __name__ == "__main__":
    main()
