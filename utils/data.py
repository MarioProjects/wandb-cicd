import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return {
            "image": torch.tensor(self.X[ix]).float().view(1, 28, 28),
            "label": torch.tensor(self.y[ix]).long()
        }


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str = "mnist_5k.csv",
            batch_size: int = 64,
            samples: int = 5000
        ):
        super().__init__()

        assert batch_size > 0, "batch_size must be greater than 0"
        assert samples > 0, "samples must be greater than 0"
        assert samples <= 5000, "samples must be less than or equal to 5000"

        self.data_path = data_path
        self.batch_size = batch_size
        self.samples = samples

    def setup(self, stage=None):
        mnist = pd.read_csv(self.data_path)
        X = mnist[mnist.columns[mnist.columns != 'label']].values
        y = mnist["label"].values
        train_size, val_size = int(0.8 * self.samples), int(0.2 * self.samples)
        print(f"Using {train_size} samples for training")
        print(f"Using {self.samples - train_size} samples for validation")
        X_train = X[:train_size] / 255.
        y_train = y[:train_size].astype(int)
        X_test = X[train_size:train_size + val_size] / 255.
        y_test = y[train_size:train_size + val_size].astype(int)
        self.train_ds = Dataset(X_train, y_train)
        self.val_ds = Dataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)
