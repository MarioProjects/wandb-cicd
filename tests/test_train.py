import pytorch_lightning as pl

from utils.data import MNISTDataModule
from utils.learner import ResnetModel


def test_train():
    dm = MNISTDataModule(
        batch_size=6,  # low bs to fit on CPU if needed
        samples=50  # small size for the smoke test
    )
    dm.setup()

    model = ResnetModel()
    trainer = pl.Trainer(max_epochs=5, default_root_dir="resnet_checkpoints/")
    trainer.fit(model, dm)
