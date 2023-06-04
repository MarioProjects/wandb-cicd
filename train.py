import pytorch_lightning as pl

from utils.data import MNISTDataModule
from utils.learner import ResnetModel


dm = MNISTDataModule()
dm.setup()

model = ResnetModel()
trainer = pl.Trainer(max_epochs=5, default_root_dir="resnet_checkpoints/")
trainer.fit(model, dm)
