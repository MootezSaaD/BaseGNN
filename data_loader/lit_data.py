from data_loader.dataset_ import PlaseDectDataset
import pytorch_lightning as pl
from utils.data import static_splitter
import dgl

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, args, num_workers = 4 , batch_size = 32):
        super().__init__()
        train_split, test_split, val_split =  static_splitter(data_dir)

        self.train_split = PlaseDectDataset(train_split, args)
        self.test_split = PlaseDectDataset(test_split, args)
        self.val_split = PlaseDectDataset(val_split, args)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
        self.train_split,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=True,
        shuffle=True
    )

    def val_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
        self.val_split,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=True,
        shuffle=False
    )

    def test_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
        self.test_split,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=True,
        shuffle=False
    )