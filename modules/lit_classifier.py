import torch
import pytorch_lightning as pl
from torch.optim import Adam
import torchmetrics

class PlastDectClassifier(pl.LightningModule):
    def __init__(self, graph_model, loss_func, lr=1e-3, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()
        #### Graph encoder (GGNN, GATv2, GCN)
        self.graph_model = graph_model
        #### loss function
        self.loss_func = loss_func
        #### Metrics
        self.acc = torchmetrics.Accuracy(task='binary')
        self.f1 = torchmetrics.F1Score(task='binary')
        self.mcc = torchmetrics.MatthewsCorrCoef(task='binary')
        #### Optimizer params
        self.lr=lr
        self.weight_decay=weight_decay
    

    def forward(self, x):
        x = self.graph_model(x)

        return x

    def training_step(self, batch, batch_idx):
        g, y = batch
        logits = self.graph_model(g)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds = preds.float().unsqueeze(1)
        acc = self.acc(preds, y)
        f1 = self.f1(preds, y)
        mcc = self.mcc(preds, y)
        self.log('train_loss', loss,  on_epoch=True, logger=True)
        self.log('train_acc', acc,  on_epoch=True, logger=True)
        self.log('train_f1', f1,  on_epoch=True, logger=True)
        self.log('train_mcc', mcc,  on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        g, y = batch
        logits = self.graph_model(g)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds = preds.float().unsqueeze(1)
        acc = self.acc(preds, y)
        f1 = self.f1(preds, y)
        mcc = self.mcc(preds, y)
        self.log('val_loss', loss,  on_epoch=True, logger=True)
        self.log('val_acc', acc,  on_epoch=True, logger=True)
        self.log('val_f1', f1,  on_epoch=True, logger=True)
        self.log('val_mcc', mcc,  on_epoch=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        g, y = batch
        logits = self.graph_model(g)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds = preds.float().unsqueeze(1)
        acc = self.acc(preds, y)
        f1 = self.f1(preds, y)
        mcc = self.mcc(preds, y)
        self.log('test_loss', loss,  on_epoch=True, logger=True)
        self.log('test_acc', acc,  on_epoch=True, logger=True)
        self.log('test_f1', f1,  on_epoch=True, logger=True)
        self.log('test_mcc', mcc,  on_epoch=True, logger=True)

        return loss
        

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-2)
        return optimizer