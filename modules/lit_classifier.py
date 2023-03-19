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

        self.acc_val = torchmetrics.Accuracy(task='binary')
        self.f1_val = torchmetrics.F1Score(task='binary')
        self.mcc_val = torchmetrics.MatthewsCorrCoef(task='binary')

        self.acc_test = torchmetrics.Accuracy(task='binary')
        self.f1_test = torchmetrics.F1Score(task='binary')
        self.mcc_test = torchmetrics.MatthewsCorrCoef(task='binary')
        #### Optimizer params
        self.lr=lr
        self.weight_decay=weight_decay
    

    def forward(self, x):
        x = self.graph_model(x)

        return x

    def training_step(self, batch, batch_idx):
        g, y = batch
        logits = self.graph_model(g)
        y = y.float().unsqueeze(1)
        loss = self.loss_func(logits, y)
        preds = logits.round().float()
        # Accumulate Accuracy, F1 and MCC (Training)
        self.acc(preds, y)
        self.f1(preds, y)
        self.mcc(preds, y)

        return loss
    
    def validation_step(self, batch, batch_idx):
        g, y = batch
        logits = self.graph_model(g)
        y = y.float().unsqueeze(1)
        loss = self.loss_func(logits, y)
        preds = logits.round().float()
        # Accumulate Accuracy, F1 and MCC (Validation)
        val_acc = self.acc_val(preds, y)
        val_f1 = self.f1_val(preds, y)
        val_mcc = self.mcc_val(preds, y)

        return loss
    
    def test_step(self, batch, batch_idx):
        g, y = batch
        logits = self.graph_model(g)
        y = y.float().unsqueeze(1)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        preds = preds.float().unsqueeze(1)
        acc = self.acc_test(preds, y)
        f1 = self.f1_test(preds, y)
        mcc = self.mcc_test(preds, y)
        self.log('test_loss', loss,  on_epoch=True, logger=True)
        self.log('test_acc', acc,  on_epoch=True, logger=True)
        self.log('test_f1', f1,  on_epoch=True, logger=True)
        self.log('test_mcc', mcc,  on_epoch=True, logger=True)

        return loss
    
    def on_train_epoch_end(self):
        # Compute metrics
        t_acc = self.acc.compute()
        t_f1 = self.f1.compute()
        t_mcc = self.mcc.compute()
        self.log('train_acc', t_acc)
        self.log('train_f1', t_f1)
        self.log('train_mcc', t_mcc)
        
        self.acc.reset()
        self.f1.reset()
        self.mcc.reset()
        print(f"\nTraining accuracy: {t_acc}, "\
    f"f1: {t_f1}, mcc: {t_mcc}")

        
    def on_validation_epoch_end(self):
        v_acc =  self.acc_val.compute()
        v_f1 = self.f1_val.compute()
        v_mcc = self.mcc_val.compute()
        self.log('val_acc',v_acc)
        self.log('val_f1', v_f1)
        self.log('val_mcc', v_mcc)
        
        self.acc_val.reset()
        self.f1_val.reset()
        self.mcc_val.reset()
        
        print(f"\nValidation accuracy: {v_acc}, "\
    f"f1: {v_f1}, mcc: {v_mcc}")
        

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-2)
        return optimizer