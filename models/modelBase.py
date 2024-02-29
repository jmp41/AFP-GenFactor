import collections
import copy
import gc

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import spearmanr
from torch import nn, optim

from scripts.utils import reset_model_weights, seed_all


class ModelBase(pl.LightningModule):
    def __init__(self, name, args):
        nn.Module.__init__(self)
        super(ModelBase, self).__init__()
        # Define model architecture
        self.name = name
        self.train_idx = 0
        self.refit_cnt = 0
        
        # initialize
        self.dl = None
        self.args = args
        self.save_path = None
        self.max_epochs = None
        self.lr_rate = args['model_params']['learning_rate']
        self.seq_len = args['data_params']['seq_len']
        self.optimizer = None
        self.seed = args['model_params']['seed']
        seed_all(self.seed)
        
        self.train_period = ['2005-11-19', '2006-01-01']
        self.valid_period = ['2006-01-01', '2006-02-09']
        self.test_period = ['2006-02-09', '2006-03-09']
    
    def get_mseloss(self, batch):
        _, _, x, y = batch
        if torch.any(x.isnan()):
            x = torch.nan_to_num(x, nan=0.0)
        pred = self.forward(x) 
        loss = nn.functional.smooth_l1_loss(pred, y[:, -1].squeeze()) # smooth l1 loss
        try:
            l1_norm = sum(p.abs().sum() for p in self.short_cut.parameters())
        except:
            l1_norm = sum(p.abs().sum() for p in self.factorVAE.parameters())
        if self.l1_reg is not None:
            return loss + l1_norm * self.l1_reg
        return loss

    def get_metrics(self, batch):
        with torch.no_grad():
            _, _, x, y = batch
            if torch.any(x.isnan()):
                x = torch.nan_to_num(x, nan=0.0)
            pred = self.forward(x) 
            IC,_ = spearmanr(pred.detach().cpu().numpy(), y[:, -1].squeeze().cpu().numpy())
            return IC

    def training_step(self, batch, batch_idx):
        loss = self.get_mseloss(batch)
        ic = self.get_metrics(batch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_ic", ic, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "monitor": "val_loss",
        }

    def validation_step(self, batch, batch_idx):
        loss = self.get_mseloss(batch)
        ic = self.get_metrics(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_ic", ic, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        loss = self.get_mseloss(batch)
        self.log("test_loss", loss)
    
    def train_model(self):
        self.train()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_ic",
            dirpath=self.save_path + "save_model" + f"/{self.name}"+f"/{int(self.valid_period[0][:4])}/",
            filename="best_model_" + f"{self.name}",
            save_top_k=1,
            mode="min",
        )
        early_stopping = EarlyStopping('val_ic',patience=self.args['model_params']['early_stopping'],mode='max')
        tb_logger = TensorBoardLogger(self.save_path+"/logs", name=self.name, version=str(self.valid_period[0][:4]))

        trainer = pl.Trainer(
            max_epochs=self.max_epochs, 
            callbacks=[early_stopping, checkpoint_callback], 
            logger=tb_logger
        )
        self.dl.update_period(self.train_period)
        self.val_dl = copy.copy(self.dl)
        self.val_dl.update_period(self.valid_period)
        print("train period", self.train_period, "valid period", self.valid_period)
        trainer.fit(self, self.dl, self.val_dl)
        del self.val_dl
        gc.collect()

    def predict(self):
        # predict test period and save to csv files
        self.eval()
        self.cuda()
        self.dl.update_period(self.valid_period)
        preds = []
        facts = []
        dates = []
        for _, batch in enumerate(self.dl):
            id, date, x, y = batch
            dates.append(date)
            if torch.any(x.isnan()):
                x = torch.nan_to_num(x, nan=0.0)

            pred = self.forward(x)
            preds.append(pd.Series(pred.detach().cpu().numpy(), index=id))
            facts.append(pd.Series((y[:, -1]).detach().cpu().numpy(), index=id))
        return preds, facts, dates
    
    def load_best_model(self):
        dir = (
            self.save_path
            + "save_model"
            + f"/{self.name}"
            + f"/{int(self.valid_period[0][:4])}/"
            + "best_model_"
            + f"{self.name}.ckpt"
        )
        return dir

    def reset_weight(self):
        reset_model_weights(self)
        self.optimizer.state = collections.defaultdict(dict)
    
    def refit(self):
        # rolling training, window length 12 months
        train_start, train_end = self.seq_len + self.refit_cnt*self.args['data_params']['refit'], self.seq_len + self.args['data_params']['train_length']+self.refit_cnt*self.args['data_params']['refit']
        valid_start, valid_end = train_end+1, train_end+1+self.args['data_params']['valid_length']
        # test_start, test_end = valid_end+1, valid_end+1+self.args['data_params']['test_length']
        if valid_end>len(self.dl.date) - 1:
            valid_end = len(self.dl.date) - 1
        self.train_period = [self.dl.date[train_start], self.dl.date[train_end]]
        self.valid_period = [self.dl.date[valid_start], self.dl.date[valid_end]]
        # self.test_period  = [self.dl.date[test_start], self.dl.date[test_end]]
        self.refit_cnt += 1