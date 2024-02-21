import copy
import math
import os
import time
import warnings

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from torch import nn, optim

from models.modelBase import ModelBase
from scripts.utils import WORK_PATH, DataLoader

warnings.filterwarnings("ignore")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(ModelBase, pl.LightningModule):
    def __init__(self, name, dl, args):
        nn.Module.__init__(self)
        ModelBase.__init__(self, name, args)
        self.d_feat = args["model_params"]["d_feat"]
        self.d_model = args["model_params"]["d_model"]
        self.nhead = args["model_params"]["nhead"]
        self.dropout = args["model_params"]["dropout"]
        self.num_layers = args["model_params"]["num_layers"]
        self.max_epochs = args["model_params"]["max_epochs"]
        self.save_path = args["model_params"]["save_path"]

        self.feature_layer = nn.Linear(self.d_feat, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead, dropout=self.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers
        )
        self.decoder_layer = nn.Linear(self.d_model, 1)

        self.l1_reg = None

        self.dl = dl

    def forward(self, src):
        # src [F, N, T] -> [N, T, F]
        src = src.permute(1, 2, 0)
        src = self.feature_layer(src)
        mask = None

        # src [N, T, F] -> [T, N, F]
        src = src.transpose(1, 0)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [T, N, F]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [N, 1]

        return output.squeeze()

    def get_mseloss(self, batch):
        _, _, x, y = batch
        if torch.any(x.isnan()):
            x = torch.nan_to_num(x, nan=0.0)
        pred = self.forward(x)
        loss = nn.functional.mse_loss(pred, y[:, -1].squeeze())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_mseloss(batch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "val_loss",
        }

    def validation_step(self, batch, batch_idx):
        loss = self.get_mseloss(batch)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.get_mseloss(batch)
        self.log("test_loss", loss)