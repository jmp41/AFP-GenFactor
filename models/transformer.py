import math
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from scipy.stats import spearmanr
from torch import nn, optim

from models.modelBase import ModelBase
from scripts.utils import WORK_PATH, DataLoader, seed_all

warnings.filterwarnings("ignore")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        '''
        d_model: dimension of the model
        max_len: max length of the sequence
        '''
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
        '''
        name: model name
        dl: data loader
        args: yaml file
        '''
        nn.Module.__init__(self)
        ModelBase.__init__(self, name, args)
        self.d_feat = args["model_params"]["d_feat"]
        self.d_model = args["model_params"]["d_model"]
        self.nhead = args["model_params"]["nhead"]
        self.dropout = args["model_params"]["dropout"]
        self.num_layers = args["model_params"]["num_layers"]
        self.max_epochs = args["model_params"]["max_epochs"]
        self.save_path = args["model_params"]["save_path"]
        self.lr_rate = args["model_params"]["learning_rate"]
        self.seed = args["model_params"]["seed"]
        seed_all(self.seed)

        self.feature_layer = nn.Linear(self.d_feat, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead, dropout=self.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers
        )
        self.decoder_layer = nn.Linear(self.d_model, 1)
        self.output_norm = nn.BatchNorm1d(1)
        self.l1_reg = args["model_params"]["l1_reg"]
        self.short_cut = nn.Linear(self.d_feat, 1)
        
        self.dl = dl
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        

    def forward(self, src):
        # src [F, N, T] -> [N, T, F]
        src = src.permute(1, 2, 0)
        resid = self.short_cut(src[:, -1, :])
        src = self.feature_layer(src)
        mask = None

        # src [N, T, F] -> [T, N, F]
        src = src.transpose(1, 0)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [T, N, F]

        # [T, N, F] --> [N, 1]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [N, 1]
        output += resid
        return output.squeeze()