import warnings

import pytorch_lightning as pl
import torch
from torch import nn

from models.modelBase import ModelBase

warnings.filterwarnings("ignore")

class MLP(ModelBase, pl.LightningModule):
    def __init__(self, name, dl, args):
        nn.Module.__init__(self)
        ModelBase.__init__(self, name, args)
        self.d_feat = args["model_params"]["d_feat"]
        self.hidden_size = args["model_params"]["hidden_size"]
        self.dropout = args["model_params"]["dropout"]
        self.max_epochs = args["model_params"]["max_epochs"]
        self.save_path = args["model_params"]["save_path"]
        self.seq_len = args["data_params"]["seq_len"]
        self.lr_rate = args["model_params"]["learning_rate"]
        
        self.encoder = nn.Linear(self.seq_len, 1) 
        
        # building decoder
        modules = []
        in_channels = self.d_feat
        for h_dim in self.hidden_size:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        modules.append(nn.Linear(h_dim, 1))
        self.decoder = nn.Sequential(*modules)
        self.output_norm = nn.BatchNorm1d(1)
        self.l1_reg = None
        self.dl = dl
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
    
    def forward(self, x):
        # x (F, N, T) -> (N, F, 1)
        x = x.permute(1,0,2)
        x = self.encoder(x)
        x = self.decoder(x.squeeze())
        x = self.output_norm(x)
        return x.squeeze() # (N, )
    