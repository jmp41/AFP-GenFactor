import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch import nn

WORK_PATH = Path.cwd()
print(WORK_PATH)
class DataLoader:
    def __init__(
        self,
        args,
        device=None,
    ):
        self.data = pd.read_csv(WORK_PATH / Path(args['data_params']['data_path']))
        # load saved data
        self.fac_name = [f for f in self.data.columns if f not in ['Ticker','1d_return', '5d_return', '1d_residual_return', '1d_predict_return', 'Date']]
        self.category = [f for f in self.data.columns if self.data[f].dtypes in ['bool', 'int64']]
        self.target = args['data_params']['target']
        self.data = self.data.dropna(subset=[self.target]) # drop nan in label, feature
        self.buffer = args['data_params']['using_buffer']
        
        # model params
        self.shuffle = args['data_params']['shuffle']
        self.seq_len = args['data_params']['seq_len']
        self.batch_size = args['model_params']['batch_size'] # -1 represents daily batch
        
        self.device = device
        
        # generator
        self.period = None
        self.index = 0
        
        # data cleaning
        if not self.buffer:
            to_clean = [f for f in self.fac_name if f not in self.category]
            self.data[to_clean] = self.data.groupby('Date')[to_clean].apply(cross_section_norm) # convert non-categorical feature to [-3,3]
            self.data[self.category] = self.data.groupby('Date')[self.category].apply(cross_section_norm_category) # convert categorical feature to [0,1]
            self.data[self.target] = self.data.groupby('Date')[self.target].apply(cross_section_norm) # convert label to [-3,3], we only care cross-section ranking
            self.data.to_csv(WORK_PATH / Path(f'data/cleaned_data_{self.target}.csv'), index=False)
        else:
            self.data = pd.read_csv(WORK_PATH / Path(f'data/cleaned_data_{self.target}.csv'))
            
        self.stock_id = self.data.set_index(['Date','Ticker'])[self.fac_name].unstack(level=1)['alpha001'].columns.values
        self.date = self.data.set_index(['Date','Ticker'])[self.fac_name].unstack(level=1)['alpha001'].index.values
        self.feature = self.data.set_index(['Date','Ticker'])[self.fac_name].unstack(level=1).values.reshape(len(self.date), len(self.stock_id), len(self.fac_name)) # (T, N, F)
        self.label = self.data.set_index(['Date','Ticker'])[self.target].unstack(level=1).values.reshape(len(self.date), len(self.stock_id)) # (T, N)
        assert self.feature.shape[1] == self.label.shape[-1], f"feature size {self.feature.shape} does not match lable size {self.label.shape}"
        
    def __len__(self):
        start_date, end_date = self.period
        assert start_date<=end_date, f"start date {start_date} is ahead of end date {end_date}"
        indices = np.arange(len(self.date))
        valid_date = np.bitwise_and(self.date>=start_date, self.date<=end_date)
        to_iter = indices[valid_date]
        return len(to_iter)
    
    def __getitem__(self, idx):
        # capture time-series information, seqence data slice -> stock id (N, ), date (1, ), feature (F, N, seq_len), label (N, seq_len)
        assert idx - self.seq_len>=0, "index is smaller than sequence length"
        seq_label = torch.from_numpy(self.label[(idx - self.seq_len):idx, :]).float().to(self.device).permute(1,0) # (N, seq_len)
        seq_feature = torch.from_numpy(self.feature[(idx - self.seq_len):idx,:,:]).float().to(self.device).permute(2,1,0) # (F, N, seq_len)
        mask = ~torch.any(seq_label.isnan(), dim = 1)
        return self.stock_id[mask.cpu().numpy()], self.date[idx], self._norm(seq_feature[:,mask,:]), seq_label[mask,:]
    
    def __iter__(self):
        '''
        yield idx for a period of times. period: [start date, end date] -> idx of date
        '''
        start_date, end_date = self.period
        assert start_date<=end_date, f"start date {start_date} is ahead of end date {end_date}"
        indices = np.arange(len(self.date))
        valid_date = np.bitwise_and(self.date>=start_date, self.date<=end_date)
        to_iter = indices[valid_date]
        if self.shuffle:
            np.random.shuffle(to_iter)
        for i in to_iter:
            yield self[i]
    
    def update_period(self, period):
        self.period = period
        
    def _norm(self, src):
        return torch.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)

def cross_section_norm(df):
    '''
    cross section normalization
    '''
    z_score = (df - np.nanmean(df))/(np.nanstd(df)+1e-6)
    z_score[z_score>3] = 3
    z_score[z_score<-3] = -3
    return z_score

def cross_section_norm_category(df):
    '''
    cross section normalization for categorical data
    '''
    if df.dtypes.all() == 'bool':
        return df.astype(int)
    else:
        return cross_section_norm(df + np.random.normal(0, 0.01, df.shape))
    
def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)