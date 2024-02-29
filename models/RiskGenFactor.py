import math
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch import nn, optim
from torchmetrics.regression import SpearmanCorrCoef

from models.modelBase import ModelBase
from scripts.utils import WORK_PATH, DataLoader, seed_all, FEAT_DIM

warnings.filterwarnings("ignore")

class FeatureExtractor(nn.Module):
    def __init__(self, num_latent, hidden_size, num_layers):
        """
        num_latent: number of latent features
        hidden_size: hidden size of GRU
        num_layers: number of layers of GRU
        """
        super(FeatureExtractor, self).__init__()
        self.num_latent = num_latent
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.normalize = nn.LayerNorm(num_latent)
        self.linear = nn.Linear(num_latent, num_latent)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(num_latent, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # Apply linear and LeakyReLU activation
        # x (F, N, T) -> (N, T, F)
        if x.shape[-1] != self.num_latent:
            x = x.permute((1,2,0))

        x = self.normalize(x)
        out = self.linear(x)
        out = self.leakyrelu(out)
        # Forward propagate GRU
        stock_latent, _ = self.gru(out)
        return stock_latent[:,-1,:] #* stock_latent[-1]: (N, hidden_size)

class FactorEncoder(nn.Module):
    def __init__(self, num_factors, num_portfolio, hidden_size):
        """
        num_factors: number of factors
        num_portfolio: number of portfolios
        hidden_size: hidden size of GRU
        """
        super(FactorEncoder, self).__init__()
        self.num_factors = num_factors
        self.linear = nn.Linear(hidden_size, num_portfolio)
        self.softmax = nn.Softmax(dim=1)
        
        self.linear2 = nn.Linear(num_portfolio, num_factors)
        self.softplus = nn.Softplus()
        
    def mapping_layer(self, portfolio_return):
        mean = self.linear2(portfolio_return.squeeze(1))
        sigma = self.softplus(mean)
        return mean, sigma
    
    def forward(self, stock_latent, returns):
        weights = self.linear(stock_latent)
        weights = self.softmax(weights) # (N, F)
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
        portfolio_return = torch.mm(weights.transpose(1,0), returns) # (F, N) * (N, 1) -> (F, 1)
        return self.mapping_layer(portfolio_return)

class AlphaLayer(nn.Module):
    """calcuate stock specific alpha(N*1)"""
    def __init__(self, hidden_size):
        super(AlphaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()
        
    def forward(self, stock_latent):
        stock_latent = self.linear1(stock_latent)
        stock_latent = self.leakyrelu(stock_latent)
        alpha_mu = self.mu_layer(stock_latent)
        alpha_sigma = self.sigma_layer(stock_latent)
        return alpha_mu, self.softplus(alpha_sigma)
        
class BetaLayer(nn.Module):
    """calcuate factor exposure beta(N*K)"""
    def __init__(self, hidden_size, num_factors):
        super(BetaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, num_factors)
    
    def forward(self, stock_latent):
        beta = self.linear1(stock_latent)
        return beta
        
class FactorDecoder(nn.Module):
    """calcuate stock return(N*1)"""
    def __init__(self, alpha_layer, beta_layer):
        super(FactorDecoder, self).__init__()

        self.alpha_layer = alpha_layer
        self.beta_layer = beta_layer
        self.norm_layer = nn.BatchNorm1d(1)
    
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, stock_latent, factor_mu, factor_sigma):
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        beta = self.beta_layer(stock_latent)

        factor_mu = factor_mu.view(-1, 1)
        factor_sigma = factor_sigma.view(-1, 1)
        factor_sigma[factor_sigma == 0] = 1e-6
        mu = alpha_mu + torch.matmul(beta, factor_mu)
        sigma = torch.sqrt(alpha_sigma**2 + torch.matmul(beta**2, factor_sigma**2) + 1e-6)

        return self.reparameterize(mu, sigma)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, stock_latent):
        self.key = self.key_layer(stock_latent)
        self.value = self.value_layer(stock_latent)
        
        attention_weights = torch.matmul(self.query, self.key.transpose(1,0)) # (N)
        #* scaling
        attention_weights = attention_weights / torch.sqrt(torch.tensor(self.key.shape[0])+ 1e-6)
        attention_weights = self.dropout(attention_weights)
        attention_weights = F.relu(attention_weights) # max(0, x)
        attention_weights = F.softmax(attention_weights, dim=0) # (N)
        
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            return torch.zeros_like(self.value[0])
        else:
            context_vector = torch.matmul(attention_weights, self.value) # (H)
            return context_vector 

class FactorPredictor(nn.Module):
    def __init__(self, hidden_size, num_factor):
        super(FactorPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_factor = num_factor
        self.attention_layers = nn.ModuleList([AttentionLayer(self.hidden_size) for _ in range(num_factor)])
        
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent):
        for i in range(self.num_factor):
            attention_layer = self.attention_layers[i](stock_latent)
            if i == 0:
                h_multi = attention_layer
            else:
                h_multi = torch.cat((h_multi, attention_layer), dim=0)
        h_multi = h_multi.view(self.num_factor, -1)

        h_multi = self.linear(h_multi)
        h_multi = self.leakyrelu(h_multi)
        pred_mu = self.mu_layer(h_multi)
        pred_sigma = self.sigma_layer(h_multi)
        pred_sigma = self.softplus(pred_sigma)
        pred_mu = pred_mu.view(-1)
        pred_sigma = pred_sigma.view(-1)
        return pred_mu, pred_sigma

class GATModel(nn.Module):
    def __init__(self, d_feat=158, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight
    
    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze() # [N, hidden size]


class RiskFeatureExtractor(nn.Module):
    """supervise variance-covariance matrix reconstuction"""
    def __init__(self, num_latent, hidden_size,num_layers=2,dropout=0.2,base_model = 'GRU'):
        super(RiskFeatureExtractor, self).__init__()
        self.num_latent = num_latent
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.normalize = nn.LayerNorm(self.num_latent)
        self.linear = nn.Linear(self.num_latent, self.num_latent)
        self.leakyrelu = nn.LeakyReLU()
        self.GAT_model = GATModel(
            d_feat=self.num_latent,
            hidden_size=self.hidden_size,
            num_layers=num_layers ,
            dropout=dropout,
            base_model=base_model,
        )

    def forward(self, x):
        if x.shape[-1] != self.num_latent:
            x = x.permute((1,2,0))
        x = self.normalize(x)
        out = self.linear(x)
        out = self.leakyrelu(out)
        risk_latent = self.GAT_model(out)
        return risk_latent #* stock_latent[-1]: (batch_size, hidden_size)

class FactorVAE(nn.Module):
    def __init__(self, feature_extractor, factor_encoder, factor_decoder, factor_predictor, risk_extrator):
        super(FactorVAE, self).__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_predictor = factor_predictor
        self.short_cut = nn.Linear(FEAT_DIM, 1)
        self.risk_extrator = risk_extrator
        self.map = nn.Linear(128, 64)

    @staticmethod
    def KL_Divergence(mu1, sigma1, mu2, sigma2):
        kl_div = (torch.log(sigma2/ sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5).sum()
        return kl_div
    
    def forward(self, x, returns):
        stock_latent = self.feature_extractor(x)
        risk_latent = self.risk_extrator(x)
        stock_latent = torch.cat([stock_latent,risk_latent],dim=1)
        stock_latent = self.map(stock_latent)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        reconstruction = self.factor_decoder(stock_latent, factor_mu, factor_sigma)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        reconstruction += self.short_cut(x[:,:,-1])
        # Define VAE loss function with reconstruction loss and KL divergence
        reconstruction_loss = F.mse_loss(reconstruction.squeeze(), returns)
        
        # Calculate KL divergence between two Gaussian distributions
        if torch.any(pred_sigma == 0):
            pred_sigma[pred_sigma == 0] = 1e-6
        kl_divergence = self.KL_Divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)

        vae_loss = reconstruction_loss + kl_divergence
        return vae_loss


    def prediction(self, x):
        stock_latent = self.feature_extractor(x)
        risk_latent = self.risk_extrator(x)
        stock_latent = torch.cat([stock_latent,risk_latent],dim=1)
        stock_latent = self.map(stock_latent)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        y_pred = self.factor_decoder(stock_latent, pred_mu, pred_sigma)
        return y_pred.squeeze()

    def latent_factor(self, x):
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        return (pred_mu, pred_sigma)


class GenFactor(ModelBase, pl.LightningModule):
    def __init__(self, name, dl, args):
        '''
        name: model name
        dl: data loader
        args: yaml file
        '''
        nn.Module.__init__(self)
        ModelBase.__init__(self, name, args)
        self.d_feat = args["model_params"]["d_feat"]
        self.hidden_size = args["model_params"]["hidden_size"]
        self.num_factor = args["model_params"]["num_factor"]
        self.dropout = args["model_params"]["dropout"]
        self.num_layers = args["model_params"]["num_layers"]
        self.max_epochs = args["model_params"]["max_epochs"]
        self.save_path = args["model_params"]["save_path"]
        self.lr_rate = args["model_params"]["learning_rate"]
        self.seed = args["model_params"]["seed"]
        seed_all(self.seed)

        self.feature_layer = FeatureExtractor(self.d_feat, self.hidden_size, self.num_layers)
        self.factor_encoder = FactorEncoder(self.num_factor, self.d_feat, self.hidden_size)
        self.alpha_layer = AlphaLayer(self.hidden_size)
        self.beta_layer = BetaLayer(self.hidden_size, self.num_factor)
        self.factor_decoder = FactorDecoder(self.alpha_layer, self.beta_layer)
        self.factor_predictor = FactorPredictor(self.hidden_size, self.num_factor)
        self.risk_extractor = RiskFeatureExtractor(self.d_feat, self.hidden_size, self.num_layers)
        
        self.factorVAE = FactorVAE(self.feature_layer,self.factor_encoder, self.factor_decoder, self.factor_predictor, self.risk_extractor)

        self.l1_reg = args["model_params"]["l1_reg"]
        self.short_cut = nn.Linear(self.d_feat, 1)

        self.dl = dl
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def forward(self, src):
        output = self.factorVAE.prediction(src)
        output += self.short_cut(src.permute(1,0,2)[:,:,-1]).squeeze()
        return output
    