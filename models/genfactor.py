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
from scripts.utils import WORK_PATH, DataLoader, seed_all

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

class FactorVAE(nn.Module):
    def __init__(self, feature_extractor, factor_encoder, factor_decoder, factor_predictor):
        super(FactorVAE, self).__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_predictor = factor_predictor

    @staticmethod
    def KL_Divergence(mu1, sigma1, mu2, sigma2):
        kl_div = (torch.log(sigma2/ sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5).sum()
        return kl_div
    
    def forward(self, x, returns):
        stock_latent = self.feature_extractor(x)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        reconstruction = self.factor_decoder(stock_latent, factor_mu, factor_sigma)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)

        # Define VAE loss function with reconstruction loss and KL divergence
        reconstruction_loss = F.mse_loss(reconstruction, returns)
            
        # Calculate KL divergence between two Gaussian distributions
        if torch.any(pred_sigma == 0):
            pred_sigma[pred_sigma == 0] = 1e-6
        kl_divergence = self.KL_Divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)

        vae_loss = reconstruction_loss + kl_divergence
        return vae_loss


    def prediction(self, x):
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        y_pred = self.factor_decoder(stock_latent, pred_mu, pred_sigma)
        return y_pred

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
        
        self.factorVAE = FactorVAE(self.feature_layer,self.factor_encoder, self.factor_decoder, self.factor_predictor)

        self.l1_reg = None

        self.dl = dl
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def forward(self, src):
        output = self.factorVAE.prediction(src)
        return output.squeeze()
    
    def training_step(self, batch, batch_idx):
        _,_,x, y = batch
        loss = self.factorVAE(x,y[:,-1])
        self.log('train_loss', loss)
        return loss