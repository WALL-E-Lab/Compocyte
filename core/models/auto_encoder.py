import tensorflow.keras as keras
import tensorflow as tf
import torch
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
try:
    from classiFire.core.models.dense_base import DenseBase

except:
    pass

class AutoEncoder(torch.nn.Module, DenseBase):
    """
    """

    possible_data_types = ['counts']

    def __init__(
        self, 
        n_genes,
        early_stopping=True,
        reduce_LR_plateau=True,
        sequential_kwargs={},
        **kwargs):
        
        super().__init__()
        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()
        self.loss_function = torch.nn.functional.mse_loss
        self.possible_data_types = ['counts']
        self.data_type = 'counts'
        self.early_stopping = early_stopping
        self.reduce_LR_plateau = reduce_LR_plateau
        self.init_sequential(n_genes, **sequential_kwargs)

    def init_sequential(
        self, 
        n_genes, 
        encoder=[256, 256],
        decoder=[256, 256],
        latent_dimensions=10,
        learning_rate=0.01,
        momentum=0.9,
        loss_function='categorical_crossentropy',
        dropout=0.4,
        l2_reg_input=False):
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg_input = l2_reg_input
        layers_encoder = [n_genes] + encoder + [latent_dimensions]
        layers_decoder = [latent_dimensions] + decoder + [n_genes]
        for i in range(len(layers_encoder) - 1):
            n_in = layers_encoder[i]
            n_out = layers_encoder[i + 1]
            new_linear = torch.nn.Linear(n_in, n_out)
            new_activation = torch.nn.LeakyReLU(0.1)
            new_batchnorm = torch.nn.BatchNorm1d(n_out)
            new_dropout = torch.nn.Dropout(dropout)
            torch.nn.init.xavier_uniform_(new_linear.weight, gain=torch.nn.init.calculate_gain('leaky_relu', 0.1))
            torch.nn.init.zeros_(new_linear.bias)
            self.encoder.append(new_linear)
            if i < (len(layers_encoder) - 2):
                self.encoder.append(new_activation)
                self.encoder.append(new_batchnorm)
                self.encoder.append(new_dropout)

        for i in range(len(layers_decoder) - 1):
            n_in = layers_decoder[i]
            n_out = layers_decoder[i + 1]
            new_linear = torch.nn.Linear(n_in, n_out)
            new_activation = torch.nn.LeakyReLU(0.1)
            new_batchnorm = torch.nn.BatchNorm1d(n_out)
            new_dropout = torch.nn.Dropout(dropout)
            torch.nn.init.xavier_uniform_(new_linear.weight, gain=torch.nn.init.calculate_gain('leaky_relu', 0.1))
            torch.nn.init.zeros_(new_linear.bias)
            self.decoder.append(new_linear)
            if i > 0:
                self.decoder.append(new_activation)
                self.decoder.append(new_batchnorm)
                self.decoder.append(new_dropout)

    def forward(self, x):
        x = torch.Tensor(x)
        for layer in self.encoder:            
            x = layer(x)

        for layer in self.decoder:            
            x = layer(x)

        return x

    def get_latent_dimensions(self, x):
        self.eval()
        x = torch.Tensor(x)
        for l in self.encoder:
            x = l(x)

        return x.detach().numpy()