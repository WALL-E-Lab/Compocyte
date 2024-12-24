import numpy as np
import torch
import os
import pickle
import logging
logger = logging.getLogger(__name__)

class DenseTorch(torch.nn.Module):
    def __init__(
            self, 
            labels: list, 
            n_input: int, 
            n_output: int,
            hidden_layers: list=[64, 64],
            dropout: float=0.4,
            batchnorm: bool=True):
        
        super().__init__()

        self.labels_enc = {label: i for i, label in enumerate(labels)}
        self.labels_dec = {self.labels_enc[label]: label for label in self.labels_enc.keys()}
        self.layers = torch.nn.ModuleList()
        layers = [n_input] + hidden_layers + [n_output]
        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]
            new_linear = torch.nn.Linear(n_in, n_out)
            new_activation = torch.nn.LeakyReLU(0.1)
            new_batchnorm = torch.nn.BatchNorm1d(n_out)
            new_dropout = torch.nn.Dropout(dropout)
            torch.nn.init.xavier_uniform_(
                new_linear.weight, 
                gain=torch.nn.init.calculate_gain('leaky_relu', 0.1))
            torch.nn.init.zeros_(new_linear.bias)
            self.layers.append(new_linear)
            if i < (len(layers) - 2):
                self.layers.append(new_activation)
                if batchnorm: 
                    self.layers.append(new_batchnorm)

                self.layers.append(new_dropout)

            else:
                self.layers.append(
                    torch.nn.Softmax(dim=1)
                )

    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        for layer in self.layers:            
            x = layer(x)

        return x

    def predict_logits(self, x):
        self.eval()
        x = torch.Tensor(x)

        return self(x)

    def predict(self, x):        
        logits = self.predict_logits(x)
        logits = logits.detach().numpy()
        pred = np.argmax(logits, axis=1)
        pred = np.array(
            [self.labels_dec[p] for p in pred]
        )

        return pred    

    def reset_output(self, n_output):
        in_features = self.layers[-2].in_features
        del self.layers[-2] # last dense
        del self.layers[-1] # softmax
        self.layers.append(
            torch.nn.Linear(in_features, n_output)
        )
        torch.nn.init.xavier_uniform_(self.layers[-1].weight, gain=torch.nn.init.calculate_gain('leaky_relu', 0.1))
        torch.nn.init.zeros_(self.layers[-1].bias)
        self.layers.append(
            torch.nn.Softmax(dim=1)
        )

    def _save(self, path):
        non_param_attr = ['histories', 'labels_enc', 'labels_dec']
        non_param_dict = {}
        for item in self.__dict__.keys():
            if item in non_param_attr:
                non_param_dict[item] = self.__dict__[item]

        torch.save(self, os.path.join(path, 'model'))
        with open(os.path.join(path, 'non_param_dict.pickle'), 'wb') as f:
            pickle.dump(non_param_dict, f)

    @classmethod
    def _load(cls, path):
        # Reinitialize because simply using keras.load_model messes up the variable type of the model
        model = torch.load(os.path.join(path, 'model'))
        with open(os.path.join(path, 'non_param_dict.pickle'), 'rb') as f:
            non_param_dict = pickle.load(f)

        for item in non_param_dict.keys():
            model.__dict__[item] = non_param_dict[item]

        return model

    @classmethod
    def import_external(
        cls,
        model,
        labels,):

        if not issubclass(type(model), torch.nn.Module):
            raise TypeError('To import an external model as DenseTorch, it must be a subclass of torch.nn.Module.')

        denseTorch = cls(labels, 2, 2)
        denseTorch.layers = torch.nn.ModuleList([model])

        return denseTorch