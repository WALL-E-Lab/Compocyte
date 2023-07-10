import torch
import os
import pickle
from sklearn.model_selection import train_test_split
from copy import deepcopy
try:
    from scFlorist.core.models.dense_base import DenseBase

except:
    pass

class DenseTorch(torch.nn.Module, DenseBase):
    """
    """

    possible_data_types = ['counts', 'normlog']

    def __init__(
        self, 
        n_input=None, 
        n_output=None,
        early_stopping=True,
        reduce_LR_plateau=True,
        sequential_kwargs={},
        module=None, 
        encoder=None,
        fit_function=None, 
        predict_function=None,
        weight=None, # class weights for loss functionn
        **kwargs):
        
        super().__init__()
        if torch.cuda.is_available():
            self.device = 'cuda'

        else:
            self.device = 'cpu'

        self.layers = torch.nn.ModuleList()
        if weight is not None:
            weight = torch.Tensor(weight).to(self.device)

        self.loss_function = torch.nn.CrossEntropyLoss(weight=weight)
        self.possible_data_types = ['counts', 'normlog']
        self.data_type = 'normlog'
        self.early_stopping = early_stopping
        self.reduce_LR_plateau = reduce_LR_plateau
        self.imported = False
        self.module = None
        self.encoder_based = False
        if module is None and encoder is None:
            self.init_sequential(n_input, n_output, **sequential_kwargs)
        
        elif encoder is not None:
            self.encoder_based = True
            if n_output is None:
                raise Exception() # define

            self.encoder = deepcopy(encoder)
            self.init_sequential(encoder[-1].out_features, n_output, **sequential_kwargs)

        else:
            self.imported = True
            self.module = module
            self.fit_function = fit_function
            self.predict_function = predict_function
            if type(fit_function) != str or type(predict_function) != str:
                raise TypeError('fit_function and predict_function must be of type str and point to the relevant functions in the import module.')

    def init_sequential(
        self, 
        n_input, 
        n_output, 
        hidden_layers=[64, 64],
        learning_rate=0.01,
        momentum=0.9,
        loss_function='categorical_crossentropy',
        dropout=0.4,
        discretization=False,
        l2_reg_input=False,
        batchnorm=True):
        
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg_input = l2_reg_input
        layers = [n_input] + hidden_layers + [n_output]
        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]
            new_linear = torch.nn.Linear(n_in, n_out)
            new_activation = torch.nn.LeakyReLU(0.1)
            new_batchnorm = torch.nn.BatchNorm1d(n_out)
            new_dropout = torch.nn.Dropout(dropout)
            torch.nn.init.xavier_uniform_(new_linear.weight, gain=torch.nn.init.calculate_gain('leaky_relu', 0.1))
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

    def reset_output(self, n_output):
        if self.imported:
            if 'reset_output' not in dir(self.module):
                raise Exception(
                    'No method for resetting the output layer (appending new output nodes) defined. ' \
                    'If you want to update the hierarchy at this node, you need to include reset_output in your imported classifier.')

            else:
                return getattr(self.module, 'reset_output')(n_output)

        else:
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


    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        x = torch.Tensor(x).to(self.device)
        if self.discretization:
            x = torch.bucketize(x, torch.Tensor([-0.675, 0, 0.675]).to(self.device)).to(self.device)

        x = x.type(torch.float32)
        if self.encoder_based:
            for layer in self.encoder:
                x = layer(x)

        for layer in self.layers:            
            x = layer(x)

        return x

    def predict(self, x):
        if self.imported:
            return getattr(self.module, self.predict_function)(x=x)

        else:
            self.eval()
            x = torch.Tensor(x).to(self.device)
            y = self(x).detach().cpu().numpy()

        return y

    def _train(
        self,
        x,
        y_onehot,
        y_int,
        batch_size=40,
        epochs=40,
        verbose=0,
        plot=False,
        fit_arguments={},
        **kwargs):

        if not hasattr(self, 'histories'):
            self.histories = []

        x_train, x_val, y_train, y_val = train_test_split(
            x, 
            y_onehot, 
            stratify=y_int, 
            test_size=0.2, 
            random_state=42)
        # TODO
        # Make sure this is transferable to torch
        history = self.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs, 
            #verbose=verbose,
            validation_data=(x_val, y_val),
            **fit_arguments)

        self.histories.append(history)
        if plot:
            self.plot_training(history, 'accuracy')
            self.plot_training(history, 'loss')    

    def _save(self, path):
        non_param_attr = ['history', 'callbacks', 'possible_data_types', 'data_type', 'imported', 'fit_function', 
            'predict_function', 'dropout', 'discretization', 'learning_rate', 'momentum', 'l2_reg_input', 'loss_function',
            'early_stopping', 'reduce_LR_plateau', 'sequential_kwargs']
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
        #feature_names, 
        data_type, 
        #label_encoding, 
        #label_decoding, 
        fit_function, 
        predict_function, 
        is_binary=False):

        if not issubclass(type(model), torch.nn.Module):
            raise TypeError('To import an external model as DenseTorch, it must be a subclass of torch.nn.Module.')

        dir_model = dir(model)
        if fit_function not in dir_model or predict_function not in dir_model:
            raise Exception('fit_function and predict_function must be of type str and point to the relevant functions in the import module.')

        denseTorch = cls(module=model, fit_function=fit_function, predict_function=predict_function)
        denseTorch.set_data_type(data_type)

        return denseTorch

    @classmethod
    def from_encoder(
        cls,
        encoder,
        n_output,
        #feature_names, 
        data_type, 
        #label_encoding, 
        #label_decoding
        ):

        if not issubclass(type(encoder), torch.nn.Module):
            raise TypeError('To import an external model as DenseTorch, it must be a subclass of torch.nn.Module.')

        denseTorch = cls(n_output=n_output, encoder=encoder)
        denseTorch.set_data_type(data_type)

        return denseTorch