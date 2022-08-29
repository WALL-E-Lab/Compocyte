import tensorflow.keras as keras
import tensorflow as tf
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DenseKeras(keras.Model):
    """
    TODO:
    Deal with history
    Deal with plotting and callbacks, train_val split now outside of network class
    """

    def __init__(
        self, 
        n_input=None, 
        n_output=None, 
        model=None,
        early_stopping=True,
        reduce_LR_plateau=True,
        sequential_kwargs={}, 
        **kwargs):

        super().__init__()
        self.possible_data_types = ['counts', 'normlog']
        self.data_type = 'normlog'
        if model is None:
            if n_input is None or n_output is None:
                raise Exception('If the model is to be defined from scratch, input and output need to be defined.')

            self.init_sequential(n_input, n_output, **sequential_kwargs)

        elif issubclass(type(model), keras.Model):
            self.model = model

        else:
            raise TypeError('To import an external model as DenseKeras, it must be a subclass of keras.Model.')

        callbacks = []
        if early_stopping: 
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10,
                restore_best_weights=True,
                verbose=1)
            callbacks.append(early_stopping_callback)

        if reduce_LR_plateau:
            reduce_LR_plateau_callback = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                factor=0.2,
                verbose=1)
            callbacks.append(reduce_LR_plateau_callback)

        if not len(callbacks) == 0:
            self.callbacks = callbacks

    def set_data_type(self, data_type):
        self.data_type = data_type

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
        l2_reg_input=False):

        self.model = keras.models.Sequential()
        activation_function = lambda x: keras.activations.relu(x, alpha=0.1)
        layers = [n_input] + hidden_layers + [n_output]
        layers_in_out = list(zip(layers[:-1], layers[1:]))
        for idx, layer in enumerate(layers_in_out):
            if idx == len(layers_in_out) - 1: # final layer
                activation = 'softmax'
                dropout_layer = 0.0

            else:
                activation = activation_function
                dropout_layer = dropout

            if idx == 0:
                self.model.add(keras.Input(
                    shape=(layer[0], )))
                if discretization:
                    self.model.add(keras.layers.Discretization(bin_boundaries=[-0.675, 0, 0.675]))

            if l2_reg_input:
                self.model.add(keras.layers.Dense(
                    input_shape=(layer[0], ),
                    units=layer[1],
                    kernel_initializer='glorot_uniform',                                          
                    bias_initializer='zeros',
                    activation=activation,
                    kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=keras.regularizers.L2(1e-4),
                    activity_regularizer=keras.regularizers.L2(1e-5)))

            else:
                self.model.add(keras.layers.Dense(
                    input_shape=(layer[0], ),
                    units=layer[1],
                    kernel_initializer='glorot_uniform',                                          
                    bias_initializer='zeros',
                    activation=activation))

            if dropout_layer > 0.0:
                self.model.add(keras.layers.Dropout(dropout_layer))

        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate, 
            momentum=momentum)
        self.compile(
            optimizer=optimizer, 
            loss=loss_function,
            metrics=['accuracy'])

    def reset_output(self, n_output):
        input_shape = (self.model.layers[-1].input.shape[1], )
        activation = self.model.layers[-1].activation
        self.model.pop()
        self.model.add(keras.layers.Dense(
                        input_shape=input_shape,
                        units=n_output,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        activation=activation)) 

    def call(self, inputs):
        return self.model.call(inputs)

    def plot_training(self, history, to_plot):
        plt.plot(history.history[f'{to_plot}'])
        plt.plot(history.history[f'val_{to_plot}'])
        plt.title(f'model {to_plot}')
        plt.ylabel(f'{to_plot}')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def _train(
        self,
        x,
        y_onehot,
        y_int,
        batch_size=40,
        epochs=1000,
        verbose=0,
        plot=False,
        **kwargs):

        if hasattr(self, 'callbacks'):
            callbacks = self.callbacks

        else:
            callbacks = []

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
            verbose=verbose,
            validation_data=(x_val, y_val),
            callbacks=callbacks)
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

        self.save(os.path.join(path, 'model.SavedModel'))
        with open(os.path.join(path, 'non_param_dict.pickle'), 'wb') as f:
            pickle.dump(non_param_dict, f)

    @classmethod
    def _load(cls, path):
        # Reinitialize because simply using keras.load_model messes up the variable type of the model
        loaded_model = keras.models.load_model(os.path.join(path, 'model.SavedModel'))
        model = cls(model=loaded_model.model)
        with open(os.path.join(path, 'non_param_dict.pickle'), 'rb') as f:
            non_param_dict = pickle.load(f)

        for item in non_param_dict.keys():
            model.__dict__[item] = non_param_dict[item]

        return model

    @classmethod
    def import_external(cls, model, feature_names, data_type, label_encoding, label_decoding, is_binary=False):
        if not issubclass(type(model), keras.Model):
            raise TypeError('To import an external model as DenseKeras, it must be a subclass of keras.Model.')

        pass

class DenseTorch(torch.nn.Module):
    """Todo: and reset_output
    """

    def __init__(
        self, 
        n_input=None, 
        n_output=None,
        early_stopping=True,
        reduce_LR_plateau=True,
        sequential_kwargs={},
        module=None, 
        fit_function=None, 
        predict_function=None,
        **kwargs):
        
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.loss_function = torch.nn.functional.cross_entropy
        self.possible_data_types = ['counts', 'normlog']
        self.data_type = 'normlog'
        self.early_stopping = early_stopping
        self.reduce_LR_plateau = reduce_LR_plateau
        if module is None:
            self.imported = False
            self.module = None
            self.init_sequential(n_input, n_output, **sequential_kwargs)

        else:
            self.imported = True
            self.module = module
            self.fit_function = fit_function
            self.predict_function = predict_function
            if type(fit_function) != str or type(predict_function) != str:
                raise TypeError('fit_function and predict_function must be of type str and point to the relevant functions in the import module.')

    def set_data_type(self, data_type):        
        self.data_type = data_type

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
        l2_reg_input=False):
        
        self.discretization = discretization
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg_input = l2_reg_input
        layers = [n_input] + hidden_layers + [n_output]
        layers_in_out = list(zip(layers[:-1], layers[1:]))
        for idx, layer in enumerate(layers_in_out):
            self.layers.append(
                torch.nn.Linear(layer[0], layer[1])
            )
            if idx != len(layers_in_out) - 1 and dropout > 0:
                self.layers.append(
                    torch.nn.Dropout(dropout)
                )

            if idx == len(layers_in_out) - 1:
                self.layers.append(
                    torch.nn.Softmax(dim=1)
                )

    def reset_output(self, n_output):
        pass

    def forward(self, x):
        x = torch.Tensor(x)
        if self.discretization:
            x = torch.bucketize(x, torch.Tensor([-0.675, 0, 0.675]))

        x = x.type(torch.float32)
        for layer in self.layers:            
            x = layer(x)

        return x

    def fit(
        self, 
        x, 
        y,
        batch_size=40,
        epochs=1000,
        validation_data=None):
        """x is torch.Tensor with shape (n_cells, n_features)
        y is torch.Tensor with shape (n_cells, n_output), containing onehot encoding"""

        if self.imported:
            return getattr(self.module, self.fit_function)(x=x, y=y, batch_size=batch_size, epochs=epochs)

        else:
            self.train()
            x, y = torch.Tensor(x), torch.Tensor(y)
            n_cells = x.shape[0]
            if not validation_data is None:
                x_val, y_val = validation_data
                x_val, y_val = torch.Tensor(x_val), torch.Tensor(y_val)
                n_val = x_val.shape[0]

            if self.l2_reg_input:
                weight_decay = 1e-5

            else:
                weight_decay = 0

            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.learning_rate, 
                momentum=self.momentum,
                weight_decay=weight_decay
            )
            history = {
                'val_accuracy': []
                'val_loss': []
                'train_accuracy': []
                'train_loss': [],
                'state_dicts': []}
            for epoch in range(epochs):
                self.eval()
                history['state_dicts'].append(self.state_dict())
                # Record loss and accuracy
                if not validation_data is None:                    
                    pred_val = self(x_val)
                    pred_int = np.argmax(pred_val.detach().numpy(), axis=-1)
                    y_int = np.argmax(y_val.detach.numpy(), axis=-1)
                    val_accuracy = np.mean(pred_int == y_int) * 100
                    val_loss = self.loss_function(pred_val, y_val).item()
                    history['val_accuracy'].append(val_accuracy)
                    history['val_loss'].append(val_loss)

                pred = self(x)
                pred_int = np.argmax(pred.detach().numpy(), axis=-1)
                y_int = np.argmax(y.detach.numpy(), axis=-1)
                accuracy = np.mean(pred_int == y_int) * 100
                loss = self.loss_function(pred, y).item()
                history['accuracy'].append(train_accuracy)
                history['loss'].append(train_loss)

                self.train()
                for i in range((n_cells - 1) // batch_size + 1):                    
                    idx_start = i * batch_size
                    idx_end = idx_start + batch_size
                    xb, yb = x[idx_start:idx_end], y[idx_start:idx_end]
                    pred = self(xb)
                    loss = self.loss_function(pred, yb)
                    loss_train = loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if self.early_stopping:
                    if history['val_loss'][-1] <= history['val_loss'][-11]:
                        # Plateau reached, restore best weights and exit epoch loop
                        self.load_state_dict(history['state_dicts'][np.argmin(history['val_loss'])])
                        break

                if self.reduce_LR_plateau:
                    # Plateau reached, lower learning rate
                    if history['val_loss'][-1] <= history['val_loss'][-6]:
                        for g in optimizer.param_groups:
                            g['lr'] = g['lr'] * 0.2

            del history['state_dicts']
            return history

    def predict(self, x):
        if self.imported:
            return getattr(self.module, self.predict_function)(x=x)

        else:
            self.eval()
            x = torch.Tensor(x)
            y = self(x).detach().numpy()

        return y

    def plot_training(self, history, to_plot):
        plt.plot(history[f'{to_plot}'])
        plt.plot(history[f'val_{to_plot}'])
        plt.title(f'model {to_plot}')
        plt.ylabel(f'{to_plot}')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def _train(
        self,
        x,
        y_onehot,
        y_int,
        batch_size=40,
        epochs=1000,
        verbose=0,
        plot=False,
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
            validation_data=(x_val, y_val))

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
        if not fit_function in dir_model or not predict_function in dir_model:
            raise Exception('fit_function and predict_function must be of type str and point to the relevant functions in the import module.')

        denseTorch = cls(module=model, fit_function=fit_function, predict_function=predict_function)
        denseTorch.set_data_type(data_type)

        return denseTorch