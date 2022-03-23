import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import confusion_matrix

class NeuralNetwork():
    """Add explanation.
    """

    def __init__(
        self, 
        x_input_data,
        y_input_onehot,
        len_of_output, 
        list_of_hidden_layer_nodes=[30],
        activation_function='relu',
        learning_rate=0.001,
        momentum=.9,
        loss_function='categorical_crossentropy',
        epochs=500
    ):
        """Add explanation.
        """

        self.x_input_data = x_input_data
        self.y_input_data_onehot = y_input_onehot
        self.list_of_layer_nodes = [len(self.x_input_data[0])]\
            + [nodes for nodes in list_of_hidden_layer_nodes] + [len_of_output]  
        self.activation_function = activation_function 
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.loss_function = loss_function
        self.epochs = epochs
        self.model = keras.models.Sequential()
        for nodes, layer_idx in zip(
            self.list_of_layer_nodes, 
            range(0, len(self.list_of_layer_nodes) - 1)
        ):
            if layer_idx !=  len(self.list_of_layer_nodes) - 2:
                activation = self.activation_function

            else:
                activation = 'softmax'

            self.model.add(keras.layers.Dense(
                input_shape = (nodes,),
                units = self.list_of_layer_nodes[layer_idx+1],                            
                kernel_initializer = 'glorot_uniform',                                          
                bias_initializer = 'zeros',                                               
                activation = activation))


    def train(self):
        """Train the NN using the x_training_data input and onehot encoded 
        y_training_onehot.
        """

        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3)
        self.optimizer = keras.optimizers.SGD(
            learning_rate = self.learning_rate, 
            momentum = self.momentum)
        self.model.compile(
            optimizer = self.optimizer, 
            loss = self.loss_function)
        history = self.model.fit(
            self.x_input_data,
            self.y_input_data_onehot,
            batch_size = 64,
            epochs = self.epochs, 
            verbose = 0,
            validation_split = .1)

    def predict(self, input_vec):
        """Calculate and return label prediction of trained model for an input
        vector input_vec (dtype=int)"""

        # Returns only absolute decisions, nothing known about the difference 
        # in prediction confidence 
        pred_vec = np.argmax(self.model.predict(input_vec), axis = -1)

        return pred_vec

    def validate(self, x_input, y_input_int):
        """Add explanation.
        """

        y_preds = self.predict(x_input)
        def calc_acc(pred_vec, known_vec):
            if type(pred_vec) == type(known_vec):
                acc = np.sum(pred_vec == known_vec, axis = 0) / len(known_vec)

            else:
                raise ValueError('self.validate: Error! \
                    Comparison of different label encoding!')

            return acc

        acc = calc_acc(y_preds, y_input_int)
        con_mat = confusion_matrix(
            y_true=y_input_int, 
            y_pred=y_preds, 
            normalize = 'pred')

        return acc, con_mat