import tensorflow.keras as keras
import numpy as np
from sklearn.metrics import confusion_matrix
# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.under_sampling import TomekLinks, NearMiss
# from imblearn.tensorflow import balanced_batch_generator

class NeuralNetwork():
    """Add explanation.
    """

    def __init__(
        self, 
        len_of_input,
        len_of_output, 
        list_of_hidden_layer_nodes=[64, 64],
        activation_function='relu',
        learning_rate=0.001,
        momentum=.9,
        dropout=0.4,
        batch_size=60,
        batch_norm=True,
        l2_reg=True,
        leakiness_ReLU=0.1,
        loss_function='categorical_crossentropy',
        epochs=500,
        **kwargs
    ):
        """Add explanation.
        """

        self.len_of_input = len_of_input
        self.len_of_output = len_of_output
        self.list_of_layer_nodes = [len_of_input] + [nodes for nodes in list_of_hidden_layer_nodes]\
            + [len_of_output]  
        self.activation_function = activation_function 
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.leakiness_ReLU = leakiness_ReLU
        self.loss_function = loss_function
        self.epochs = epochs
        self.model = keras.models.Sequential()
        if self.leakiness_ReLU > 0.0 and self.activation_function == 'relu':
            self.activation_function = lambda x: keras.activations.relu(
                x, 
                alpha=self.leakiness_ReLU)

        for nodes, layer_idx in zip(
            self.list_of_layer_nodes, 
            range(0, len(self.list_of_layer_nodes) - 1)
        ):
            if layer_idx != len(self.list_of_layer_nodes) - 2:
                activation = self.activation_function

            else:
                activation = 'softmax'

            if l2_reg != True:
                regularizer = None

            else:
                regularizer = keras.regularizers.l2(l2=0.01)

            self.model.add(keras.layers.Dense(
                input_shape = (nodes,),
                units = self.list_of_layer_nodes[layer_idx+1],                            
                kernel_initializer = 'glorot_uniform',                                          
                bias_initializer = 'zeros',                                               
                activation = activation,
                activity_regularizer=regularizer))
            if layer_idx != len(self.list_of_layer_nodes) - 2:
                if self.batch_norm == True:
                    self.model.add(keras.layers.BatchNormalization())

                self.model.add(keras.layers.Dropout(
                    self.dropout
                ))

    def train(self, x, y_onehot, **kwargs):
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
            x,
            y_onehot,
            batch_size=self.batch_size,
            epochs = self.epochs, 
            verbose = 0,
            validation_split = .1,
            callbacks=[early_stopping_callback])

    def predict(self, input_vec):
        """Calculate and return label prediction of trained model for an input
        vector input_vec (dtype=int)"""

        # Returns only absolute decisions, nothing known about the difference 
        # in prediction confidence 
        pred_vec = np.argmax(self.model.predict(input_vec), axis = -1)

        return pred_vec

    def predict_proba(self, input_vec):
        """Calculate and return probabilities of label predictions of trained model for an input
        vector input_vec (dtype=int)"""

        proba_pred_vec = self.model.predict(input_vec)

        return proba_pred_vec

    def validate(self, x, y_int, **kwargs):
        """Add explanation.
        """

        y_preds = self.predict(x)
        def calc_acc(pred_vec, known_vec):
            if type(pred_vec) == type(known_vec):
                acc = np.sum(pred_vec == known_vec, axis = 0) / len(known_vec)

            else:
                raise ValueError('self.validate: Error! \
                    Comparison of different label encoding!')

            return acc

        acc = calc_acc(y_preds, y_int)
        con_mat = confusion_matrix(
            y_true=y_int, 
            y_pred=y_preds, 
            normalize = 'true')

        return acc, con_mat