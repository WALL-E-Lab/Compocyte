import tensorflow.keras as keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .tools import z_transform_properties

class NodeMemory():
    """Add explanation.
    """

    def __init__(self, x_input_data, y_input_data):
        """Params: x_input and y_input for whole (ie not splitted) data set
        """

        self.x_input_data = x_input_data  
        self.y_input_data = y_input_data
        self.bool_processed = False

    def _get_raw_x_input_data(self):
        return self.x_input_data 

    def _get_raw_y_input_data(self):
        return self.y_input_data

    def _set_y_input_grouped_labels(self, y_input_grouped_labels):
        """Should only be run once, sets and fits LabelEncoder to node specific 
        output labels
        """

        self.y_input_grouped_labels = y_input_grouped_labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.y_input_grouped_labels)

    def _get_y_input_grouped(self):
        return self.y_input_grouped_labels

    def _set_processed_input_data(self):
        """Processes y- and x- input data (encoding and 
        standardization/transforming)
        """

        self.y_input_data_int = self.label_encoder.transform(
            self.y_input_grouped_labels)
        self.y_input_data_onehot = keras.utils.to_categorical(
            self.y_input_data_int)
        self.x_input_data = z_transform_properties(self.x_input_data)
        self.bool_processed = True

    def _get_y_input_data_int(self):
        if self.bool_processed:
            return self.y_input_data_int 

        else:
            print('Warning! Input Data not yet processed, cannot return \
                y_input_data_int.')

    def _get_y_input_onehot(self):
        if self.bool_processed:
            return self.y_input_data_onehot

        else:
            print('Warning! Input Data not yet processed, cannot return \
                y_input_data_onehot.')

    def _set_output_len_of_node(self, output_len):
        self.output_len = output_len

    def _get_output_len_of_node(self):
        return self.output_len

    def _set_prediction_vector(self, prediction_vec):
        """Model should be fitted to all data once more after kfold CV
        """

        self.pred_vec = prediction_vec

    def _get_prediction_vector(self):
        return self.pred_vec

    def _set_trainings_accuracy(self, acc):
        self.training_acc = acc

    def _get_trainings_accuracy(self):
        return self.training_acc 

    def _set_test_accuracy(self, acc):
        self.test_acc = acc

    def _get_test_accuracy(self):
        return self.test_acc

    def _set_confusion_matrix(self, conmat):
        self.conmat = conmat

    def _get_confusion_matrix(self):
        return self.conmat

    def _set_training_conmat(self, train_conmat):
        self.train_conmat = train_conmat 

    def _get_training_conmat(self):
        return self.train_conmat 

    def _set_test_conmat(self, test_conmat):
        self.test_conmat = test_conmat 

    def _get_test_conmat(self):
        return self.test_conmat

    def _delete_incorrect_labeled_cells(self, idx_incorrect_cells):
        """Deletes cells at indices given by array idx_incorrect_cells
        """

        self.x_input_data = np.delete(np.asarray(self.x_input_data), idx_incorrect_cells, axis=0)
        self.y_input_data = np.delete(np.asarray(self.y_input_data), idx_incorrect_cells, axis=0)

    def _setup_local_classifier(self, local_classifier):
        self.local_classifier = local_classifier

    def _get_indexed_data(self, index_vec):
        """Returns sliced array of saved data sets
        _________________________________
        Params:
        ---------------------------------
        index_vec: array_like, indices of array entries of data that will be kept

        _________________________________
        Returns:
        ---------------------------------
        tuple (X_int, y_int, y_onehot): sliced input data sets
        """

        if self.bool_processed:
            X = np.asarray(self.x_input_data)[index_vec]
            y_int = np.asarray(self.y_input_data_int)[index_vec]
            y_onehot = np.asarray(self.y_input_data_onehot)[index_vec]

        else:
            raise Exception('Warning, data in Node_Memory not yet processed.')

        return X, y_int, y_onehot