import numpy as np
from time import time
import os
import pickle

class SingleAssignment():

    possible_data_types = ['counts']
    data_type = 'counts'
    input_as_adata = False

    def save(self, save_path, name):
        timestamp = str(time()).replace('.', '_')
        model_path = os.path.join(
            save_path, 
            'models',
            name,
            timestamp
        )
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        settings_dict = {'classifier_type': 'SA'}
        for key in self.__dict__.keys():
            settings_dict[key] = self.__dict__[key]

        with open(os.path.join(model_path, 'classifier_settings.pickle'), 'wb') as f:
            pickle.dump(settings_dict, f)

    def __init__(self, assignment, **kwargs):
        self.assignment = assignment

    def train(self, **kwargs):
        pass

    def validate(self, x, y, **kwargs):
        return (0,0)

    def predict(self, x_test):
        y_pred = np.array([self.assignment for i in range(len(x_test))])

        return y_pred 