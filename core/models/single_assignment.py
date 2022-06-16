import numpy as np

class SingleAssignment():

    possible_data_types = ['counts']
    data_type = 'counts'
    input_as_adata = False

    def __init__(self, assignment, **kwargs):
        self.assignment = assignment

    def train(self, **kwargs):
        pass

    def validate(self, x, y, **kwargs):
        return (0,0)

    def predict(self, x_test):
        y_pred = np.array([self.assignment for i in range(len(x_test))])

        return y_pred 