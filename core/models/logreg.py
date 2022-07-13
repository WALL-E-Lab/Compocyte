from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scipy
import os
import pickle

class LogRegWrapper():

    possible_data_types = ['counts']
    data_type = 'counts'
    input_as_adata = False

    def save(self, save_path, name):
        timestamp = str(time()).replace('.', '_')
        model_path = os.path.join(
            self.save_path, 
            'models',
            name,
            timestamp
        )
        settings_dict = {'classifier_type': 'LogReg'}
        for key in self.__dict__.keys():
            if key == 'model':
                pickle.dump(self.__dict__[key], os.path.join(model_path, 'model.pickle'))
                settings_dict[key] = os.path.join(model_path, 'model.pickle')

            else:
                settings_dict[key] = self.__dict__[key]

        pickle.dump(settings_dict, os.path.join(model_path, 'classifier_settings.pickle'))

    def __init__(self, **kwargs):
        self.model = LogisticRegression(class_weight='balanced', random_state=1, max_iter=1e4)

    def train(self, x, y_onehot=None, y=None, y_int=None):
        self.model.fit(x, y)

    def validate(self, x, y, **kwargs):
        return (0,0)

    def predict(self, x_test):

        y_pred = self.model.predict(x_test)

        return y_pred 