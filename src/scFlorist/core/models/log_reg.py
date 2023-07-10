from sklearn.linear_model import LogisticRegression
from torch.nn.functional import softmax
import pickle
import os
import torch

class LogisticRegression():
    def __init__(self, model, data_type):
        self.model = model
        self.data_type = data_type

    def _train(
        self,
        x,
        y_onehot,
        y_int,
        **kwargs):
        # This method exists only for compatibility. Log reg models can only be imported and used to predict.

        print('Classifier not trained. Within this module imported log reg models cannot be iteratively trained or adjusted to a new hierarchy.')

    def predict(self, x):
        pred_activations = self.model.predict_log_proba(x)
        return softmax(torch.Tensor(pred_activations), dim=1).detach().cpu().numpy()

    def _save(self, path):
        with open(os.path.join(path, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)

        with open(os.path.join(path, 'data_type.pickle'), 'wb') as f:
            pickle.dump(self.data_type, f)

    @classmethod
    def _load(cls, path):
        with open(os.path.join(path, 'model.pickle'), 'rb') as f:
            model = pickle.load(f)

        with open(os.path.join(path, 'data_type.pickle'), 'rb') as f:
            data_type = pickle.load(f)

        log_reg = cls(model, data_type)

        return log_reg