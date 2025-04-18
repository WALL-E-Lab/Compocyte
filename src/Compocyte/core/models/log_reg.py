import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
import pickle
import os

class LogisticRegression():

    def __init__(self, labels=None, labels_enc=None, labels_dec=None, model=None, C=1.0, solver='sag', max_iter=1000, n_jobs=None, logreg_kwargs={}, **kwargs):
        if model is not None:
            self.labels_enc = labels_enc
            self.labels_dec = labels_dec
            self.model = model

        else:
            if labels is None:
                raise ValueError('Labels must be provided if model is not provided')
            
            self.labels_enc = {label: i for i, label in enumerate(labels)}
            self.labels_dec = {self.labels_enc[label]: label for label in self.labels_enc.keys()}
            self.model = LogReg (
                C=C, 
                solver=solver, 
                max_iter=max_iter, 
                multi_class='ovr', 
                n_jobs=n_jobs, 
                **logreg_kwargs)

    def predict_logits(self, x) -> np.array:
        pred_activations = self.model.predict_proba(x)
            
        return pred_activations

    def _save(self, path):
        for attribute in ['model', 'labels_enc', 'labels_dec']:
            with open(os.path.join(path, f'{attribute}.pickle'), 'wb') as f:
                pickle.dump(
                    getattr(self, attribute, None), f)

    @classmethod
    def _load(cls, path):
        args = {}
        for attribute in ['model', 'labels_enc', 'labels_dec']:
            with open(os.path.join(path, f'{attribute}.pickle'), 'rb') as f:
                args[attribute] = pickle.load(f)

        log_reg = cls(**args)

        return log_reg
