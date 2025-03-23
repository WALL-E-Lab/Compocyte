import numpy as np
from catboost import CatBoostClassifier, metrics
import pickle
import os

class BoostedTrees():
    def __init__(self, labels=None, labels_enc=None, labels_dec=None, model=None, catboost_kwargs={}, **kwargs):
        if model is not None:
            self.labels_enc = labels_enc
            self.labels_dec = labels_dec
            self.model = model

        else:
            if labels is None:
                raise ValueError('Labels must be provided if model is not provided')
            
            self.labels_enc = {label: i for i, label in enumerate(labels)}
            self.labels_dec = {self.labels_enc[label]: label for label in self.labels_enc.keys()}
            self.model = CatBoostClassifier (
                custom_loss=[metrics.Accuracy()],
                random_seed=42,
                logging_level='Silent',
                *catboost_kwargs)

    def predict_logits(self, x) -> np.array:
        pred_activations = self.model.predict_proba(x)
            
        return pred_activations

    def _save(self, path):
        self.model.save_model(os.path.join(path, 'model.cbm'), format='cbm')
        for attribute in ['labels_enc', 'labels_dec']:
            with open(os.path.join(path, f'{attribute}.pickle'), 'wb') as f:
                pickle.dump(
                    getattr(self, attribute, None), f)

    @classmethod
    def _load(cls, path):
        args = {}
        args['model'] = CatBoostClassifier.load_model(os.path.join(path, 'model.cbm'))
        for attribute in ['labels_enc', 'labels_dec']:
            with open(os.path.join(path, f'{attribute}.pickle'), 'rb') as f:
                args[attribute] = pickle.load(f)

        trees = cls(**args)

        return trees
