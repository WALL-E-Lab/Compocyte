import os
import pickle
import numpy as np


class DummyClassifier():
    def __init__(self, labels: list, **kwargs):
        """Dummy classifier for test cases in which there is only one available child \
        label. Will return the available label as prediction for all input cells with \
        a simulated activation of 1.

        Args:
            labels (list): Labels available during training. Should have length 1 to \
            use DummyClassifier.
        """

        self.labels = labels[0]
        self.labels_enc = {label: i for i, label in enumerate(labels)}

    def fit(self, *args, **kwargs):
        pass

    def predict_logits(self, x: np.array) -> np.array:
        return np.ones(
            shape=(x.shape[0], 1)
        )

    def predict(self, x: np.array, **kwargs) -> np.array:
        pred = np.array([self.labels[0]] * x.shape[0])

        return pred
    
    def _save(self, path):
        for attribute in ['labels', 'labels_enc']:
            with open(os.path.join(path, f'{attribute}.pickle'), 'wb') as f:
                pickle.dump(
                    getattr(self, attribute, None), f)

    @classmethod
    def _load(cls, path):
        args = {}
        for attribute in ['labels', 'labels_enc']:
            with open(os.path.join(path, f'{attribute}.pickle'), 'rb') as f:
                args[attribute] = pickle.load(f)

        classifier = cls(**args)

        return classifier