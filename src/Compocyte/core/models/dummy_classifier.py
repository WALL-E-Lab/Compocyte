import numpy as np


class DummyClassifier():
    def __init__(self, labels: list):
        """Dummy classifier for test cases in which there is only one available child \
        label. Will return the available label as prediction for all input cells with \
        a simulated activation of 1.

        Args:
            labels (list): Labels available during training. Should have length 1 to \
            use DummyClassifier.
        """

        self.label = labels[0]

    def fit(self, *args, **kwargs):
        pass

    def predict_logits(self, x: np.array) -> np.array:
        return np.ones(
            shape=(x.shape[0], 1)
        )

    def predict(self, x: np.array, **kwargs) -> np.array:
        pred = np.array([self.label] * x.shape[0])

        return pred