from sklearn.linear_model import LogisticRegression
import pickle
import os

class LogisticRegression():

    possible_data_types = ['counts', 'normlog']

    def __init__(self, data_type, model=None, C=1.0, solver=None, max_iter=None, n_jobs=None, **kwargs):
        if model is None:
            self.model = LogisticRegression(
                C=C, 
                solver=solver, 
                max_iter=max_iter, 
                multi_class='ovr', 
                n_jobs=n_jobs, 
                **kwargs)
        
        else:
            self.model = model

        self.data_type = data_type

    def _train(
        self,
        x,
        y_onehot,
        y_int,
        **kwargs):
        
        self.model.fit(x, y_int)

    def predict(self, X):
        pred_activations = self.model.predict_proba(X)
        return pred_activations

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
