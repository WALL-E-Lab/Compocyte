from sklearn.linear_model import LogisticRegression as LogReg
import pickle
import os
import numpy as np

class LogisticRegression():

    possible_data_types = ['counts', 'normlog']

    def __init__(self, data_type=None, model=None, C=1.0, solver='sag', max_iter=1000, n_jobs=None, logreg_kwargs={}, **kwargs):
        self.fixed = None
        if model is None:
            self.model = LogReg (
                C=C, 
                solver=solver, 
                max_iter=max_iter, 
                multi_class='ovr', 
                n_jobs=n_jobs, 
                **logreg_kwargs)
            self.data_type = 'normlog'
        
        else:
            self.model = model
            if data_type is None:
                raise Exception('Must provide data type for import.') 

    def _train(
        self,
        x,
        y_onehot,
        y_int,
        **kwargs):
        
        # Cannot train logreg model with only one class present
        # Most sensible approach is to set all predictions in the future to the only learned class
        if len(np.unique(y_int)) == 1:
            self.fixed = 0

        else:
            self.model.fit(x, y_int)

    def predict(self, X):
        if self.fixed is None:
            pred_activations = self.model.predict_proba(X)

        else:
            pred_activations = np.ones(shape=(len(X), 1))
            
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

        log_reg = cls(model=model, data_type=data_type)

        return log_reg
