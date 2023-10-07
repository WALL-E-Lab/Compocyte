from sklearn.linear_model import LogisticRegression as LogReg
import pickle
import os
import numpy as np

class LogisticRegression():

    possible_data_types = ['counts', 'normlog']

    def __init__(self, n_output, data_type=None, fixed=None, model=None, C=1.0, solver='sag', max_iter=1000, n_jobs=None, logreg_kwargs={}, **kwargs):
        self.fixed = fixed
        self.n_output = n_output
        self.data_type = data_type
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
            self.fixed = np.unique(y_int)[0]

        else:
            self.model.fit(x, y_int)

    def predict(self, X):
        if self.fixed is None:
            pred_activations = self.model.predict_proba(X)
            # Augment prediction matrix if one of the child classes
            # was not present in training data to avoid a mismatch
            # between the idx in prediction matrix and the i in
            # for label encoding
            if len(self.model.classes_) != self.n_output:
                augmented_pred_activations = np.zeros(
                    shape=(len(pred_activations), self.n_output))
                for idx, label_i in enumerate(self.model.classes_):
                    activations_class = pred_activations[:, idx]
                    augmented_pred_activations[:, label_i] = activations_class

                return augmented_pred_activations

        else:
            pred_activations = np.zeros(
                    shape=(len(X), self.n_output))
            pred_activations[:, self.fixed] = 1
            
        return pred_activations

    def _save(self, path):
        for attribute in ['model', 'data_type', 'n_output', 'fixed']:
            with open(os.path.join(path, f'{attribute}.pickle'), 'wb') as f:
                pickle.dump(
                    getattr(self, attribute, None), f)

    @classmethod
    def _load(cls, path):
        args = {}
        for attribute in ['model', 'data_type', 'n_output', 'fixed']:
            with open(os.path.join(path, f'{attribute}.pickle'), 'rb') as f:
                args[attribute] = pickle.load(f)

        log_reg = cls(**args)

        return log_reg
