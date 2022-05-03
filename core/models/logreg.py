from sklearn.linear_model import LogisticRegression as LR


class LogRegWrapper():

    def __init__():
        pass 

    def data_preparation():
        pass

    def train(self, x, y, **kwargs):
        x = x.copy()
        self.model = LR.fit(x, y) 
    
    def validate():
        pass 

    def predict(self, x):
        x = x.copy()
        pred = self.model.predict_proba()

        print(f"calling from predict in LogRegWrapper: \n pred vec is \n {pred}")

        return pred 
