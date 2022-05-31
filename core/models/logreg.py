from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scipy


class LogRegWrapper():

    def __init__():
        pass 

    def train(self, x, y):
        self.LR = LogisticRegression(class_weight='balanced', random_state=1, max_iter=1e4)
        self.LR.fit(x, y)

    
    def validate(self, x, y, **kwargs):
        return (0,0)
    
    def predict(self, x_test):
        
        y_pred = self.LR.predict(x_test)

        return y_pred