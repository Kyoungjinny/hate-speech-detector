#import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
#from .prep import export
#import os.path
#import sys
#file_dir = os.path.dirname(__file__)
#sys.path.append(file_dir)


def confusionMatrix():
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    #y_train = np.load('y_train.npy')

    loaded_model = load_model('lstm_model.h5', compile=False)
    predictions = loaded_model.predict(X_test)
    y_pred = (predictions > 0.5)

    a = accuracy_score(y_test, y_pred)
    print(a)
    p = precision_score(y_test, y_pred)
    #print(p)
    r = recall_score(y_test, y_pred)
    #print(r)
    f1 = f1_score(y_test, y_pred)
    #print(f1)

    #print(classification_report(y_test, y_pred, target_names=['hate', 'none']))

    score = []
    score.append(a)
    score.append(p)
    score.append(r)
    score.append(f1)
    return score

if __name__ == '__main__':
    cm = confusionMatrix()