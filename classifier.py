
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection as sk_ms

class Classification(object):

    def __init__(self, features, labels, kfold):
        self.features = features
        self.labels = labels
        self.kfold = kfold

    def classification(self):
        c1 = DecisionTreeClassifier(random_state=0)
        c2 = KNeighborsClassifier(n_neighbors=5) ## testar outros parametros 3 41.6666666  ### 5 45.
        c3 = GaussianNB()
        c4 = SVC(kernel='linear', probability=True)
        classifiers = [c1,c2,c3,c4]
        results = []
        for i in classifiers:
            scores = sk_ms.cross_val_score(i, self.features, self.labels, cv=self.kfold, scoring='accuracy', n_jobs=-1, verbose=0)
            score = round(scores.mean() * 100, 2)
            results.append(score)
        return results

    def get_scores(self):
        return np.array(self.classification())
