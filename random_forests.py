import random
import collections    
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier as rfc
import numpy as np
import numpy 



class RandomForest:

    def __init__(self, n_trees=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None,
        random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, max_samples=None):

        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.max_samples = max_samples

        self.trees = None

    
    def fit(self, X, y):
        """
        param X: iterable of type list, np.array, sparse matrix, etc.
        param y: iterable of type list, np.array, sparse matrix, etc.
        """
        self.trees = []
        data_indices = list(range(X.shape[0]))

        if not self.max_samples:
            # max_samples == None
            self.max_samples = X.shape[0]
        elif self.max_samples <= 1.0:
            # max_samples is float 
            self.max_samples = X.shape[0]*self.max_samples
        else:
            # max_samples is int
            self.max_samples = self.max_samples if self.max_samples <= X.shape[0] else X.shape[0]

        random.seed(self.random_state)

        for n in range(self.n_trees):
            # fit n_trees to random bootstrap of data
            
            sample = random.sample(data_indices, self.max_samples)

            tree = dtc(criterion = self.criterion, max_depth = self.max_depth, min_samples_split = self.min_samples_split, 
                min_samples_leaf = self.min_samples_leaf, max_features = self.max_features, random_state = self.random_state,
                max_leaf_nodes = self.max_leaf_nodes, min_impurity_decrease = self.min_impurity_decrease, 
                min_impurity_split = self.min_impurity_split, 
                class_weight = self.class_weight).fit(X[sample], y[sample])

            self.trees.append(tree)

        return self


    def predict(self, X)->np.array:
        """
        param X: iterable of type list, np.array, sparse matrix, etc.
        """
        forest_predictions = np.array([])

        for x in X:
            # get prediction for each X
            prediction_sum = 0

            for tree in self.trees:
                prediction_sum += int(tree.predict(x)[0])

            if prediction_sum / self.n_trees < 0.5:
                forest_predictions = np.append(forest_predictions, 0)
            else:
                forest_predictions = np.append(forest_predictions, 1)

        return forest_predictions


    def score(self, X, y):
        """
        param X: iterable of type list, np.array, sparse matrix, etc.
        param y: iterable of type list, np.array, sparse matrix, etc.
        """
        predictions = self.predict(X)
        score = 0

        for prediction, y_ in zip(predictions, y):
            if prediction == int(y_):
                score+=1

        return score/y.shape[0]
    
    
class RandomForest_knn(RandomForest):

    def __init__(self, n_trees=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None,
        random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, max_samples=None, k=0.2):

        super().__init__(n_trees, criterion, max_depth, min_samples_split, min_samples_leaf, max_features,
        random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight, max_samples)
        
        self.knn = None
        self.data = None
        self.labels = None
 
        self.k = k
    
#         print(locals())

    def fit(self, X_train, y_train, metric='cosine'):
        """
        Constructs a K-nearest neighbors relationship.
        k is class attribute - assumes variable k will not be of significant benefit

        param X_train: iterable of type list, np.array, sparse matrix, etc.
        param y_train: iterable of type list, np.array, sparse matrix, etc.
        param metric: str, default 'cosine',
            metric used for evaluating neighbor distances. from sklearn.metrics
        """
        
        if type(self.k) != int:
            self.k = int(np.ceil(X_train.shape[0]*self.k))
        
        self.knn = NearestNeighbors(n_neighbors = self.k, metric = metric)
        self.knn.fit(X_train)
        self.labels = y_train
        self.data = X_train

        return self


    def predict(self, X):
        """
        param X: iterable of type list, np.array, sparse matrix, etc.
        """
        predictions = np.array([])
        for x in X:
            distances, indices = self.knn.kneighbors(x)

            X_train = self.data[indices[0]]
            y_train = self.labels[indices[0]]
            
            super().fit(X_train, y_train)
            pred = super().predict(x)[0]

            predictions = np.append(predictions, pred)

        return predictions


    def score(self, X, y):
        """
        param X: iterable of type list, np.array, sparse matrix, etc.
        param y: iterable of type list, np.array, sparse matrix, etc.
        """
        return super().score(X, y)
    
        
class RandomForest_skl_knn(rfc):

    def __init__(self, n_trees=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                 min_impurity_split=None, class_weight=None, max_samples=None, k=0.2, n_jobs=None):

        super().__init__(n_trees, criterion, max_depth, min_samples_split, min_samples_leaf, max_features,
        random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight, max_samples,
                        n_jobs)
        
        self.knn = None
        self.data = None
        self.labels = None
 
        self.k = k

    def fit(self, X_train, y_train, metric='cosine'):
        """
        Constructs a K-nearest neighbors relationship.
        k is class attribute - assumes variable k will not be of significant benefit

        param X_train: iterable of type list, np.array, sparse matrix, etc.
        param y_train: iterable of type list, np.array, sparse matrix, etc.
        param metric: str, default 'cosine',
            metric used for evaluating neighbor distances. from sklearn.metrics
        """
        print(type(self.k))
        
        if type(self.k) != int:
            self.k = int(np.ceil(X_train.shape[0]*self.k))
        
        self.knn = NearestNeighbors(n_neighbors = self.k, metric = metric)
        self.knn.fit(X_train)
        self.labels = y_train
        self.data = X_train

        return self


    def predict(self, X):
        """
        param X: iterable of type list, np.array, sparse matrix, etc.
        """
        predictions = np.array([])
        for x in X:
            
            distances, indices = self.knn.kneighbors(x)

            X_train = self.data[indices[0]]
            y_train = self.labels[indices[0]]
            
            super().fit(X_train, y_train)
            pred = super().predict(x)[0]

            predictions = np.append(predictions, pred)

        return predictions


    def score(self, X, y):
        """
        param X: iterable of type list, np.array, sparse matrix, etc.
        param y: iterable of type list, np.array, sparse matrix, etc.
        """
        return super().score(X, y)
    