from matplotlib import pylab as pl
from itertools import cycle
import numpy as np
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics 

class MLPlot(object):
    
    
    '''Various plotting functions for Scikit-Learn Machine Learning Functions:
   
    MLPlot().preview() = Classification based plotting.
    
    MLPlot().pca_plot() = Principal Component Analysis based plotting.
    
    MLPlot().grid_knn() = Combined Grid Search and Visualisation for Nearest Neighbors.
                          
    MLPlot().grid_svr() = Combined Grid Search and Visualisation for Support Vector Regression.                          ''' 
   
      
    
    def __init__(self):
        pass
        

    
    def preview_plot(self, data, X, y, feature_num1, feature_num2):
        
        '''Classification - Initial view of 1 vs 1 features for each classifier.
        Gives a plot of two features plotted against each other for each of the two
        (binary) target values.
        
        Parameters
        __________
        
        data = dataset: eg DataFrame
        
        X = X
        
        y = y
        
        feature_num1 = 1st feature to view.
        
        feature_num2 = 2nd feature to view.
        
        '''
        
        header = data.columns.values
        
        pl.figure()
        pl.scatter(X[y==0, feature_num1], X[y==0, feature_num2], color = 'b', label='Class 0')
        pl.scatter(X[y==1, feature_num1], X[y==1, feature_num2], color = 'r', label='Class 1')
        pl.xlabel(header[feature_num1])
        pl.ylabel(header[feature_num2])
        pl.legend()
        pl.show()
    

    def pca_plot(self, X, y, target_names):
        
        '''Visualise  2 Principal Components PCA output.
        
        Parameters
        __________
        
        X = Features / dataset
        
        y = target classifiers
        
        target_names = list of target names as strings'''

        colors = cycle('rgbcmykw')
        ids = range(len(target_names))
        pl.figure()
        for i, c, label in zip(ids, colors, target_names):
            pl.scatter(X[y == i, 0], X[y == i, 1],
                                          c=c, label=label)
        pl.legend()
        pl.show()



    def grid_knn(self,k,Xt,Yt,Xtest,Ytest, cv = 5):
        '''Combined Grid Search and visualisation for most K Nearest Neighbors.
        
        Parameters
        __________
        
        k = Max K nearest neighbors search value
        
        Xt = Xtrain
        
        yt = Ytrain'''
        
        Xtrain = Xt
        Ytrain = Yt
        
    
        n_neighbors = np.arange(1,k,2)
        weights = ['uniform','distance']

        parameters = [{'n_neighbors':n_neighbors, 'weights':weights}]

        grid = GridSearchCV(KNeighborsClassifier(), parameters, cv = cv)
        
        grid.fit(Xtrain,Ytrain)
        
        bestn = grid.best_params_['n_neighbors']
        bestw = grid.best_params_['weights']
        
        print "optimal n_neighbors:", bestn
        print "optimal weight:", bestw
            
        scores = [x[1] for x in grid.grid_scores_]
        scores = np.array(scores).reshape(len(n_neighbors), len(weights))
        scores = np.transpose(scores)
        
        pl.figure(figsize=(12, 6))
        pl.imshow(scores, interpolation='nearest', origin='higher', cmap=pl.cm.get_cmap('jet_r'))
        pl.xticks(np.arange(len(n_neighbors)), n_neighbors)
        pl.yticks(np.arange(len(weights)), weights)
        pl.xlabel('Value of "n_neighbors"')
        pl.ylabel('weights')
        
        cbar = pl.colorbar()
        cbar.set_label('Classification Accuracy', rotation=270, labelpad=20)
        
        pl.show()
        
        optimal_knn = KNeighborsClassifier(n_neighbors=bestn, weights = bestw)
        optimal_knn.fit(Xtrain,Ytrain)
        opt_pred = optimal_knn.predict(Xtest)
        print "this is performance of optimal classifier on Ytest", metrics.classification_report(Ytest, opt_pred)

    def grid_svr(self,c ,e, Xt, Yt,cv=5):
        """Combined Grid Search and Visualisation for Support Vector Regression.
        
        Parameters
        __________
        
        c = Max value for C
        
        e = Max value for epsilon
        
        cv = Cross Validation no: (optional)
        
        Xt = Xtrain
        
        Yt = Ytrain
        """
        c = float(c)
        e = float(e)
        Xtrain = Xt
        Ytrain = Yt
        
        C_ = np.arange(0.1,c,0.1)
        epsilon_ = np.arange(0,e,0.1)
        kernel_ = ['linear','poly','rbf']
        
        parameters = [{'C':C_, 'epsilon':epsilon_,'kernel':kernel_}]
        
        grid = GridSearchCV(SVR(), parameters, cv = cv)
        
        grid.fit(Xtrain,Ytrain)
        
        bestc = grid.best_params_['C']
        beste = grid.best_params_['epsilon']
        bestk = grid.best_params_['kernel']
        print bestc
        print beste
        print bestk
        
        
        scores = [x[1] for x in grid.grid_scores_]
        scores = np.array(scores).reshape(len(C_), len(epsilon_))
        scores = np.transpose(scores)
        
        pl.figure(figsize=(12, 6))
        pl.imshow(scores, interpolation='nearest', origin='higher', cmap=pl.cm.get_cmap('jet_r'))
        pl.xticks(np.arange(len(C_)), C_)
        pl.yticks(np.arange(len(epsilon_)), epsilon_)
        pl.xlabel('Value of "C"')
        pl.ylabel('epsilon')
        
        cbar = pl.colorbar()
        cbar.set_label('Classification Accuracy', rotation=270, labelpad=20)
        
        pl.show()