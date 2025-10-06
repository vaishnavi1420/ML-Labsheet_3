

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

def plot_boundary(clf, X, y, title):
    # Only works for 2D data
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,200), np.linspace(y_min,y_max,200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx,yy,Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k')
    plt.title(title)
    plt.show()

def main():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # reduce to 2D for visualization
    y = iris.target
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = {
        'linear': svm.SVC(kernel='linear'),
        'poly': svm.SVC(kernel='poly', degree=3),
        'rbf': svm.SVC(kernel='rbf')
    }
    for name, model in models.items():
        model.fit(Xs, y)
        print('Training', name)
        plot_boundary(model, Xs, y, f'SVM {name} kernel')

    # grid search for RBF
    param_grid = {'C':[0.1,1,10], 'gamma':['scale','auto',0.1,1]}
    gs = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=3)
    gs.fit(Xs, y)
    print('Best params (RBF):', gs.best_params_)

if __name__ == '__main__':
    main()
