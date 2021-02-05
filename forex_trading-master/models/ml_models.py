import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import auc, accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
from scipy import stats


def kendaltau(gt, pred, *args, **kwargs):
    
    #print(args)
    #print(kwargs)
    tau, p_value = stats.kendalltau(gt, pred)
    tau = tau / 2 + 0.5
    
    return tau

'''
regressors = {'gb': GradientBoostingRegressor(random_state=1),
              'rf': RandomForestRegressor(random_state=1),
              'lgb': GridSearchCV(lgb.LGBMRegressor(),
                                  {'min_child_samples': [0, 2],
                                   'class_weight':['balanced'],
                                   'max_depth': [1, 2],
                                   'num_leaves': [3, 7],
                                   'min_split_gain': [0, 0.1],
                                   'min_child_weight': [0],
                                   'reg_alpha': [0.01, 0.1],
                                   'reg_lambda': [0.01, 0.1],
                                   'n_estimators': [16, 64],}),
              'linear': GridSearchCV(LinearRegression(),
                                  {'normalize': [True, False]}),
              'svr': GridSearchCV(SVR(),
                                  {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                  'C': [0.1, 1, 10]}),
              'mlp': MLPRegressor(random_state=1, max_iter=500, activation='logistic'),
              'lasso': GridSearchCV(Lasso(),
                                  {'alpha': [0.001, 0.01, 0.1]}),
              'kernel_ridge': GridSearchCV(KernelRidge(),
                                  {'alpha': [0.1, 1, 10]})
              }
'''
regressors = {'gb': GradientBoostingRegressor(random_state=1),
              'rf': RandomForestRegressor(random_state=1),
              'lgb': lgb.LGBMRegressor(),
              'linear': LinearRegression(),
              'svr': SVR(kernel='linear', C=1),
              'mlp': MLPRegressor(random_state=1, max_iter=500, activation='logistic'),
              'lasso': Lasso(alpha=1),
              'kernel_ridge': KernelRidge(alpha=1)
              }

regression_metrics = {'mae': (mean_absolute_error, False),
                      'mse': (mean_squared_error, False),
                      'r2': (r2_score, True),
                      'max_error': (max_error, False),
                      'kendaltau': (kendaltau, True),
                     }
                      


classifiers = {'knn': KNeighborsClassifier(3),
               'lgb': lgb.LGBMClassifier(),
               'linear_svc': SVC(kernel="linear", C=0.025),
               'gamma_svc': SVC(gamma=2, C=1),
               'gaussiance': GaussianProcessClassifier(1.0 * RBF(1.0)),
               'dt': DecisionTreeClassifier(max_depth=5),
               'rf': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               'mlp': MLPClassifier(alpha=1, max_iter=1000),
               'ada': AdaBoostClassifier(),
               'nb': GaussianNB()}

binary_classification_metrics = {'roc_auc': (roc_auc_score, True),
                                 'f1': (f1_score, True),
                                 'acc': (accuracy_score, True)}



