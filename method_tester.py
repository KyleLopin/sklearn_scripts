# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge, Lars, LassoLars
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.linear_model import ElasticNet, ARDRegression, BayesianRidge
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, ridge_regression
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, ShuffleSplit, KFold
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import LinearSVR, NuSVR, SVR

# local files
import data_getter
import processing

plt.style.use('seaborn')

# data = pd.read_csv('as7262_mango.csv')
# data = data.groupby('Leaf number', as_index=True).mean()

x_data, data = data_getter.get_data("as7262 mango", remove_outlier=True,
                                    only_pos2=False)

data_columns = []
for column in data.columns:
    if 'nm' in column:
        data_columns.append(column)
print(data_columns)
# x_data = data[data_columns]

y_columns = ['Total Chlorophyll (ug/ml)',
             'Chlorophyll a (ug/ml)',
             'Chlorophyll b (ug/ml)']



invert_y = False
# x_data = np.log(x_data)
conditions = "Partial Least Squared"
if invert_y:
    conditions += "\nInverted Y"
    modeler = TransformedTargetRegressor(regressor=PLS(n_components=3),
                                         func=inverse,
                                         inverse_func=inverse)
else:
    modeler = PLS(n_components=1)
    modeler_name = "Partial Least Squared"

modeler = TransformedTargetRegressor(regressor=PLS(n_components=3),
                                     func=np.reciprocal,
                                     inverse_func=np.reciprocal)


def plot_learning_curve(estimator, axis, X, y, ylim=None, cv=None,
                        group=None, scorer=make_scorer(mean_absolute_error),
                        n_jobs=None, train_sizes=np.linspace(.35, 1.0, 5),
                        final_y_predict=False, add_legend=False):
    """
    from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    # print("hello", estimator.__class__.__name__)
    # print(dir(estimator))
    # if estimator.__class__.__name__ == "TransformedTargetRegressor":
    #     print(estimator._estimator_type, estimator.regressor.__class__.__name__)
    # if ylim is not None:
    #     axis.ylim(*ylim)
    axis.set_xlabel("Training examples")
    axis.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        groups=group, shuffle=True, scoring=scorer)
    # explained_variance, neg_median_absolute_error
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if final_y_predict:
        axis.scatter(y.shape[0], final_y_predict, color='r', marker="+")

    axis.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    axis.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axis.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    axis.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    if add_legend:
        axis.legend(loc="best")


def plot_training_set(estimator, axis, X, y, ylim=None,
                      cv=None, group=None, test_size=0.25):
    if not cv:
        cv = KFold(n_splits=5)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)
    estimator.fit(X_train, y_train)



cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)


def method_tester(x_data, y_names, modeler, test_size=0.25):
    for i, y_name in enumerate(y_names):
        # calculate full train set
        y_data = data[y_name]
        modeler.fit(x_data, y_data)
        y_predict = modeler.predict(x_data)
        r2_train = r2_score(y_data, y_predict)
        mae_train = mean_absolute_error(y_data, y_predict)

        # # calculate a test set
        # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
        #                                                     test_size=test_size)
        # modeler.fit(x_train, y_train)
        # y_test_predict = modeler.predict(x_test)
        # r2_test = r2_score(y_test, y_test_predict)
        # mae_test = mean_absolute_error(y_test, y_test_predict)
        # print(r2_train, mae_train, r2_test, mae_test)
        # return r2_train, mae_train, r2_test, mae_test
        local_scores = cross_validate(modeler, x_data, y=y_data,
                                      cv=cv, scoring=['r2', 'neg_mean_absolute_error'])
        r2_test = local_scores['test_r2'].mean()
        mae_test = local_scores['test_neg_mean_absolute_error'].mean()
        return r2_train, mae_train, r2_test, mae_test


def fit_n_plot_estimator(x_data, y_names, modeler, title):

    figure, axes, = plt.subplots(3, 2, figsize=(9, 8), constrained_layout=True)
    figure.suptitle(title)

    # print(x_data)

    for i, y_name in enumerate(y_names):

        y_data = data[y_name]

        fitter = modeler.fit(x_data, y_data)
        y_predict = modeler.predict(x_data)

        # y vs y_predict
        axes[i][0].scatter(y_predict, y_data)
        axes[i][0].set_xlabel("Predicted {0}".format(y_name))
        axes[i][0].set_ylabel("Measured {0}".format(y_name))

        r2 = r2_score(y_data, y_predict)
        mae = mean_absolute_error(y_data, y_predict)
        print(mae)

        add_legend = False
        if i == 0:
            add_legend = True
        if mae <= 100:

            plot_learning_curve(modeler, axes[i][1],
                                x_data, y_data, cv=cv,
                                final_y_predict=mae, add_legend=add_legend)

            axes[i][0].annotate(u"R\u00B2 ={:.3f}".format(r2), xy=(0.1, 0.85),
                                xycoords='axes fraction', color='#101028')
            axes[i][0].annotate(u"MAE ={:.3f}".format(mae), xy=(0.1, 0.75),
                            xycoords='axes fraction', color='#101028')


models_to_test = [PLS(n_components=1), PLS(n_components=2),
                  PLS(n_components=3), PLS(n_components=4),
                  Lasso(alpha=1), Lasso(alpha=0.1),
                  Lasso(alpha=0.01), Lasso(alpha=0.001),
                  LassoLars(alpha=1), LassoLars(alpha=0.1),
                  LassoLars(alpha=0.01), LassoLars(alpha=0.001),
                  Ridge(alpha=0.01, max_iter=5000),
                  Ridge(alpha=0.001, max_iter=5000),
                  Ridge(alpha=0.0001, max_iter=5000),
                  Ridge(alpha=0.00001, max_iter=5000),
                  Lars(), GaussianProcessRegressor(),
                  GradientBoostingRegressor(), SVR(), LinearSVR(), NuSVR(),
                  LogisticRegression(), LinearRegression(), SGDRegressor(),
                  ElasticNet(), ARDRegression(), BayesianRidge(),
                  HuberRegressor(), RANSACRegressor(), TheilSenRegressor(),
                  PassiveAggressiveRegressor(),
                  AdaBoostRegressor(), BaggingRegressor(), GradientBoostingRegressor(),
                  RandomForestRegressor(n_estimators=10, max_depth=2),
                  ExtraTreesRegressor(n_estimators=10, max_depth=2),
                  KernelRidge()]

model_names = ["PLS 1-component", "PLS 2-component",
               "PLS 3-component", "PLS 4-component",
               "Lasso alpha 1", "Lasso alpha 0.1",
               "Lasso alpha 0.01", "Lasso alpha 0.001",
               "LassoLars alpha 1", "LassoLars alpha 0.1",
               "LassoLars alpha 0.01", "LassoLars alpha 0.001",
               "Ridge alpha 0.01", "Ridge alpha 0.001",
               "Ridge alpha 0.0001", "Ridge alpha 0.00001",
               "Lars", "Guassian Regression",
               "Gradient Boosting",
               "SVR", "LinearSVR", "NuSVR",
               "LogisticRegression", "LinearRegression", "SGDRegressor",
               "ElasticNet", "ARDRegression", "BayesianRidge",
               "HuberRegressor", "RANSACRegressor", "TheilSenRegressor",
               "PassiveAggressiveRegressor",
               "AdaBoostRegressor", "BaggingRegressor", "GradientBoostingRegressor",
               "RandomForestRegressor", "ExtraTreesRegressor",
               "Kernel Ridge"
               ]


y_transformations = [None, (np.exp, np.log), (np.log, np.exp), (np.log1p, np.expm1),
                     (np.reciprocal, np.reciprocal),
                     QuantileTransformer(n_quantiles=10),
                     QuantileTransformer(n_quantiles=10, output_distribution='normal'),
                     PowerTransformer()]

# y_transformations = [None, (np.reciprocal, np.reciprocal)]
x_transformations = [None, np.exp, np.log, np.reciprocal, np.log1p, np.expm1]
# x_transformations = [None, np.exp]
# x_transformations = [np.log, np.reciprocal]

transormation_names = ["None", "Exponential", "Logarithm",
                       "Reciprocol",  "Logarithm plus 1", "Exponential minus one",
                       "Quantile Transform", "Normal Quantile Transform",
                       "Power Transform"]

# results = pd.DataFrame(columns=["Model", "y transform", "x transform", "r2 train", "mae train", "r2 test", "mae test"])
results = []

# x_data = np.diff(x_data)

# x_data = processing.snv(x_data)
x_data = StandardScaler().fit_transform(x_data)

for i, test_model in enumerate(models_to_test):
    for j, y_transformation in enumerate(y_transformations):
        for k, x_transformation in enumerate(x_transformations):

            title = "{0}\ny transformer: {1}" \
                    "\nx transformer: {2}".format(model_names[i],
                                                  transormation_names[j],
                                                  transormation_names[k])

            if x_transformation:
                x_data_new = x_transformation(x_data)
            else:
                x_data_new = x_data.copy()

            if type(y_transformation) == tuple:
                _model = TransformedTargetRegressor(regressor=clone(test_model),
                                                    func=y_transformation[0],
                                                    inverse_func=y_transformation[1])
            elif y_transformation:
                _model = TransformedTargetRegressor(regressor=clone(test_model),
                                                    transformer=y_transformation)
            else:
                _model = clone(test_model)


            print(title)

            try:
                # fit_n_plot_estimator(x_data_new, y_columns,
                #                      _model, title)
                r2_train, mae_train, r2_test, mae_test = method_tester(x_data_new, y_columns, _model)
                results.append([model_names[i], transormation_names[j],
                                transormation_names[k], r2_train,
                                mae_train, r2_test, mae_test])
                # fit_n_plot_estimator(x_data_new, y_columns,
                #                      _model, title)
                # method_tester(x_data_new, y_columns, _model)
            except Exception as e:
                print("FAIL===============>>>>>>>>>")
                print(e)

print(results)
results_pd = pd.DataFrame(results, columns=["Model", "y transform", "x transform", "r2 train", "mae train", "r2 test", "mae test"])
print(results_pd)

results_pd.to_csv("results_Scalar_processing.csv")

plt.show()