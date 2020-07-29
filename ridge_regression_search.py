# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, SGDRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
from sklearn.svm import SVR
# local files
import data_getter
import processing

plt.style.use('seaborn')

# x_data, _, data = data_getter.get_data('as7265x mango')
x_data, _, data = data_getter.get_data('as7262 betal')
print(data)
# data = data.loc[(data['LED'] == 'UV (405 nm) LED')]
# data = data.loc[(data['position'] == 'pos 2')]
# data = data.loc[(data['integration time'] == 150)]
data = data.groupby('Leaf number', as_index=True).mean()
print('++++++')
print(data)
print(data.columns)
# print(data['LED'].unique())
# data = data.loc[data['LED'] == ' White LED']
# print(data.shape)
# chlorophyll data
data = data.groupby('Leaf number', as_index=True).mean()

print(data.columns)
print('======')
print(data)
spectrum_columns = []
for column in data.columns:
    if 'nm' in column:
        spectrum_columns.append(column)
x_data = data[spectrum_columns]
alphas = np.logspace(-15, -5, 20)
# alphas = np.array([0.01, 0.1, 1, 10, 100])
print(alphas)
# x_data = processing.snv(x_data)
# x_data = np.exp(-x_data).fillna(0).replace([np.inf, -np.inf], 0)
# x_data = np.log(x_data)
x_data = x_data.fillna(0).replace([np.inf, -np.inf], 0)
print('=====')

tuned_parameters = [{'alpha': alphas}]
tuned_parameters = [{'gamma': alphas}]
# tuned_parameters = [{'C': alphas}]
n_folds = 5
cv = ShuffleSplit(n_splits=100, test_size=0.35, random_state=42)

qt = QuantileTransformer(n_quantiles=10, random_state=0)
pt = PowerTransformer()
# x_data = qt.fit_transform(x_data)

# y_data = 1 / data['Total Chlorophyll (ug/ml)']

y_data = np.log(data['Chlorophyll b (ug/ml)'])

ridge = Ridge(random_state=6, max_iter=10000)
svr = SVR(C=10, kernel='rbf')
lasso = Lasso(max_iter=5000)
estimators1 = [('reduce_dim', PCA(n_components=6)), ('clf', Lasso(max_iter=5000))]
pipe1 = Pipeline(estimators1)
tuned_parameters1 = [{'alpha': alphas, 'n_components': [1, 2, 3, 4, 5, 6, 7, 8]}]


clf = GridSearchCV(svr, tuned_parameters,
                   scoring='r2', return_train_score=True,
                   cv=cv, refit=False)
clf.fit(x_data, y_data)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
train_score = clf.cv_results_['mean_train_score']
train_scores_std = clf.cv_results_['std_train_score']


plt.figure()  # .set_size_inches(8, 6)
plt.subplot(121)
plt.semilogx(alphas, scores,
             label="Testing set")
plt.semilogx(alphas, train_score,
             c='maroon', label="Training set")

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

plt.semilogx(alphas, train_score + train_scores_std, 'r--')
plt.semilogx(alphas, train_score - train_scores_std, 'r--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.fill_between(alphas, train_score + train_scores_std,
                 train_score - train_scores_std,
                 alpha=0.2, color='tomato')

plt.ylabel(U'CV R\u00B2 score +/- std error')
# plt.ylim([0.6, 0.9])
plt.xlabel('alpha')
plt.title("Roseapple AS7263 LASSO fit to Total Chlorophyll", size=16)
plt.axhline(np.max(scores), linestyle='--', color='.5')

plt.annotate(u"Best R\u00B2 ={:.3f}".format(np.max(scores)),
             xy=(0.25, 0.25),
            xycoords='axes fraction', color='#101028', size=14)
plt.legend()
plt.xlim([alphas[0], alphas[-1]])

plt.subplot(122)



plt.show()
