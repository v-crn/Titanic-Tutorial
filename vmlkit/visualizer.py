from vmlkit import validator

import numpy as np
from scipy import interp
import pandas_profiling as pdp
from IPython.display import HTML
from sklearn.metrics import auc, plot_roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


def display_html(path):
    HTML(filename=path)


def create_profile(df, savepath):
    report = pdp.ProfileReport(df)
    report.to_file(savepath)


def plot_learning_curves(clf, X, y, cv=None, path_img=None):
    count = 1

    if cv is None:
        cv = StratifiedKFold(n_splits=5,
                             random_state=47, shuffle=True)

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        plot_learning_curves(X_train, y_train, X_test, y_test, clf)
        plt.subplots_adjust(left=0.2, right=0.9, bottom=0.15, top=0.5)
        path_imgs = path_img.replace('.', '_%d.' % count)
        plt.savefig(path_imgs)
        plt.show()
        count += 1


def plot_roc_curve_with_cv(model, X, y, cv=None, n_splits=5,
                           n_repeats=10, test_size_ratio=0.2,
                           savepath=None, random_state=37):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if cv is None:
        cv = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True)

    if type(cv) is str:
        cv = validator.get_cv(cv, n_splits=n_splits, test_size_ratio=0.2,
                              n_repeats=n_repeats,
                              random_state=random_state, shuffle=True)

    fig, ax = plt.subplots()
    for i, (train, val) in enumerate(cv.split(X, y)):
        model.fit(X.loc[train], y.loc[train])
        viz = plot_roc_curve(model, X.loc[val], y.loc[val],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic")
    ax.legend(loc="lower right")

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

    return ax
