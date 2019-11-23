__doc__ = """
contains a method for producing multiple statistics related to multiclass
classifiers implemented through sklearn. the function returns a figure that
contains a confusion matrix, ROC curve, and optionally coefficients/feature
importances based on the type of classifier passed in. the classifier must
be fitted already, as if not then the coef_ or feature_importances_ attributes
will not exist (feature_importances_ actually a function attached to @property).

IMPORTANT: matplotlib<=3.1.0 recommended as 3.1.1 makes the confusion matrix
annotations all messed up. i wrote this code with matplotlib==3.1.0.
"""
# Changelog:
#
# 11-22-2019
#
# initial creation. for some reason the ROC curve produced was not exactly
# matching the ROC curve produced by manual line-by-line plotting, but then i
# realized it was an error in my manual code. hence why having this wrapper
# makes repeated plotting a lot more convenient. i also chose the default sizes
# for the plots to be the maximum width to display well in git without having
# to scroll to the right.

# needed to test whether the object is a classifier or not
from sklearn.base import ClassifierMixin
# metrics
from sklearn.metrics import (confusion_matrix, precision_score, roc_curve,
                             roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from sys import stderr
from pandas import DataFrame

# main function
def multiclass_stats(mce, X_test, y_test, model_name = "auto",
                     feature_ws = False, figsize = "auto", verbose = False,
                     best_model = False, cmap = "Blues", cbar = False,
                     roc_color = "coral", no_return = False, out_file = None):
    """
    produces multiple statistics for multiclass classifier implemented in
    sklearn. several parameters are available to control the output. returns a
    figure object with all the plots, a confusion matrix, and the dictionary:

    {"mc_rates": [...], "accuracy": x, "precision": y, "auc": z}

    some parameters control the titles of the plots produced. the format of the
    confusion matrix and ROC curve subplot titles can be changed by the
    model_name and best_model parameters, and is given by

    "confusion matrix for [best if best_model is True else blank] [model_name if
    model_name is not 'auto' else object name"

    "ROC curve for [best if best_model is True else blank] [model_name if
    model_name is not 'auto' else object name"

    the feature_ws parameters controls whether or not coefficients/feature
    importances of a model are included as well, depending on the model. the
    plot for the coefficients/feature importances has the title

    "[coefficients/feature importances], [model_name if model_name is not 'auto'
    else object name"

    parameters:

    mce         multiclass classifier; must inherit sklearn.base.ClassifierMixin
    X_test      feature matrix for test data
    y_test      response vector for test data

        note: mce should already be fit on X_train, y_train data

    model_name   optional, default "auto" gives the class name of mce. will
                 change titles of the confusion matrix and of the ROC curve.
    feature_ws   optional, default False. whether or not to include feature
                 importances or model coefficients, depending on the model. if
                 either feature_importances_ or coef_ cannot be found, the
                 function will print an error and terminate.
    figsize      optional, default "auto". will allocate 4.2666 inches in width
                 and 4 inches in height if 3 subplots (feature_ws is True) or
                 only 4 inches in width and height if 2 subplots.
    verbose      optional, default False. if True, prints some error messages.
    best_model   optional, default False. whether or not to include "best"
                 before the name of the model in the subplot titles for the
                 confusion matrix and the ROC curve. doesn't apply to the plot
                 of coefficients/feature importances.
    cmap         optional, default "Blues". color map for heatmap.
    cbar         optional, default False. if True, then a color bar will be
                 drawn for the heatmap that the confusion matrix is displayed.
    roc_color    optional, default "coral". color for the ROC curve lineplot.
    no_return    optional, default False. if True, then returns None instead of
                 the figure, confusion matrix, and dictionary tuple. useful when
                 displaying inline plots using matplotlib in jupyter notebooks.
    out_file     optional, default None. if a string, the method will attempt to
                 save to the figure into that file.
    """
    # save the name of the function for convenience
    fname_ = multiclass_stats.__name__
    # check that the estiamator mce is a classifier. if not, print error
    if isinstance(mce, ClassifierMixin) == False:
        print("{0}: must pass in classifier inheriting from sklearn.base."
              "ClassifierMixin".format(fname_), file = sys.stderr)
    # check the length of X_test and y_test are the same
    if len(X_test) != len(y_test):
        print("{0}: X_test and y_test must have same umber of observations"
              "".format(fname_), file = sys.stderr)
    # dictionary of statistics
    stats_dict = {}
    # compute confusion matrix
    cmat = confusion_matrix(y_test, mce.predict(X_test))
    # number of classes
    nclasses = len(cmat)
    # compute misclassification rates
    mc_rates = [None for _ in range(nclasses)]
    for i in range(nclasses):
        # misclassification rate is 1 - correct / sum of all
        mc_rates[i] = 1 - cmat[i][i] / sum(cmat[i])
    # add entry in stats_dict
    stats_dict["mc_rates"] = mc_rates
    # predict values from X_test and get accuracy, precision, auc score
    y_test_pred = mce.predict(X_test)
    stats_dict["accuracy"] = mce.score(X_test, y_test)
    stats_dict["precision"] = precision_score(y_test, y_test_pred)
    stats_dict["auc"] = roc_auc_score(y_test, y_test_pred)
    # coefs, feature importances
    coefs, feature_imps = None, None
    # if feature_ws is True, check if either coef_ or feature_importances_ does
    # exist within mce. if an exception is raised, catch it, print error, exit.
    if feature_ws == True:
        # first look for coefficients
        try:
            coefs = getattr(mce, "coef_")
        # else do nothing, as we print errors later
        except: pass
        # then look for feature importances if we have it
        try:
            feature_imps = getattr(mce, "feature_importances_")
        # else do nothing, as we print errors later
        except: pass
        # if errors are verbose, print error for any None values
        if verbose == True:
            if coefs is None:
                print("{0}: error: object\n{1} does not have attribute coef_"
                      "".format(fname_, mce), file = sys.stderr)
            if feature_imps is None:
                print("{0}: error: object\n{1} does not have attribute "
                      "feature_importances_".format(fname_, mce), file = sys.stderr)
        # if both are None, print error and exit
        if (coefs is None) and (feature_imps is None):
            print("{0}: error: object\n{1} does not have attributes coef_ and "
                  "feature_importances_".format(fname_, mce), file = sys.stderr)
            quit(1)
    # if the feature_ws block evaluates correctly, make the figure
    # compute true and false positive rates
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    # number of subplots in the figure; 2 default unless feature_ws is True
    nplots = 3 if feature_ws == True else 2
    # if figsize is "auto", 4.2666 inches width and 4 inches height/subplot if
    # nplots == 3, else do 4 inches width and 4 inches height/subplot if we
    # have that nplots == 2
    if figsize == "auto":
        if nplots == 2: figsize = (4 * nplots, 4)
        elif nplots == 3: figsize = (4.2666 * nplots, 4)
    # make the plot using the given parameters
    fig, axs = plt.subplots(nrows = 1, ncols = nplots, figsize = figsize)
    # set best option
    best_ = ""
    if best_model is True: best_ = "best "
    # set model name; if auto, set to object name
    if model_name == "auto": model_name = str(mce).split("(")[0]
    ### create confusion matrix ###
    # set title of confusion matrix
    axs[0].set_title("confusion matrix for {0}{1}".format(best_, model_name))
    # heatmap, with annotations, decimal format for values
    sns.heatmap(cmat, annot = True, cmap = cmap, cbar = cbar, ax = axs[0],
                fmt = "d")
    ### create ROC curve ###
    # set title and axis labels
    axs[1].set_title("ROC curve for {0}{1}".format(best_, model_name))
    axs[1].set_xlabel("false positive rate")
    axs[1].set_ylabel("true positive rate")
    # create line plot
    sns.lineplot(fpr, tpr, color = roc_color, ax = axs[1])
    # if feature_ws is True, then also display feature importances/coefficients
    # based on whatever is first found to be true
    #coefficients/feature importances
    # if coefs is not None, then plot coefficients of the model
    if coefs is not None:
        axs[2].set_title("coefficients, {0}".format(model_name))
        sns.barplot(data = DataFrame([coefs], columns = X_test.columns),
                    ax = axs[2], orient = "h")
    elif feature_imps is not None:
        axs[2].set_title("feature importances, {0}".format(model_name))
        sns.barplot(data = DataFrame([feature_imps], columns = X_test.columns),
                    ax = axs[2], orient = "h")
    # if both are None, do nothing; feature_ws is probably False
    # adjust figure for tightness
    fig.tight_layout()
    # if out_file is not None, save to out_file
    if out_file is not None: fig.savefig(out_file)
    # if no_return is True, return None
    if no_return == True: return None
    # else return figure, confusion matrix cmat, and statistics in stats_dict
    return fig, cmat, stats_dict
