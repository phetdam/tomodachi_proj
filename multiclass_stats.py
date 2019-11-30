__doc__ = """
contains a method for producing multiple statistics related to multiclass
classifiers implemented through sklearn. the function returns a figure that
contains a confusion matrix, ROC curve, and optionally coefficients/feature
importances based on the type of classifier passed in. the classifier must
be fitted already, as if not then the coef_ or feature_importances_ attributes
will not exist (feature_importances_ actually a function attached to @property).
note that if there are multiple classes, for models that have coefficients,
there will be a coefficients graph for each of the classes.

IMPORTANT: matplotlib<=3.1.0 recommended as 3.1.1 makes the confusion matrix
annotations all messed up. i wrote this code with matplotlib==3.1.0.

example:

suppose we are given a fitted classifier cf, test data X_test, y_test, and want
to indiciate that cf is the best model out of a few other models, with the name
my_best_model_20. we also want to show coefficient graphs, and save the image to
./cf_stats.png. we also want to keep the returned figure, confusion matrix, and
dictionary containing the misclassification rates, precision, and AUC.

therefore, we would use the following function call:

from multiclass_stats import multiclass_stats
fig, cmat, stats_dict = multiclass_stats(cf, X_test, y_test, best_model = True,
                                         model_name = "my_best_model_20",
                                         feature_ws = True,
                                         out_file = "./cf_stats.png")
"""
# Changelog:
#
# 11-30-2019
#
# modified initial argument type checking since there is erroneous fall-through.
#
# 11-27-2019
#
# added short main to warn if user tries to run module as script.
#
# 11-26-2019
#
# added proper support for the coef_ attribute, for both the two-class and the
# one-vs-all multi-class classification coefficient schemes. also added option
# to change the color palette being used for the coefficient graphs/feature
# importance graph for the aesthetic. wish i knew of the ravel() method earlier;
# it simplifies the nested axes plotting problem with multiple rows by simply
# flattening into a 1d array. updated docstring to reflect new changes, and
# also corrected some minor docstring typos; added example to module docstring.
#
# note: i did not actually test the multi-class case; i instead modified the
# single class case to have extra plots and then manually plotted a few more
# coefficient graphs, so performance for a real multi-class classifier that
# returns coef_ with shape (n_classes, n_features) is not guaranteed. but my
# tests imply that it should work fine.
#
# 11-25-2019
#
# started work on adding proper support for the coef_ attribute for both the
# two-class and the multi-class classification cases. updated docstring.
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
from numpy import ravel

# main function
def multiclass_stats(mce, X_test, y_test, model_name = "auto",
                     feature_ws = False, figsize = "auto", verbose = False,
                     best_model = False, cmap = "Blues", cbar = False,
                     roc_color = "coral", palette = None, no_return = False,
                     out_file = None):
    """
    produces multiple statistics for multiclass classifier implemented in
    sklearn. several parameters are available to control the output. returns a
    figure object with all the plots, a confusion matrix, and the dictionary:

    {"mc_rates": [...], "accuracy": x, "precision": y, "auc": z}

    some parameters control the titles of the plots produced. the format of the
    confusion matrix and ROC curve subplot titles can be changed by the
    model_name and best_model parameters, and is given by

    "confusion matrix for [best if best_model is True else blank] [model_name if
    model_name not 'auto' else object name]"

    "ROC curve for [best if best_model is True else blank] [model_name if
    model_name not 'auto' else object name]"

    the feature_ws parameters controls whether or not coefficients*/feature
    importances of a model are included as well, depending on the model. the
    plot for the coefficients/feature importances has the title

    "[coefficients/feature importances], [model_name if model_name not 'auto'
    else object name]"

    *the above is for the two-class case. in the multi-class case, each graph
    of coefficients for each class k will have the title given by

    "coefficients k, [model_name if model_name not 'auto' else object name]"

    feature importances are global in multi-class classification.

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
    figsize      optional, default "auto". will allocate only 4 inches in width
                 and height if 2 subplots (feature_ws == False). for 3 subplots
                 4.2666 inches width and 4 inches for each plot, while for more
                 than 3 subplots, 4.65 inches width and 4 inches for each plot;
                 both these cases are when feature_ws == True. for more than
                 3 total plots, there will be 3 plots per row, with multiple
                 rows as necessary. we need to vary the widths because the
                 tight_layout() makes different adjustments in each case.
    verbose      optional, default False. if True, prints some error messages.
    best_model   optional, default False. whether or not to include "best"
                 before the name of the model in the subplot titles for the
                 confusion matrix and the ROC curve. doesn't apply to the plot
                 of coefficients/feature importances.
    cmap         optional, default "Blues". color map for heatmap.
    cbar         optional, default False. if True, then a color bar will be
                 drawn for the heatmap that the confusion matrix is displayed.
    roc_color    optional, default "coral". color for the ROC curve lineplot.
    palette      optional, default None for current palette. only matters when
                 feature_ws is True and if the classifier has either the coef_
                 or feature_importances_ attribute. determines color palette
                 to use for the feature importances/coefficients graphs,
                 corresponding exactly to the palette keyword argument in the
                 sns.color_palette() function.
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
        raise SystemExit(1)
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
                print("{0}: error: object\n{1} does not have attribute feature"
                      "_importances_".format(fname_, mce), file = sys.stderr)
        # if both are None, print error and exit
        if (coefs is None) and (feature_imps is None):
            print("{0}: error: object\n{1} does not have attributes coef_ and "
                  "feature_importances_".format(fname_, mce), file = sys.stderr)
            quit(1)
    # if the feature_ws block evaluates correctly, make the figure
    # compute true and false positive rates
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    # number of subplots in the figure; 2 default unless feature_ws is True
    nplots = 2
    # shape of coefs; if coefs is not None, shape is (1, n_features) if there
    # are only two classes. for multiple classes, (n_classes, n_features).
    coefs_shape = None
    # if feature_ws is True, we have to determine three cases: 1. if there is no
    # coefs and only feature_imps is not None, then 3 plots, 1 row. 2. there is
    # coefs so no feature_imps, but only two classes, which means coefs will
    # have shape (1, n_features), so again 3 plots. 3. there is coefs so no
    # feature_imps, but multiple classes, where coefs will have the shape
    # (n_classes, n_features), so n_classes + 2 plots.
    if feature_ws is True:
        # first check that we have coefficients
        if coefs is not None:
            # get the shape
            coefs_shape = (len(coefs), len(coefs[0]))
            # if coefs_shape[0] == 1, then set nplots = 3
            if coefs_shape[0] == 1: nplots = 3
            # else > 1 so set nplots = 2 + coefs_shape[0]
            else: nplots = 2 + coefs_shape[0]
        # else if coefs is None, then feature_imps is not None (we already did
        # error checking in the the previous statement already), so nplots = 3
        elif feature_imps is not None: nplots = 3
    # if figsize is "auto", do 4 inch width + height if nplots == 2, 4.2666
    # inches width and 4 inches height/subplot is nplots == 3, and 4.65 inches
    # width and 4 inches height/subplot if nplots > 3 in order to maintain
    # differing square shapes based on tight_layout() adjustments.
    if figsize == "auto":
        if nplots == 2: figsize = (4 * nplots, 4)
        if nplots == 3: figsize = (4.2666 * nplots, 4)
        # add an extra row if nplots % 3 > 0; i.e. nplots / 3 > nplots // 3 so
        # there is an extra row needed for the remaining 1 or 2 plots
        else: figsize = (4.65 * 3,
                         4 * (nplots // 3 + (1 if nplots % 3 > 0 else 0)))
    # make the plot using the given parameters; use same expression as in
    # figsize to determine the appropriate number of rows and columns
    fig, axs = plt.subplots(nrows = nplots // 3 + (1 if nplots % 3 > 0 else 0),
                            ncols = min(nplots, 3), figsize = figsize)
    # flatten the axes
    axs = ravel(axs)
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
    # if coefs is not None, then plot coefficients of the model (harder)
    if coefs is not None:
        # will use same color palette, specified by palette argument, with
        # coefs_shape[1] colors, for each of the coefficients graphs
        colors = sns.color_palette(palette = palette, n_colors = coefs_shape[1])
        # for each remaining subplot, index 2 to nplots - 1, make barplots using
        # palette colors for each of the plots, setting title and plotting.
        # note special case if coefs_shape[0] == 1, where only one set of coefs.
        if coefs_shape[0] == 1:
            axs[2].set_title("coefficients, {0}".format(model_name))
            sns.barplot(data = DataFrame([coefs[0]], columns = X_test.columns),
                        palette = colors, ax = axs[2], orient = "h")
        # else there are multiple sets of coefficients, so in the title for each
        # of the barplots, index by the class number
        else:
            for i in range(2, nplots):
                axs[i].set_title("coefficients {0}, {1}".format(i - 2,
                                                                model_name))
                sns.barplot(data = DataFrame([coefs[i - 2]],
                                             columns = X_test.columns),
                            palette = colors, ax = axs[i], orient = "h")
    # else if feature_imps is not None, plot the feature importances
    elif feature_imps is not None:
        axs[2].set_title("feature importances, {0}".format(model_name))
        sns.barplot(data = DataFrame([feature_imps], columns = X_test.columns),
                    palette = colors, ax = axs[2], orient = "h")
    # if both are None, do nothing; feature_ws is probably False
    # adjust figure for tightness
    fig.tight_layout()
    # if out_file is not None, save to out_file
    if out_file is not None: fig.savefig(out_file)
    # if no_return is True, return None
    if no_return == True: return None
    # else return figure, confusion matrix cmat, and statistics in stats_dict
    return fig, cmat, stats_dict

# main
if __name__ == "__main__":
    print("{0}: do not run module as script. refer to docstring for usage."
          "".format(multiclass_stats.__name__), file = stderr)
