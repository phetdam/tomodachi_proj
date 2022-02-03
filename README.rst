.. tomodachi_proj README.rst

   last updated: 2022-02-02
   file created: 2019-10-22

BAC Insight Team Fall 2019: Spotify Song Recommendations
========================================================

.. image:: ./banner.png
   :alt: ./banner.png

Originally BAC Insight Team's project on crime in Chicago, but due to some
issues with the data set, we have changed our project to using panel data on
songs extracted from the Spotify API to predict a song's danceability. Our goal
is to see which features are relevant for predicting danceability, and to
create models that could be used to recommend automatically construct a
playlist of Spotify songs to dance to at the club or other venue.

You may find the original data here__ on Kaggle__, or use the extracted
``.csv`` files in the ``./data`` directory.

.. __: https://www.kaggle.com/snapcrack/the-billboard-200-acoustic-data

.. __: https://www.kaggle.com/

   **IMPORTANT:** The pickled models in the ``./models`` directory were
   created under 32-bit Python 3.7.4, so the tree-based models **cannot** be
   unpickled under 64-bit Python. This is an issue dating from 2014 that has
   never been resolved and only affects tree-based models in `sklearn`.
   Details can be found on StackOverflow here__.
   
.. __: https://stackoverflow.com/questions/21033038/scikits-learn-
   randomforrest-trained-on-64bit-python-wont-open-on-32bit-python

Data files
----------

All the data used in this project are saved in .csv files in the ``./data``
directory. Descriptions below.

``./data/X_full.csv``
   The original full preprocessed data set, containing the entire 26 columns of
   features. Split this data into your own training and test data and for
   performing your own feature engineering, cross validation, resampling, etc.

``./data/X_train.csv``
   The original preprocessed training data split from the full 26-column
   feature matrix ``X_full.csv``.

``./data/y_train.csv``
   Response vector for the train data from the full feature matrix.

``./data/X_test.csv``
   Original preprocessed test data split from the full 26-column feature matrix.

``./data/y_test.csv``
   Rresponse vector for the test data from the full feature matrix.

``./data/Xm_train.csv``
   Preprocessed, resampled, reduced feature training data with only the 9
   continuous features.

``./data/ym_train.csv``
   Response vector for the reduced feature, resampled test data.

``./data/Xm_test.csv``
   Preprocessed reduced feature test data that is the test partition
   complementing ``./data/Xm_train.csv``.  9 columns.

``./data/ym_train.csv``
   Response vector for the reduced feature test data ``./data/Xm_test.csv``.

Models
------

The pickles of the ``GridSearchCV`` objects containing different base models
can be found in ``./models``. Descriptions contain instructions for accessing
certain fields and properties that may be useful or desirable.

``dtc_rfecv.pickle``
   Pickle of the ``RFECV`` object containing a ``DecisionTreeClassifier`` as
   the ``estimator_`` used to greedily select features with cross validation on
   the training set data in ``X_train.csv``. A useful attribute besides the
   model accessed from ``estimator_`` is the ``support_`` attribute, a boolean
   mask of the selected features from the columns of ``X_train.csv``.

``dtc_gscv.pickle``
   Pickle of the ``GridSearchCV`` object containing a ``DecisionTreeClassifier``
   as the estimator used, fit on the resampled reduced feature data from
   ``Xm_train.csv``, **not** the full feature data from ``X_train.csv``. Useful
   attributes are ``cv_results_`` for the full set of cross validation metrics.
   Can retrieve the best fitting estimator across all CV folds by through
   ``best_estimator_``, its cross validation score through ``best_score_``, and
   the mean cross validation scores through ``cv_results_["mean_test_score"]``,
   which I like to call ``.mean()`` on.

``ada_stump_gscv.pickle``
   Pickle of the ``GridSearchCV`` object containing an ``AdaBoostClassifier``
   as the estimator, with ``DecisionTreeClassifier(max_depth = 1)`` as the base
   estimator (a tree stump), fit on the resampled reduced feature data from
   ``Xm_train.csv``. Contains 80 stumps. Refer to above for useful attributes
   of the ``GridSearchCV`` object. Can retrieve the base estimators through the
   ``estimators_`` property of the ``AdaBoostClassifier``.

``ada_tuned_gscv.pickle``
   Pickle of the ``GridSearchCV`` object containing an ``AdaBoostClassifier``
   as the estimator, with a full ``DecisionTreeClassifier`` as the base
   estimator, fit on the resampled reduced feature data from ``Xm_train.csv``.
   Contains 50 trees. Refer to above for useful attributes of the
   ``GridSearchCV`` object and for how to retrieve the base estimators.

``ada_ext_gscv.pickle.zip``
   Zipped pickle of the ``GridSearchCV`` object containing an
   ``AdaBoostClassifier`` as the estimator, with a full
   ``DecisionTreeClassifier`` as the base estimator, fit on the resampled
   reduced feature data from ``Xm_train.csv``. Contains 150 trees; zipped as
   the file size is over the 100 MB Git limit. Refer to above for useful
   attributes of the ``GridSearchCV`` object and how to for retrieve the base
   estimators.

``ada_fstump_gscv.pickle``
   Pickle of the ``GridSearchCV`` object containing an ``AdaBoostClassifier``
   as the estimator, with a decision tree stump as the base estimator, fit on
   the full feature data from ``X_train.csv``. Contains 80 stumps. Refer to
   above for useful attributes of the ``GridSearchCV`` object and for how to
   retrieve the base estimators.

``ada_ftuned_gscv.pickle``
   Pickle of the ``GridSearchCV`` object containing an ``AdaBoostClassifier``
   as the estimator, with a full ``DecisionTreeClassifier``as the base
   estimator, fit on the full feature data from ``X_train.csv``. Contains 50
   trees. Refer to above for useful attributes of the ``GridSearchCV`` object
   and for how to retrieve the base estimators.

``flr_gscv.pickle``
   Pickle of the ``GridSearchCV`` object containing a ``LogisticRegression``
   model as the estimator, fit on the full feature data from ``X_train.csv``.
   Refer to above for useful attributes of the ``GridSearchCV`` object. Can
   retrieve model coefficients from the estimator through the ``coef_``
   attribute, which takes varying shape: the shape is ``(1, n_features)`` if
   the problem is a two-class problem, while the shape is
   ``(n_classes, n_features)`` if the problem is a multi-class problem.

``lr_gscv.pickle``
   Pickle of the ``GridSearchCV`` object containing a ``LogisticRegression``
   model as the estimator, fit on the resampled, reduced feature data from
   ``Xm_train.csv``. Refer to above for useful attributes of the
   ``GridSearchCV`` object. Can retrieve model coefficients from the estimator
   through the ``coef_`` attribute; see above for shape.

``fsgd_gscv.pickle``
   Pickle of the ``GridSearchCV`` object containing a ``SGDClassifier`` model
   as the estimator, fit on the full feature data from ``X_train.csv``. Refer
   to above for useful attributes of the ``GridSearchCV`` object. Can retrieve
   model coefficients from the estimator through the ``coef_`` attribute; see
   above for shape.

``sgd_gscv.pickle``
   Pickle of the ``GridSearchCV`` object containing a ``SGDClassifier`` model
   as the estimator, fit on the resampled, reduced feature data from
   ``Xm_train.csv``. Refer to above for useful attributes of the
   ``GridSearchCV`` object. Can retrieve model coefficients from the estimator
   through the ``coef_`` attribute; see above for shape.

Figures
-------

The figures in the ``./figures`` directory mostly display model statistics for
a single model, namely the confusion matrix, ROC curve, and feature importances
(for trees) or coefficients (for linear models). The file name of each
model-related figure has the form ``[model_name]_stats.png``, and as implied,
each corresponds to a pickle ``[model_name].pickle``.