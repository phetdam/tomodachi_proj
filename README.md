# BAC Insight Team Fall 2019: Spotify Song Recommendations

![./banner.png](./banner.png)

_last updated: 11-21-2019_  
_file created: 10-22-2019_

Was originally the working project folder for BAC Insight Team's project on crime in Chicago, but due to some issues with the data set, we have changed our project to using panel data on songs extracted from the Spotify API to predict a song's danceability. Our goal is to see which features are relevant for predicting danceability, and to create models that could be used to recommend automatically construct a playlist of Spotify songs to dance to at the club or other venue.

You may find the original data [here](https://www.kaggle.com/snapcrack/the-billboard-200-acoustic-data) on [Kaggle](https://www.kaggle.com/), or use the extracted .csv files in the `./data` directory. Stay tuned for more.

__Remark.__ This repository is a work of progress, so expect frequent changes.  

Contributors: Derek Huang, TBA

## Data Files

All the data used in this project are saved in .csv files in the `./data` directory. Descriptions below.

* **./data/X_full.csv**

    The original full preprocessed data set, containing the entire 26 columns of features. Split this data into your own training and test data and for performing your own feature engineering, cross validation, resampling, etc. on.

* **./data/X_train.csv**

    The original preprocessed training data split from the full 26-column feature matrix `X_full.csv`.

* **./data/y_train.csv**

    The response vector to go along with the train data from the full feature matrix.

* **./data/X_test.csv**

    The original preprocessed test data split from the full 26-column feature matrix.

* **./data/y_test.csv**

    The response vector to go along with the test data from the full feature matrix.

* **./data/Xm_train.csv**

    Preprocessed, resampled, reduced feature training data with only the 9 continuous features.

* **./data/ym_train.csv**

    The response vector to go along with the reduced feature, resampled test data.

* **./data/Xm_test.csv**

    Preprocessed reduced feature test data that is the test partition complementing `./data/Xm_train.csv`.  9 columns.

* **./data/ym_train.csv**

    The response vector to go along with the reduced feature test data `./data/Xm_test.csv`.

## Pickles

The various pickle files of the models/their wrapper objects can be found here. Descriptions contain instructions for accessing certain fields and properties that may be useful or desirable.

* **dtc_rfecv.pickle**

    Pickle of the `RFECV` object containing a `DecisionTreeClassifier` as the `estimator_` used to greedily select features with cross validation on the training set data in `X_train.csv`. A useful attribute besides the model accessed from `estimator_` is the `support_` attribute, a boolean mask of the selected features from the columns of `X_train.csv`.

* **dtc_gscv.pickle**

    Pickle of the `GridSearchCV` object containing a `DecisionTreeClassifier` as the estimator used, fit on the resampled reduced feature data from `Xm_train.csv`, **not** the full feature data from `X_train.csv`. Useful attributes are `cv_results_` for the full set of cross validation metrics. Can retrieve the best fitting estimator across all CV folds by through `best_estimator_`, its cross validation score through `best_score_`, and the mean cross validation scores through `cv_results_["mean_test_score"]`, which I like to call `.mean()` on.

* **ada_stump_gscv.pickle**

    Pickle of the `GridSearchCV` object containing an `AdaBoostClassifier` as the estimator, with `DecisionTreeClassifier(max_depth = 1)` as the base estimator (a tree stump), fit on the resampled reduced feature data from `Xm_train.csv`. Contains 80 stumps. Refer to above for useful attributes of the `GridSearchCV` object. Can retrieve the base estimators through the `estimators_` property of the `AdaBoostClassifier`.

* **ada_tuned_gscv.pickle**

    Pickle of the `GridSearchCV` object containing an `AdaBoostClassifier` as the estimator, with a full `DecisionTreeClassifier` as the base estimator, fit on the resampled reduced feature data from `Xm_train.csv`. Contains 50 trees. Refer to above for useful attributes of the `GridSearchCV` object and for how to retrieve the base estimators.

* **ada_ext_gscv.pickle**

    Pickle of the `GridSearchCV` object containing an `AdaBoostClassifier` as the estimator, with a full `DecisionTreeClassifier` as the base estimator, fit on the resampled reduced feature data from `Xm_train.csv`. Contains 150 trees. Refer to above for useful attributes of the `GridSearchCV` object and how to for retrieve the base estimators.

* **ada_fstump_gscv.pickle.zip**

    Zipped pickle of the `GridSearchCV` object containing an `AdaBoostClassifier` as the estimator, with a decision tree stump as the base estimator, fit on the full feature data from `X_train.csv`. True file size is over the 100 MB Git file size limit. Contains 80 stumps. Refer to above for useful attributes of the `GridSearchCV` object and for how to retrieve the base estimators.