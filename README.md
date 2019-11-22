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