### Classification of Benign and Malignant Mammographic Masses Based on BI-RADS Attributes

The program was written in R, and the studies were applied over the dataset found at the following address: https://archive.ics.uci.edu/dataset/161/mammographic+mass.

Several preprocessing techniques were used:

- Data type changes to the most relevant types in R, such as `factor`.
- Imputation using `missforest` and `mice`.

The models used to classify the data are:

- K-Nearest Neighbours (KNN).
- Logistic Regression.
- Random Forest Classifier.
- Support Vector Machine (SVM).

Contributors:

- Eduard Stan (https://github.com/EdyStan).
- Oana Sîrbu (https://github.com/Oana4).
- Sebi Barbu (https://github.com/sebibarbu).
- Geo Plăian (https://github.com/geoplaian).

More details are presented in `Documentation.pdf`, and the code is written in `models.R`.
