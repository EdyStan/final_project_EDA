### Classification of Benign and Malignant Mammographic Masses Based on BI-RADS Attributes

The program was written in R, and the studies were applied over the dataset found at the following address: https://archive.ics.uci.edu/dataset/161/mammographic+mass.

Several preprocessing techniques were used:

- Changed the data types to the most relevant types in R, such as `factor`.
- Imputed the missing values using MICE algorithm (it iteratively updates the values using linear regression until convergence is achieved) and MissForest algorithm (for each feature with missing values, it trains a Random Forest on the other features to predict the missing values. Continue iteratively until convergence). 

The models used to classify the data are:

- K-Nearest Neighbors (KNN).
- Logistic Regression.
- Random Forest Classifier.
- Support Vector Machine (SVM).

The best results were achieved with MissForest preprocessing and Random Forest Classifier as the predictor (accuracy = 0.853, f1_score = 0.849), meaning that the model is not only making accurate predictions but also effectively capturing relevant instances from the dataset. 

Of course, the model’s results could be improved. Some of my ideas would be: 

- Using the interaction between certain features as separate features. 

- Generating artificial points to handle the data imbalance (some options would be SMOTE or ADASYN). 

- Applying logarithmic or polynomial scale to generate new, more relevant dimensions of the data set. 

- Trying out different architectures of neural networks. 

More details are presented in `Documentation.pdf`, and the code is written in `models.R`.

Contributors:

- [Eduard Stan](https://github.com/EdyStan).
- [Oana Sîrbu](https://github.com/Oana4).
- [Sebi Barbu](https://github.com/sebibarbu).
- [Geo Plăian](https://github.com/geoplaian).
