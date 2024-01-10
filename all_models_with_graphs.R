library(mice)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(class)
library(pROC)
library(glmnet)

# load data into a data frame using read.csv function
Data =  as.data.frame(read.csv("mammographic_masses_margin.data"))

# preview the data frame. We observe that missing values are represented by "?"
# and types are not ideal (chr for int values)
str(Data)
dim(Data)

# convert "?" to NA so that the interpreter can see these values as missing values
Data[Data == "?"] <- NA

# now, change types of the columns. We use 'factor' type for the categorical values
# (such as our label, 'Severity' column, which can have only two values)
# as 'Age' column has a wide range of values, we keep it integer.
Data$BI.RADS.assessment <- as.factor(Data$BI.RADS.assessment)
Data$Age <- as.integer(Data$Age)
Data$Shape <- as.factor(Data$Shape)
Data$Margin <- as.factor(Data$Margin)
Data$Density <- as.factor(Data$Density)
Data$Severity <- as.factor(Data$Severity)

# there is only one occurrence of value "55" in the whole BI.RADS.assessment feature
# so we decide to remove it completely. After we do that, we drop the unused levels.
Data <- Data[Data$BI.RADS.assessment != "55", ]
Data$BI.RADS.assessment <- droplevels(Data$BI.RADS.assessment)

# inspect the data frame once again to see the results
str(Data)

# count the missing values in each column
missing_values <- colSums(is.na(Data))

# print the number of missing values. As we can see, missing values are far from
# a few. We have the options impute or remove the rows with missing values. 
# we decide to impute them 
print(missing_values)

# impute data and create the updated data frame
imputed_model <- mice(Data)
imputed_Data <- complete(imputed_model)

# inspect the new data frame
str(imputed_Data)

# plot a histogram to illustrate the distribution of values for the label
ggplot(imputed_Data, aes(x = imputed_Data$Severity)) +
  geom_bar(width=0.7, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Severity", x = "Severity", y = "Frequency")

#we check 'Severity' type 
column_type <- class(imputed_Data$Severity) 
print(column_type)

#now, we might consider our data ready
#################################################################

# Create training and test sets
# p is the proportion of data that is included in the training set
# set.seed is smth like random_state from Python
set.seed(1)
train_indices <- createDataPartition(imputed_Data$Severity, p = 0.8, list = FALSE)
train_data <- imputed_Data[train_indices, ]
test_data <- imputed_Data[-train_indices, ]

#now we can start using our models for predictions
################################################################

#KNN model

# we create vectors to store results further used for plotting
x_knn <- c()  # to store k values
y_knn <- c()  # to store F1 scores (the chosen metric)

# Define a range of k values for grid search
k_values <- seq(1, 30, by = 5)  

# Loop through each k value from k_values
for (k in k_values) {
  
  print(paste('for k = ', k))
  # check various configurations of the KNN model
  knn_model <- knn(train = train_data[, c("BI.RADS.assessment", "Age", "Shape", "Margin", "Density")],
                   test = test_data[, c("BI.RADS.assessment", "Age", "Shape", "Margin", "Density")],
                   cl = train_data$Severity,
                   k = k)
  
  # we evaluate the KNN model
  knn_confusion_matrix <- table(Predicted = knn_model, Actual = test_data$Severity)
  
  # then we calculate F1 Score
  knn_accuracy <- sum(diag(knn_confusion_matrix)) / sum(knn_confusion_matrix)
  knn_precision <- knn_confusion_matrix["1", "1"] / sum(knn_confusion_matrix["1", ])
  knn_recall <- knn_confusion_matrix["1", "1"] / sum(knn_confusion_matrix[, "1"])
  knn_f1_score <- 2 * (knn_precision * knn_recall) / (knn_precision + knn_recall)
  
  print(paste("KNN Accuracy:", knn_accuracy))
  print(paste("KNN Precision:", knn_precision))
  print(paste("KNN Recall:", knn_recall))
  print(paste("KNN F1 Score:", knn_f1_score)) 
  
  # we store results
  x_knn <- c(x_knn, k)
  y_knn <- c(y_knn, knn_f1_score)
}

# Find the best k based on the maximum F1 score
best_k <- x_knn[which.max(y_knn)]

# Print the best k
print(paste("Best K for KNN:", best_k))
print(paste('k-values: ', x_knn))
print(paste('f1_scores: ', y_knn))

# Create a scatter plot
plot(x_knn, y_knn, type = "p", col = "red", pch = 16,
     xlab = "Number of neighbors (k)", ylab = "F1 Score", 
     main = "k tuning")

# Connect scatter points with a line for better visualization
lines(x_knn, y_knn, col = "black", type = "l")

# Just for clarity, we will work with the best KNN model separately,
# to present all the metrics 
best_knn_model <- knn(train = train_data[, c("BI.RADS.assessment", "Age", "Shape", "Margin", "Density")],
                 test = test_data[, c("BI.RADS.assessment", "Age", "Shape", "Margin", "Density")],
                 cl = train_data$Severity,
                 k = 11) 

# Evaluate the KNN model
best_knn_confusion_matrix <- table(Predicted = best_knn_model, Actual = test_data$Severity)

# Plot the confusion matrix
table(best_knn_confusion_matrix)

# Calculate evaluation metrics
best_knn_accuracy <- sum(diag(best_knn_confusion_matrix)) / sum(best_knn_confusion_matrix)
best_knn_precision <- best_knn_confusion_matrix["1", "1"] / sum(best_knn_confusion_matrix["1", ])
best_knn_recall <- best_knn_confusion_matrix["1", "1"] / sum(best_knn_confusion_matrix[, "1"])
best_knn_f1_score <- 2 * (best_knn_precision * best_knn_recall) / (best_knn_precision + best_knn_recall)

# Print the evaluation results for KNN
print(paste("Best KNN Accuracy:", knn_accuracy))
print(paste("Best KNN Precision:", knn_precision))
print(paste("Best KNN Recall:", knn_recall))
print(paste("Best KNN F1 Score:", knn_f1_score)) 

TP_knn <- best_knn_confusion_matrix[2, 2]
FP_knn <- best_knn_confusion_matrix[1, 2]
FN_knn <- best_knn_confusion_matrix[2, 1]
TN_knn <- best_knn_confusion_matrix[1, 1]

# Print the values
cat("True Positives (TP):", TP_knn, "\n")
cat("False Positives (FP):", FP_knn, "\n")
cat("False Negatives (FN):", FN_knn, "\n")
cat("True Negatives (TN):", TN_knn, "\n")

################################################################

# Logistic Regression model

# only for Grid search we will use the glmnet package
# that allows as to play with the regularization parameter of lr_model

lr_predictors <- model.matrix(Severity ~ BI.RADS.assessment + Age + Shape + Margin + Density, data = train_data)
lr_predicted <- as.numeric(train_data$Severity) - 1

# Fit logistic regression model with regularization and cross-validation
# it finds internally the best lambda (regularization parameter)
lr_model <- cv.glmnet(lr_predictors, lr_predicted, alpha = 0, family = "binomial")

# Extract the best lambda
best_lambda <- lr_model$lambda.min

# Print the best lambda
print(paste("Best Lambda for Logistic Regression:", best_lambda))

# Now that we explored the way we can fine tune the regularization parameter 
# of such a model, for documenting the metrics achieved we will use a classical
# approach, with the glm function

logistic_regression_model <- glm(Severity ~ BI.RADS.assessment + Age + Shape + Margin + Density, 
                                 data = train_data,
                                 family = "binomial")

levels(train_data$BI.RADS.assessment)  # Check levels in training data
levels(test_data$BI.RADS.assessment)

# Make predictions on the 'unseen' test set
lr_predictions <- as.factor(as.numeric(predict(logistic_regression_model, newdata = test_data, type = "response") > 0.5))

lr_conf_matrix <- table(Predicted = lr_predictions, Actual = test_data$Severity)

# Calculate accuracy
lr_accuracy <- sum(diag(lr_conf_matrix)) / sum(lr_conf_matrix)

# Calculate precision, recall, and F1 score
lr_precision <- lr_conf_matrix[2, 2] / sum(lr_conf_matrix[, 2])
lr_recall <- lr_conf_matrix[2, 2] / sum(lr_conf_matrix[2, ])
lr_f1_score <- 2 * (lr_precision * lr_recall) / (lr_precision + lr_recall)

# Print the evaluation results
print(paste("LR Accuracy:", lr_accuracy))
print(paste("LR Precision:", lr_precision))
print(paste("LR Recall:", lr_recall))
print(paste("LR F1 Score:", lr_f1_score))

# Extract TP, FP, FN, TN values
TP_lr <- conf_matrix[2, 2]
FP_lr <- conf_matrix[1, 2]
FN_lr <- conf_matrix[2, 1]
TN_lr <- conf_matrix[1, 1]

# Print the values
cat("True Positives (TP):", TP_lr, "\n")
cat("False Positives (FP):", FP_lr, "\n")
cat("False Negatives (FN):", FN_lr, "\n")
cat("True Negatives (TN):", TN_lr, "\n")


# plot the ROC curve of our label and the predicted data. 
# the input variables need to be numerical or ordered, so we decided to order them
target <- factor(test_data$Severity, ordered=TRUE)
pred <- factor(predictions, ordered=TRUE)
roc_curve <- roc(target, pred)
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2, xlab="Precision", ylab="Recall")

################################################################

# Random Forest Classifier model

# Performing a search of the best ntree parameter, following the same template
# as we used for the KNN model
y_rf <- c()  # to store F1 scores
x_rf <- c()  # to store the number of trees (ntree)

# We define a range of ntree values for grid search
ntree_values <- seq(100, 300, by = 50)

for (ntree in ntree_values) {
  
  print(paste('for ntree = ', ntree))
  
  # Fit Random Forest Classifier model
  random_forest_model <- randomForest(Severity ~ BI.RADS.assessment + Age + Shape + Margin + Density, 
                                      data = train_data,
                                      ntree = ntree)
  
  # Make predictions on the 'unseen' test set
  predictions <- predict(random_forest_model, newdata = test_data)
  
  # Evaluate the model
  rf_confusion_matrix <- confusionMatrix(predictions, test_data$Severity)
  
  # Calculate F1 Score
  rf_precision <- rf_confusion_matrix$byClass["Precision"]
  rf_recall <- rf_confusion_matrix$byClass["Recall"]
  rf_f1_score <- 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)
  
  print(paste('rf_f1-score: ', rf_f1_score))
  
  # Store results
  x_rf <- c(x_rf, ntree)
  y_rf <- c(y_rf, rf_f1_score)
}

# we find the ntree value with the maximum F1 score
best_ntree <- x_rf[which.max(y_rf)]

# then we print the best ntree
print(paste("The best number of trees for a Random Forest Classifier is: ", best_ntree))

ggplot(data.frame(x = x_rf, y = y_rf), aes(x, y)) +
  geom_point(color = 'red') +
  geom_line() +
  labs(x = 'Number of trees', y = 'F1 score', title = 'ntree tuning') +
  theme_minimal()

# Classical approach, with the best model found as best_ntree = 200
random_forest_model <- randomForest(Severity ~ BI.RADS.assessment + Age + Shape + Margin + Density, 
                                    data = train_data,
                                    ntree = 250)

# Make predictions on the 'unseen' test set
rf_predictions <- predict(random_forest_model, newdata = test_data)

# Extract the confusion matrix from the list
conf_matrix <- table(Predicted = rf_predictions, Actual = test_data$Severity)

# Calculate accuracy
rf_accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Calculate precision, recall, and F1 score
rf_precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
rf_recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
rf_f1_score <- 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)

# Print the evaluation results
print(paste("RFC Accuracy:", rf_accuracy))
print(paste("RFC Precision:", rf_precision))
print(paste("RFC Recall:", rf_recall))
print(paste("RFC F1 Score:", rf_f1_score))

# Extract TP, FP, FN, TN values
TP_rf <- conf_matrix[2, 2]
FP_rf <- conf_matrix[1, 2]
FN_rf <- conf_matrix[2, 1]
TN_rf <- conf_matrix[1, 1]

# Print the results
cat("True Positives (TP):", TP_rf, "\n")
cat("False Positives (FP):", FP_rf, "\n")
cat("False Negatives (FN):", FN_rf, "\n")
cat("True Negatives (TN):", TN_rf, "\n")

################################################################

# SVM model

# we are searching the best value for the cost parameter, c, of a SVM model
y_svm <- c()  # to store F1 scores
x_svm <- c()  # to store cost values

# We define a range of cost values for grid search
cost_values <- c(0.1, 1, 10, 100)  # Adjust the range based on your preference

# Loop through each cost value
for (cost in cost_values) {
  
  print(paste('for cost = ', cost))
  
  # We fit the SVM model with different cost values
  svm_model <- svm(Severity ~ BI.RADS.assessment + Age + Shape + Margin + Density, 
                   data = train_data,
                   kernel = "radial",
                   cost = cost)
  
  # Make predictions on the 'unseen' test set
  svm_predictions <- predict(svm_model, newdata = test_data)
  
  # Evaluate the model
  svm_confusion_matrix <- table(Predicted = svm_predictions, Actual = test_data$Severity)
  
  # Calculate F1 Score
  svm_precision <- svm_confusion_matrix["1", "1"] / sum(svm_confusion_matrix["1", ])
  svm_recall <- svm_confusion_matrix["1", "1"] / sum(svm_confusion_matrix[, "1"])
  svm_f1_score <- 2 * (svm_precision * svm_recall) / (svm_precision + svm_recall)
  
  print(svm_f1_score)
  
  # Store results
  x_svm <- c(x_svm, cost)
  y_svm <- c(y_svm, svm_f1_score)
}

# Find the cost value with the maximum F1 score
best_cost <- x_svm[which.max(y_svm)]

# Print the best cost
print(paste("Best cost for an SVM model:", best_cost))

ggplot(data.frame(x = x_svm, y = y_svm), aes(x, y)) +
  geom_point(color = 'red') +
  labs(x = 'Cost', y = 'F1 score', title = 'Cost parameter tuning') +
  theme_minimal()


# also the straightforward approach is displayed below:
svm_model <- svm(Severity ~ BI.RADS.assessment + Age + Shape + Margin + Density, 
                 data = train_data,
                 kernel = "radial", 
                 cost = 10)   

# prediction on test
svm_predictions <- predict(svm_model, newdata = test_data)

# evaluate svm model
svm_confusion_matrix <- table(Predicted = svm_predictions, Actual = test_data$Severity)

# metrics
svm_accuracy <- sum(diag(svm_confusion_matrix)) / sum(svm_confusion_matrix)
svm_precision <- svm_confusion_matrix["1", "1"] / sum(svm_confusion_matrix["1", ])
svm_recall <- svm_confusion_matrix["1", "1"] / sum(svm_confusion_matrix[, "1"])
svm_f1_score <- 2 * (svm_precision * svm_recall) / (svm_precision + svm_recall)

# print metrics
print(paste("SVM Accuracy:", svm_accuracy))
print(paste("SVM Precision:", svm_precision))
print(paste("SVM Recall:", svm_recall))
print(paste("SVM F1 Score:", svm_f1_score))

# Create the confusion matrix
svm_confusion_matrix <- table(Predicted = svm_predictions, Actual = test_data$Severity)

# Extract TP, FP, FN, TN values
TP <- svm_confusion_matrix[2, 2]
FP <- svm_confusion_matrix[1, 2]
FN <- svm_confusion_matrix[2, 1]
TN <- svm_confusion_matrix[1, 1]

# Print the values
cat("True Positives (TP):", TP, "\n")
cat("False Positives (FP):", FP, "\n")
cat("False Negatives (FN):", FN, "\n")
cat("True Negatives (TN):", TN, "\n")



