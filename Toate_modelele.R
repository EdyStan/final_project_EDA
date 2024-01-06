library(mice)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(class)
library(pROC)

# load data into a data frame using read.csv function
Data =  as.data.frame(read.csv("mammographic_masses.data"))

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
knn_model <- knn(train = train_data[, c("BI.RADS.assessment", "Age", "Shape", "Margin", "Density")],
                 test = test_data[, c("BI.RADS.assessment", "Age", "Shape", "Margin", "Density")],
                 cl = train_data$Severity,
                 k = 20) 


# Evaluate the KNN model
knn_confusion_matrix <- table(Predicted = knn_model, Actual = test_data$Severity)

# Plot the confusion matrix
table(knn_confusion_matrix)

# Calculate evaluation metrics
knn_accuracy <- sum(diag(knn_confusion_matrix)) / sum(knn_confusion_matrix)
knn_precision <- knn_confusion_matrix["1", "1"] / sum(knn_confusion_matrix["1", ])
knn_recall <- knn_confusion_matrix["1", "1"] / sum(knn_confusion_matrix[, "1"])
knn_f1_score <- 2 * (knn_precision * knn_recall) / (knn_precision + knn_recall)

# Print the evaluation results for KNN
print(paste("KNN Accuracy:", knn_accuracy))
print(paste("KNN Precision:", knn_precision))
print(paste("KNN Recall:", knn_recall))
print(paste("KNN F1 Score:", knn_f1_score))

################################################################

#Logistic Regression model
logistic_regression_model <- glm(Severity ~ BI.RADS.assessment + Age + Shape + Margin + Density, 
                                 data = train_data,
                                 family = "binomial")

levels(train_data$BI.RADS.assessment)  # Check levels in training data
levels(test_data$BI.RADS.assessment)

# Make predictions on the 'unseen' test set
predictions <- as.factor(as.numeric(predict(logistic_regression_model, newdata = test_data, type = "response") > 0.5))

# Evaluate the final model
confusion_matrix <- confusionMatrix(predictions, test_data$Severity)

# extract statistics from the confusion matrix
accuracy <- confusion_matrix$overall["Accuracy"]
precision <- confusion_matrix$byClass["Precision"]
recall <- confusion_matrix$byClass["Recall"]
f1_score <- confusion_matrix$byClass["F1"]

# print the evaluation results
print(paste("LR Accuracy:", accuracy))
print(paste("LR Precision:", precision))
print(paste("LR Recall:", recall))
print(paste("LR F1 Score:", f1_score))


# plot the ROC curve of our label and the predicted data. 
# the input variables need to be numerical or ordered, so we decided to order them
target <- factor(test_data$Severity, ordered=TRUE)
pred <- factor(predictions, ordered=TRUE)
roc_curve <- roc(target, pred)
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2, xlab="Precision", ylab="Recall")

################################################################

#Random Forest Classifier model
random_forest_model <- randomForest(Severity ~ BI.RADS.assessment + Age + Shape + Margin + Density, 
                                    data = train_data,
                                    ntree = 200)

# Make predictions on the 'unseen' test set
predictions <- predict(random_forest_model, newdata = test_data)

# Evaluate the final model
confusion_matrix <- confusionMatrix(predictions, test_data$Severity)

# extract statistics from the confusion matrix
accuracy <- confusion_matrix$overall["Accuracy"]
precision <- confusion_matrix$byClass["Precision"]
recall <- confusion_matrix$byClass["Recall"]
f1_score <- confusion_matrix$byClass["F1"]

# print the evaluation results
print(paste("RFC Accuracy:", accuracy))
print(paste("RFC Precision:", precision))
print(paste("RFC Recall:", recall))
print(paste("RFC F1 Score:", f1_score))


################################################################

#SVM model
svm_model <- svm(Severity ~ BI.RADS.assessment + Age + Shape + Margin + Density, 
                 data = train_data,
                 kernel = "radial", # "radial", "polynomial"
                 cost = 1)          # cost parameter

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
