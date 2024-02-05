# install.packages(c("mice", "caret", "e1071"))
library(mice)
library(caret)
library(e1071)

# load into a data frame
Data <- as.data.frame(read.csv("mammographic_masses.data"))

str(Data)
dim(Data)

# convert from "?" to NA
Data[Data == "?"] <- NA

# change columns type
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

# impute values
imputed_model <- mice(Data)
imputed_Data <- complete(imputed_model)

str(imputed_Data)

# plot a histogram to illustrate the distribution of values for the label
library(ggplot2)
ggplot(imputed_Data, aes(x = imputed_Data$Severity)) +
  geom_bar(width=0.7, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Severity", x = "Severity", y = "Frequency")


# training and test
set.seed(1)
train_indices <- createDataPartition(imputed_Data$Severity, p = 0.8, list = FALSE)
train_data <- imputed_Data[train_indices, ]
test_data <- imputed_Data[-train_indices, ]

# svm model
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