# Install required packages if not already installed
# install.packages(c("mice", "caret", "class"))
library(mice)
library(caret)
library(class)
install.packages(c("caret", "e1071"))



# Load data into a data frame using read.csv function
Data <- as.data.frame(read.csv("mammographic_masses.data"))

# Convert "?" to NA so that the interpreter can see these values as missing values
Data[Data == "?"] <- NA

# Change types of the columns
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

# Impute missing values
imputed_model <- mice(Data)
imputed_Data <- complete(imputed_model)

# inspect the new data frame
str(imputed_Data)

# plot a histogram to illustrate the distribution of values for the label
library(ggplot2)
ggplot(imputed_Data, aes(x = imputed_Data$Severity)) +
  geom_bar(width=0.7, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Severity", x = "Severity", y = "Frequency")


# Create training and test sets
set.seed(1)
train_indices <- createDataPartition(imputed_Data$Severity, p = 0.8, list = FALSE)
train_data <- imputed_Data[train_indices, ]
test_data <- imputed_Data[-train_indices, ]

# Fit KNN model
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