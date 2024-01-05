# load data into a data frame using read.csv function
Data =  as.data.frame(read.csv("data/mammographic_masses.data"))

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
Data$Malign <- as.factor(Data$Malign)
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

# 'mice': R package for Multivariate Imputation by Chained Equations. 
# It imputes missing values using regression models, iteratively updating variables.
# Multiple imputations are generated for uncertainty consideration.
# (Multiple imputations involve creating several versions of the data frame, 
# each with different imputed values for the missing data. These datasets 
# collectively reflect the uncertainty about the true values of the missing observations.)
#install.packages("mice")
library(mice)

# impute data and create the updated data frame
imputed_model <- mice(Data)
imputed_Data <- complete(imputed_model)

# inspect the new data frame
str(imputed_Data)

# now, we might consider our data ready for predictions

column_type <- class(imputed_Data$Severity) 
print(column_type)

# first, we want to create training and test sets
#install.packages("caret")
library(caret)

# smth like random_state from Python
set.seed(1)

# p is the proportion of data that is included in the training set
train_indices <- createDataPartition(imputed_Data$Severity, p = 0.8, list = FALSE)

# here we create train and test sets
train_data <- imputed_Data[train_indices, ]
test_data <- imputed_Data[-train_indices, ]


logistic_regression_model <- glm(Severity ~ BI.RADS.assessment + Age + Shape + Malign + Density, 
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
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))


