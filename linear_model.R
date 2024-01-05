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
install.packages("mice")
library(mice)

# impute data and create the updated data frame
imputed_model <- mice(Data)
imputed_Data <- complete(imputed_model)

# inspect the new data frame
str(imputed_Data)

# now, we might consider our data ready for predictions
##################################################################

# create new variable "severity_numeric" to have the outcome as 0(benign) or 1(malignant)
imputed_Data$severity_numeric <- as.numeric(imputed_Data$Severity) -1

# create a linear model
Linear_Model <- lm(severity_numeric ~ BI.RADS.assessment + Age + Shape + Malign + Density, data = imputed_Data)
summary(Linear_Model)



