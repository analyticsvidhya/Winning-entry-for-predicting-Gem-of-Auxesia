## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
##                                                                            ##
##   RStudio: Version 0.99.447                                                ##
##   Revolution R Open 8.0.3: Using CRAN snapshot taken on 2015-04-01         ##
##   R version 3.1.3 (2015-03-09) -- "Smooth Sidewalk"                        ##
##   Platform: x86_64-apple-darwin13.4.0 (64-bit)                             ## 
##                                                                            ##
##   Packages used:                                                           ##
##     a. caret: ver 6.0-41                                                   ##
##     b. flexclust: ver 1.3-4                                                ##  
##     c. randomForest: ver 4.6-10                                            ##
##     d. miscTools: 0.6-16                                                   ##
##                                                                            ##         
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

## Online hackathon - July 11-12, 2015
## Challenge: Predict the gem of Auxesia for Magazino!
## Link: http://discuss.analyticsvidhya.com/t/predict-the-gem-of-auxesia-for-magazino/2227/28

## Approach: a. Divide and conquer
##           b. Split data sets into clusters and find optimal model for each cluster
##           c. Fit models using log of shares to circumvent the skewness

### LET THE FUN BEGIN :D

## Set Working Directory
setwd("~/+Analytic_Vidhya")

## Read train and test csv files
train = read.csv("train.csv")
test = read.csv("test.csv")

## Generate train and test for K-Means
limitedTrain = train
limitedTrain$id = NULL
limitedTrain$Day_of_publishing = NULL
limitedTrain$Category_article = NULL
limitedTrain$shares = NULL

limitedTest = test
limitedTest$id = NULL
limitedTest$Day_of_publishing = NULL
limitedTest$Category_article = NULL

## Preprocess using caret 
library(caret)
preproc = preProcess(limitedTrain)
normTrain = predict(preproc, limitedTrain)
normTest = predict(preproc, limitedTest)

## Generate three clusters (optimal found via hit-and-trial
km <- kmeans(normTrain, centers=3)

## Generate classification for train and test sets
library(flexclust)
km.kcca = as.kcca(km, normTrain)
clusterTrain = predict(km.kcca)
clusterTest = predict(km.kcca, newdata=normTest)
table(clusterTrain)
table(clusterTest)

## Subset the train and test sets
train1 = subset(train, clusterTrain == 1)
train2 = subset(train, clusterTrain == 2)
train3 = subset(train, clusterTrain == 3)

test1 = subset(test, clusterTest == 1)
test2 = subset(test, clusterTest == 2)
test3 = subset(test, clusterTest == 3)

## Fit models or each cluster .. Use all predictors except id
## Use log(shares) to alleviate the skewness of shares
library(randomForest)
mod_rf1 = randomForest(log(shares) ~ . - id, data=train1, ntree=10000, importance = TRUE, keep.forest = TRUE, do.trace = 1000)
mod_rf2 = randomForest(log(shares) ~ . - id, data=train2, ntree=10000, importance = TRUE, keep.forest = TRUE, do.trace = 1000)
mod_rf3 = randomForest(log(shares) ~ . - id, data=train3, ntree=10000, importance = TRUE, keep.forest = TRUE, do.trace = 1000)

## Test the prediction on train set
preds_train1 = predict(mod_rf1, train1)
preds_train2 = predict(mod_rf2, train2)
preds_train3 = predict(mod_rf3, train3)

preds_train = c(exp(preds_train1), exp(preds_train2), exp(preds_train3))
vals_train = c(train1$shares, train2$shares, train3$shares)

## Generate rsquared and rmse
library(miscTools)
rSquared(vals_train,vals_train - preds_train)
sqrt(mean((vals_train-preds_train)^2))

## Generate predictions on test set
preds_test1 = predict(mod_rf1, test1)
preds_test2 = predict(mod_rf2, test2)
preds_test3 = predict(mod_rf3, test3)

## Combine results from individual sets after exponentiating the predictions
preds_test = c(exp(preds_test1), exp(preds_test2), exp(preds_test3))
allIds = c(test1$id, test2$id, test3$id)

## Sort as per id and generate the file for submission
out_df = data.frame(allIds, preds_test)
out_df = out_df[with(out_df, order(allIds, preds_test)), ]
names(out_df) = c("id", "predictions")
write.csv(out_df, "KM_RF_R_AKumar.csv", row.names=F, quote=F)
