---
title: "Machine Learning Project"
author: "Lucie AD"
date: "28 March 2018"
output: 
  html_document:
    keep_md: true
    fig_width: 12
    fig_height: 16
---



### Introduction

From Assignment: Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. People regularly quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, the aim is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict if the participants conduct the exercises properly. They were asked to perform barbell lifts correctly and incorrectly in five different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Data preparation

Exploratory data analysis (str not executed here) revealed a large number of observations (19622) with a large number of parameters each (160). However, for a large number of parameters, 98% of the data (19216 out of 19622 observations) are missing.


```r
## Load libraries
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)

## Download dataset
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if(!exists("training.csv")) download.file(url1, destfile = "training.csv", method="curl")
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if(!exists("test.csv")) download.file(url2, destfile = "test.csv", method="curl")
dld <- date()

## Read in training file
training <- read.csv("training.csv", header=TRUE)

## Exploratory data analysis
## str(training)
FindNA <- data.frame(sapply(training, function(y) sum(length(which(is.na(y))))))
table(FindNA)
```

```
## FindNA
##     0 19216 
##    93    67
```

Machine learning algorithms do not deal well with missing data, so it should either be imputed or discarded. Since the column in question only have 2% of data, it does not make much sense to impute the data, so all columns with NAs were discarded. Since the dataset contained NAs as well as missing data, the data was read in again unifying all missing values. Only columns with complete data were kept. In a second step, metadata columns irrelevant for analysis in this context were removed before further analysis. These were identified by containing "X", "timestamp" or "window" in their column name. The resulting data set consisted of 54 columns.


```r
## Keep only complete columns that contain data
training2 <- read.csv("training.csv", header= TRUE, na.strings=c(""," ","NA"))
FindNA <- data.frame(sapply(training2, function(y) sum(length(which(is.na(y))))))
table(FindNA)
```

```
## FindNA
##     0 19216 
##    60   100
```

```r
training_small <- subset(training2, select=colMeans(is.na(training2)) == 0)
training_data <- training_small[ ,!grepl("X|timestamp|window", names(training_small))]
```

To assess the need for preprocessing the data and get ideaas for model choice, the remaining variables were summarized by their mean and plotted for each of the five classes. Users were coded by colour.


```r
## Plot all variables against each other have a look at how variables are spread
training_agg <- aggregate(.~user_name+classe, training_data, mean)
training_agg_long <- gather(training_agg, -user_name, -classe, key="variable", value="value")
g1 <- ggplot(training_agg_long, aes(x = classe, y = value, color = user_name)) +
  geom_point() + facet_wrap(~ variable, ncol=6, scales = "free") + theme_bw()
g1
```

![](Machine_Learning_Course_Project_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

The absolute values for the different parameters vary over three orders of magnitude and some of the parameters seem to follow a similar pattern. The data set was therefore checked for highly correlated parameters and linear dependencies and 21 highly correlated parameters were removed, reducing the dataset from 54 to 33 columns. To account for the different value, the remaining parameters scaled, centred and transformed. Data points vary more by user than across classes.

*Note: I did not manage to also remove paramters that are highly negatively correlated, at least not with Caret. If somebody did this or can explain why you would not want to do that, please let me know in the comments.* 


```r
## Remove highly correlated predictors
correlations <- cor(training_data[ ,2:53])
summary(correlations[upper.tri(correlations)])
```

```
##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
## -0.992008 -0.110080  0.002092  0.001790  0.092552  0.980924
```

```r
highly_correlated <- findCorrelation(correlations, cutoff = 0.75)
highly_correlated
```

```
##  [1] 10  1  9 22  4 36  8  2 37 35 38 21 34 23 25 13 48 45 31 33 18
```

```r
training_uncorrelated <- training_data[,-highly_correlated]
ncol(training_uncorrelated)
```

```
## [1] 33
```

```r
correlations2 <- cor(training_uncorrelated[,2:32])
summary(correlations2[upper.tri(correlations2)])
```

```
##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
## -0.974932 -0.082352  0.002535  0.005275  0.069876  0.845563
```

```r
## Check for linear dependencies
comboInfo <- findLinearCombos(training_uncorrelated[,2:32])
comboInfo
```

```
## $linearCombos
## list()
## 
## $remove
## NULL
```

```r
## Center, scale and transform
preProcValues <- preProcess(training_uncorrelated, method = c("center", "scale", "YeoJohnson"))
training_final <- predict(preProcValues, training_uncorrelated)
```

### Model selection

Since many parameters have a high variance, it seemed safest to perform cross-validation with a leave-one-out method, to avoid cross-validating on an unrepresentative subsample. However, the computer could not cope with the required computing, so the standard cross-validation method of bootstrappping with 25 repeats was chosen in the end. Three approaches, predicting with trees (rpart), bagging with trees (gbm), and random forest (rf) were chosen and their performances compared.


```r
## Define cross validation
## fitControl <- trainControl(method = "loocv", number = 1)

## Fit different models
set.seed(23580)
if(!exists("modelFit_rpart")) modelFit_rpart <- train(classe~.,data=training_final, method="rpart")
modelFit_rpart
```

```
## CART 
## 
## 19622 samples
##    32 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa    
##   0.02054551  0.5389778  0.4106404
##   0.03567868  0.4712914  0.3087865
##   0.06318544  0.3905057  0.1782395
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.02054551.
```

```r
if(!exists("modelFit_gbm")) modelFit_gbm <- train(classe~.,data=training_final, method="gbm", verbose=FALSE)
modelFit_gbm
```

```
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    32 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.6998093  0.6181286
##   1                  100      0.7677189  0.7055378
##   1                  150      0.8060544  0.7542863
##   2                   50      0.8177179  0.7689569
##   2                  100      0.8710325  0.8366532
##   2                  150      0.8995766  0.8728236
##   3                   50      0.8618696  0.8249914
##   3                  100      0.9128295  0.8896173
##   3                  150      0.9371939  0.9204837
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
if(!exists("modelFit_rf")) modelFit_rf <- train(classe~.,data=training_final, method="rf")
modelFit_rf
```

```
## Random Forest 
## 
## 19622 samples
##    32 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9918942  0.9897452
##   17    0.9893010  0.9864646
##   32    0.9768322  0.9706872
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

Predicting with trees (rpart) returned an accuracy of 53.9% and therefore an in-sample error of 46.1%. The out-of-sample error will be higher and given the high in-sample error, unacceptably high. Bagging with trees (gbm) returned an accuracy of 93.7 % and therefore an in-sample error of 6.3%. Again the out-of-sample error will be higher and approximately one our of ten samples will still be misclassified. The random forest approach (rf) returned a very high accuracy of 99.1% and therefore a very small in-sample error of 0.9%. even thought the out-of-sample error will be higher, it is very likely to be in an acceptable range and chances are high that all 20 samples in the test set will be predicted correctly. The random forest approach was therefore chosen to predict classes in the test set.

### Prediction

The test set was prepared in the same way as the training set and class was predicted with a random forest approach.


```r
## Prepare test set in the same way
test <- read.csv("test.csv", header= TRUE, na.strings=c(""," ","NA"))
test_small <- subset(test, select=colMeans(is.na(training2)) == 0)
test_data <- test_small[ ,!grepl("X|timestamp|window", names(test_small))]

test_uncorrelated <- test_data[,-highly_correlated]
test_final <- predict(preProcValues, test_uncorrelated)

## Predict values for test set
predict(modelFit_rf, newdata=test_final)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Conclusion

The random forest approach predicted 20 out of 20 samples correctly but was also by far the most computationally intense approach. Further work could therefore focus on speeding up this process, e.g. by further reducing the number of parameters by removing highly negatively correlated parameters or by using a principal component approach.

