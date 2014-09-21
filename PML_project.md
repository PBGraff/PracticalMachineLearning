# Practical Machine Learning Course Project:  Weight Lifting Exercise
Philip Graff  
September 21, 2014  

## Introduction
There are currently many self-monitoring devices on the market, such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit*. These allow the wearers to collect a large amount of data about their personal activity and have helped start the 'quantified self' movement. In this project, we analyze the data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants as they perform barbell lifts in 5 different ways. Using this recorded data, we create a model and predict how lift was performed.

The five methods are as follows:

1. exactly according to the specification (A)
1. throwing the elbows to the front (B)
1. lifting the dumbbell only halfway (C)
1. lowering the dumbbell only halfway (D)
1. throwing the hips to the front (E)

More information about the data is available from <http://groupware.les.inf.puc-rio.br/har> and the paper:  Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

## Loading packages
Several packages will be needed for this analysis. We load them in the beginning in order to have the tools available throughout.

```r
require(ggplot2,quietly=TRUE)
require(lattice,quietly=TRUE)
require(caret,quietly=TRUE)
require(rattle,quietly=TRUE)
require(rpart,quietly=TRUE)
require(randomForest,quietly=TRUE)
require(plyr,quietly=TRUE)
require(gbm,quietly=TRUE)
require(e1071,quietly=TRUE)
require(klaR,quietly=TRUE)
```

## Obtaining and Cleaning the Data
We begin by downloading the training and test data (tested 15 Sept 2014) and then loading it into data frames.

```r
if( !file.exists("pml-training.csv") )
        download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile="pml-training.csv",method="curl")
if( !file.exists("pml-testing.csv") )
        download.file(url="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile="pml-testing.csv",method="curl")
training <- read.csv("pml-training.csv",header=TRUE)
testing <-  read.csv("pml-testing.csv",header=TRUE)
```

Many columns are mostly 'NA' values while others have none, so we begin by eliminating those with 'NA' values. We then eliminate columns that are timestamps and window information, since these are irrelevant for our task. We also eliminate the first two columns, as they are just a trial number and name of the lifter.

```r
colKeep <- colSums(is.na(training))==0
training <- training[,colKeep]
testing <- testing[,colKeep]
timeCols <- grep("timestamp|window",names(training))
training <- training[,-c(1,2,timeCols)]
testing <- testing[,-c(1,2,timeCols)]
```

In our final removal of possible predictors, we eliminate those variables that have a near-zero variance. As these do not vary significantly between different trials, they will be poor predictors.

```r
nzv <- nearZeroVar(training)
training <- training[,-nzv]
testing <- testing[,-nzv]
```
This leaves us with 52 predictor variables to use to predict the 'classe' variable, which is the method in whch the lift was performed.

## Exploratory Analysis
Our first look at the data is to simply histogram the occurrences of each lift method in the training data. This will show us if any are over- or under-represented.

![plot of chunk classe_hist](./PML_project_files/figure-html/classe_hist.png) 

Although method A is slightly over-represented compared to the others, it is not so drastic that it will cause problems in the training.

We then make a scatter plot of a few of the variables, in particular those from the accelerometers in the arms of the participants. The different methods are distinguished by color and it is clear from these that no simple relationship between the observed variables exists and more complex models will be needed. A random subset of 5% of the training data is used so that the points are visible.

![plot of chunk pairs_plot](./PML_project_files/figure-html/pairs_plot.png) 

For an initial analysis of the complexity of the problem, we use this same small subset of the data and fit a single decision tree. We analyze this to determine how well it performs.


```r
modExpFit<-train(smallTrain$classe~.,data=smallTrain,method="rpart")
```
![plot of chunk exploratory_plot_1](./PML_project_files/figure-html/exploratory_plot_1.png) 

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 239   0   5  11  24
##          B  43  50  48  18  31
##          C  12   2 132  22   4
##          D  27   1  31  65  37
##          E   1  12  16   9 143
## 
## Overall Statistics
##                                        
##                Accuracy : 0.64         
##                  95% CI : (0.609, 0.67)
##     No Information Rate : 0.328        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.543        
##  Mcnemar's Test P-Value : <2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.742   0.7692    0.569   0.5200    0.598
## Specificity             0.939   0.8475    0.947   0.8881    0.949
## Pos Pred Value          0.857   0.2632    0.767   0.4037    0.790
## Neg Pred Value          0.882   0.9811    0.877   0.9270    0.880
## Prevalence              0.328   0.0661    0.236   0.1272    0.243
## Detection Rate          0.243   0.0509    0.134   0.0661    0.145
## Detection Prevalence    0.284   0.1933    0.175   0.1638    0.184
## Balanced Accuracy       0.841   0.8084    0.758   0.7041    0.774
```

It is clear from the confusion matrix that this classification method does only moderately well. A more sophisticated approach is required.

## Fitting Models
In this seciton, we will fit and test various models in order to find which performs the best. For each, we set the random seed prior to fitting so that the results will be identically reproducible.

### Evaluation data
The first thing we need to do, however, is create a subset of the training data that can be used for evaluating the model fits and comparing against one another. Accuracy on this set will determine which is used for the final 20 examples.

```r
set.seed(1066)
inTrain <- createDataPartition(training$classe,p=0.8,list=FALSE)
trainDat <- training[inTrain,]
trainEval <- training[-inTrain,]
```
In this report we use the accuracy as our figure of merit. This is simply 1-Err, where Err is the out-of-bag error on predictions. Confidence intervals for accuracy/error measures are obtained by performing 10-fold cross-validation in all cases.

### Random Forests
The first model we train on the data is a random forest. We use 5-fold cross-validation for model fitting, where in each fold 80% of data is used for training and the remaining 20% for evaluation. The default settings for the random forest use 500 trees. Through exploratory analysis, it was found that mtry=27 was optimal, where mtry is the number of parameters considered at each node of a tree. The statistics on this model fit are displayed below and we see that the random forest performs very well.

```r
set.seed(12345)
modRF <- train(classe~., data=trainDat, method = "rf", tuneGrid = expand.grid(mtry = c(2,15,27,39,52)), trControl = trainControl(method = "cv", number = 5))
modRF
```

```
## Random Forest 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 12559, 12558, 12559, 12561, 12559 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    1         1      0.002        0.002   
##   15    1         1      0.002        0.002   
##   27    1         1      0.001        0.002   
##   39    1         1      0.002        0.003   
##   52    1         1      0.003        0.004   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 15.
```

The best random forest must then make predictions on the evaluation data we prepared for proper comparison with other methods.

```r
confMatRF <- confusionMatrix(trainEval$classe,predict(modRF,trainEval))
confMatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    4  755    0    0    0
##          C    0    6  675    3    0
##          D    0    0   10  633    0
##          E    0    0    2    2  717
## 
## Overall Statistics
##                                        
##                Accuracy : 0.993        
##                  95% CI : (0.99, 0.995)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.991        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.992    0.983    0.992    1.000
## Specificity             1.000    0.999    0.997    0.997    0.999
## Pos Pred Value          1.000    0.995    0.987    0.984    0.994
## Neg Pred Value          0.999    0.998    0.996    0.998    1.000
## Prevalence              0.285    0.194    0.175    0.163    0.183
## Detection Rate          0.284    0.192    0.172    0.161    0.183
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.995    0.990    0.995    0.999
```
We see an accuracy of **99.3118%** for the random forest method with a 95% confidence interval of (99.0002%,99.546%).

### Generalized Boosted Modeling with Trees
The next model to fit to the data involves boosted decision trees via the GBM method. This will train many decision trees sequentially, each time increasing the weight of data points predicted incorrectly the previous time and decreasing those that were predicted correctly. By combining the many trees, this produces a stronger predictor. Furthermore, GBM uses greedy selection of basis functions to improve classification. Cross-validation is used with 5 folds.

```r
set.seed(54321)
modGBM <- train(classe~., data=trainDat, method="gbm", tuneGrid = expand.grid(n.trees=seq(100,500,by=100),interaction.depth=seq(1,5),shrinkage=0.1), verbose = FALSE, trControl = trainControl(method = "cv", number = 5))
modGBM
```

```
## Stochastic Gradient Boosting 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 12559, 12559, 12558, 12559, 12561 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy  Kappa  Accuracy SD  Kappa SD
##   1                  100      0.8       0.8    0.013        0.016   
##   1                  200      0.9       0.8    0.008        0.010   
##   1                  300      0.9       0.9    0.009        0.012   
##   1                  400      0.9       0.9    0.009        0.011   
##   1                  500      0.9       0.9    0.010        0.012   
##   2                  100      0.9       0.9    0.008        0.010   
##   2                  200      0.9       0.9    0.008        0.010   
##   2                  300      1.0       1.0    0.006        0.007   
##   2                  400      1.0       1.0    0.003        0.004   
##   2                  500      1.0       1.0    0.003        0.004   
##   3                  100      0.9       0.9    0.009        0.011   
##   3                  200      1.0       1.0    0.003        0.004   
##   3                  300      1.0       1.0    0.002        0.003   
##   3                  400      1.0       1.0    0.002        0.003   
##   3                  500      1.0       1.0    0.002        0.002   
##   4                  100      1.0       0.9    0.006        0.008   
##   4                  200      1.0       1.0    0.003        0.004   
##   4                  300      1.0       1.0    0.002        0.002   
##   4                  400      1.0       1.0    0.002        0.003   
##   4                  500      1.0       1.0    0.003        0.003   
##   5                  100      1.0       1.0    0.004        0.006   
##   5                  200      1.0       1.0    0.001        0.002   
##   5                  300      1.0       1.0    0.002        0.002   
##   5                  400      1.0       1.0    0.002        0.002   
##   5                  500      1.0       1.0    0.002        0.002   
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 500,
##  interaction.depth = 5 and shrinkage = 0.1.
```

This ensemble of boosted trees must then be used to make predictions on the evaluation data we prepared for proper comparison with other methods.

```r
confMatGBM <- confusionMatrix(trainEval$classe,predict(modGBM,trainEval))
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    1    0    0    0
##          B    4  753    2    0    0
##          C    0    5  677    2    0
##          D    0    0    8  635    0
##          E    0    0    0    4  717
## 
## Overall Statistics
##                                        
##                Accuracy : 0.993        
##                  95% CI : (0.99, 0.996)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.992        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.992    0.985    0.991    1.000
## Specificity             1.000    0.998    0.998    0.998    0.999
## Pos Pred Value          0.999    0.992    0.990    0.988    0.994
## Neg Pred Value          0.999    0.998    0.997    0.998    1.000
## Prevalence              0.285    0.193    0.175    0.163    0.183
## Detection Rate          0.284    0.192    0.173    0.162    0.183
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.995    0.992    0.994    0.999
```
We see an accuracy of **99.3372%** for generalized boosted modeling with trees with a 95% confidence interval of (99.0304%,99.5666%).

### Naive Bayes
The next machine learning method that we try is Naive Bayes. In this, each variable is assumed to be independent to fit a probability density model to the training data. In addition to training using the raw parameter values, a fit using principal component analysis pre-processing was done. This did not perform as well, potentially due to PCA only conserving 95% of the variance of the original data and only minial reduction in input dimension. The PCA pre-processed fit is thus omitted here.

```r
set.seed(9069)
modNB <- train(classe~., data=trainDat, method="nb", tuneGrid = expand.grid(fL=c(0,3),usekernel=c(FALSE,TRUE)), trControl = trainControl(method = "cv", number = 5))
modNB
```

```
## Naive Bayes 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 12560, 12561, 12557, 12559, 12559 
## 
## Resampling results across tuning parameters:
## 
##   fL  usekernel  Accuracy  Kappa  Accuracy SD  Kappa SD
##   0   FALSE      0.5       0.4    0.04         0.05    
##   0    TRUE      0.7       0.7    0.02         0.03    
##   3   FALSE      0.5       0.4    0.04         0.05    
##   3    TRUE      0.7       0.7    0.02         0.03    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were fL = 0 and usekernel = TRUE.
```

These two Naive Bayes models are now evaluated on the test data set in order for proper comparison to the other results.

```r
confMatNB <- confusionMatrix(trainEval$classe,predict(modNB,trainEval))
confMatNB
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1005   20   19   65    7
##          B  166  481   58   47    7
##          C  153   44  467   20    0
##          D  144    1   80  387   31
##          E   37   77   30   18  559
## 
## Overall Statistics
##                                         
##                Accuracy : 0.739         
##                  95% CI : (0.725, 0.753)
##     No Information Rate : 0.384         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.665         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.668    0.772    0.714   0.7207    0.925
## Specificity             0.954    0.916    0.934   0.9244    0.951
## Pos Pred Value          0.901    0.634    0.683   0.6019    0.775
## Neg Pred Value          0.822    0.955    0.942   0.9543    0.986
## Prevalence              0.384    0.159    0.167   0.1369    0.154
## Detection Rate          0.256    0.123    0.119   0.0986    0.142
## Detection Prevalence    0.284    0.193    0.174   0.1639    0.184
## Balanced Accuracy       0.811    0.844    0.824   0.8225    0.938
```
We see an accuracy of **73.8975%** for Naive Bayes, with a 95% confidence interval of (72.493%,75.2664%).

### Linear and Quadratic Discriminant Analysis
Our final set of methods of classification is linear and quadratic discriminant analysis. This will form linear or quadratic decision boundaries in the input feature space. These are determined by fitting multivariate Guassians to each class. Linear discriminant analysis requires that they all use the same covariance matrix, while quadratic discriminant analysis allows the covariances to vary. These fits were repeated with PCA pre-processing, but as with Naive Bayes there was actually a reduction in the performance so they are omitted here.

```r
set.seed(31269)
modLDA <- train(classe~., data=trainDat, method="lda", trControl = trainControl(method = "cv", number = 5))
set.seed(57348)
modQDA <- train(classe~., data=trainDat, method="qda", trControl = trainControl(method = "cv", number = 5))
modLDA
```

```
## Linear Discriminant Analysis 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 12558, 12560, 12560, 12561, 12557 
## 
## Resampling results
## 
##   Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.7       0.6    0.006        0.007   
## 
## 
```

```r
modQDA
```

```
## Quadratic Discriminant Analysis 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 12561, 12561, 12558, 12559, 12557 
## 
## Resampling results
## 
##   Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.9       0.9    0.004        0.005   
## 
## 
```

These four models are now evaluated on the test data set in order for proper comparison to the other results.

```r
confMatLDA <- confusionMatrix(trainEval$classe,predict(modLDA,trainEval))
confMatQDA <- confusionMatrix(trainEval$classe,predict(modQDA,trainEval))
confMatLDA
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 911  25  93  85   2
##          B 119 482  98  25  35
##          C  70  76 447  76  15
##          D  41  25  84 469  24
##          E  30 120  73  65 433
## 
## Overall Statistics
##                                         
##                Accuracy : 0.699         
##                  95% CI : (0.684, 0.713)
##     No Information Rate : 0.298         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.619         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.778    0.662    0.562    0.651    0.851
## Specificity             0.926    0.913    0.924    0.946    0.916
## Pos Pred Value          0.816    0.635    0.654    0.729    0.601
## Neg Pred Value          0.907    0.922    0.893    0.923    0.976
## Prevalence              0.298    0.186    0.203    0.184    0.130
## Detection Rate          0.232    0.123    0.114    0.120    0.110
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.852    0.788    0.743    0.799    0.883
```

```r
confMatQDA
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1042   39   17   15    3
##          B   36  631   85    1    6
##          C    2   38  641    1    2
##          D    0    3  104  528    8
##          E    0   28   32   19  642
## 
## Overall Statistics
##                                         
##                Accuracy : 0.888         
##                  95% CI : (0.878, 0.898)
##     No Information Rate : 0.275         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.859         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.965    0.854    0.729    0.936    0.971
## Specificity             0.974    0.960    0.986    0.966    0.976
## Pos Pred Value          0.934    0.831    0.937    0.821    0.890
## Neg Pred Value          0.986    0.966    0.927    0.989    0.994
## Prevalence              0.275    0.188    0.224    0.144    0.168
## Detection Rate          0.266    0.161    0.163    0.135    0.164
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.969    0.907    0.858    0.951    0.974
```
We see an accuracy of **69.8955%** for linear discriminant analysis and **88.8096%** for quadratic discriminant analysis. LDA has a 95% confidence interval of (68.4329%,71.3284%) and for QDA this is (87.7813%,89.7796%).

## Predictions on 20 Test Cases
From the comparisons of predictions on the evaluation data set, we see that the random forests  and generalized boosting methods produced the best results, with greater than 99% accuracy. We use both to make predictions for the testing data set with its 20 observations.

```r
testAnswersRF <- as.character(predict(modRF,testing[,-53]))
testAnswersGBM <- as.character(predict(modGBM,testing[,-53]))
testAnswersRF
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

```r
testAnswersGBM
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

Since these predictions are the same, we feel safe in printing them to individual text files for submission.


## Conclusions
On the whole, it appears that from the methods tested here, those using ensembles of decision trees were able to perform best. Random forests and generalized boosting both performed with over 99% accuracy on the evaluation data set. The other methods did not perform nearly as well, probably due to their inability to model the complex decision boundaries needed for this classification task. It was also found that principal component analysis did not help any of these. This is likely due to poor breakdown into principal components and low compression as many PCs were needed to keep 95% of the variance. Furthermore, the loss of 5% of the variance could be important for borderline cases, thus lowering the accuracy from applying the method to raw data.
