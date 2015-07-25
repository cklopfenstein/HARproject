# EDA for ML project

setwd("~/Devel/R/DataScience/ML/Project")

library(dplyr)
library(data.table)
library(ggplot2)
library(AppliedPredictiveModeling)
library(caret)
library(randomForest)

set.seed(1234) # for reproducibility

trainFile <- "./data/pml-training.csv"
testFile <- "./data/pml-testing.csv"

dtTrain <- data.table(read.table(trainFile, header = TRUE, sep = ","))
dtTest <- data.table(read.table(testFile, header = TRUE, sep = ","))

histogram(dtTrain$classe)  # values: 'A', 'B', 'C', 'D', 'E'

qplot(user_name, roll_belt, colour = classe, data = dtTrain)
qplot(user_name, yaw_belt, colour = classe, data = dtTrain)
qplot(user_name, pitch_belt, colour = classe, data = dtTrain)

ATrain = filter(dtTrain, classe == 'A')
BTrain = filter(dtTrain, classe == 'B')
CTrain = filter(dtTrain, classe == 'B')
DTrain = filter(dtTrain, classe == 'D')
ETrain = filter(dtTrain, classe == 'E')

qplot(user_name, total_accel_belt, colour = classe, data = ATrain)


# classification tree
# need to install package randomForest
treeFit <- train(classe ~ roll_dumbbell + pitch_dumbbell + yaw_dumbbell, method = "rpart", data = dtTrain)

print(treeFit$finalModel)

# first maybe do some cleaning - deal with NAs, etc.

# should do PCA to reduce n(features) from 160 to something manageable
# first need to convert classe variable from char to numeric
# non-numeric values: user_name, new_window, classe (training), problem_id (testing)
preProc <- dtTrain
#preProc <- dtTrain[-user_name]
#preProc <- preProc[-new_window]
#preProc <- preProc[-classe]
#preProc <- dtTrain[,user_name:=NULL]
#preProc <- preProc[,new_window:=NULL]
#preProc <- preProc[,classe:=NULL]

#preProc <- na.omit(preProc)  # this removes all rows with NA

# can I use as.numeric on all columns?
# alternatively, looks like I want columns roll*, pitch*, yaw*, total*, gyros*, 
# accel*, magnet*
# can use grep for this
#numericCols <- grep("^(roll|pitch|yaw|total|gyros|accel|magnet).*", 
#                    names(preProc), value = TRUE) # count from 0?
numericCols <- grep("^(roll|pitch|yaw|total|gyros|accel|magnet).*", 
                    names(dtTrain), value = TRUE) # count from 0?
# don't need this step if using col names instead of indices
#numIndices <- unlist(lapply(numIndices, function(x) x + 1))
# 52 entries

# select columns with numeric values
#preProc <- preProc[,numIndices]  # no
preProc <- subset(preProc, TRUE, numericCols)

#preProc80 <- preProcess(preProc, method = "pca", thresh = 0.8)
#preProc80$numComp  # 12 principal components

#preProc95 <- preProcess(preProc, method = "pca", thresh = 0.95)
#preProc95$numComp  # 25 principal components

#preProc90 <- preProcess(preProc, method = "pca", thresh = 0.9)
#preProc90$numComp  # 19 principal components

# think I will use PCA with thresh = 0.80 - to reduce number of variables
# to fit from 52 to 12. 

# don't know how to use the object returned by preProcess(), but can
# specify 
#preProcess = "pca",
#trControl = trainControl(preProcOptions = list(thresh = 0.8)))
# in the train function.

# try it out here

trainData <- cbind(dtTrain[,dtTrain$classe], preProc)
colnames(trainData)[1] = "classe"

#rpartPCA <- train(classe ~ ., data = trainData, method = "rpart",
#                  preProcess = "pca",
#                  trControl = trainControl(preProcOptions = list(thresh = 0.8)))
rpartPCA <- train(classe ~ ., data = trainData, method = "rpart",
                  preProcess = "pca",
                  trControl = trainControl(preProcOptions = list(thresh = 0.95)))

confusionMatrix(rpartPCA, newdata=trainData)
# for thresh = 0.8
#Bootstrapped (25 reps) Confusion Matrix 

#(entries are percentages of table totals)

#Reference
#Prediction    A    B    C    D    E
#A 23.1 11.0 12.7  7.9  8.9
#B  1.6  2.6  1.5  2.0  2.2
#C  0.6  0.4  0.7  0.4  0.3
#D  0.7  1.5  0.4  2.0  0.9
#E  2.5  3.9  2.1  4.1  6.0

# for thresh = 0.95

#Bootstrapped (25 reps) Confusion Matrix 

#(entries are percentages of table totals)

#Reference
#Prediction    A    B    C    D    E
#A 26.0 14.4 15.3 11.2 10.5
#B  0.5  0.7  0.5  0.6  0.6
#C  0.4  0.4  0.7  0.3  0.3
#D  1.0  1.8  0.7  2.6  1.2
#E  0.4  2.1  0.3  1.8  5.8

rpartNoPCA <- train(classe ~ ., data = trainData, method = "rpart")

confusionMatrix(rpartNoPCA, newdata=trainData)

#Bootstrapped (25 reps) Confusion Matrix 

#(entries are percentages of table totals)

#Reference
#Prediction    A    B    C    D    E
#A 24.8  8.3  7.6  7.1  2.8
#B  0.8  6.3  0.6  2.8  2.4
#C  2.4  3.8  8.5  4.6  3.9
#D  0.2  0.6  0.6  1.7  0.6
#E  0.2  0.3  0.1  0.3  8.7

#
#  oops problem_id != classe
#testData <- cbind(dtTest[,dtTest$problem_id], subset(dtTest, TRUE, numericCols))
#colnames(testData)[1] = "classe"

#predTest <- predict(rpartPCA, newdata = testData)
#confusionTest <- confusionMatrix(testData$classe, predTest)

# need to split training data into training and validation sets

#inTrain <- createDataPartition(dtTrain$classe, p = 0.8, list = FALSE)
#training <- dtTrain[inTrain,]
#validation <- dtTrain[-inTrain,]

# some problems with data.table, so convert it to data frame, and try
# everything again

dfTrain <- as.data.frame.matrix(dtTrain)
dfTest <- as.data.frame.matrix(dtTest)

inTrain <- createDataPartition(dfTrain$classe, p = 0.8, list = FALSE)
training <- dfTrain[inTrain,]
validation <- dfTrain[-inTrain,]

numericCols <- grep("^(roll|pitch|yaw|total|gyros|accel|magnet).*", 
                    names(dfTrain), value = TRUE) # count from 0?

preProc <- subset(training, TRUE, numericCols)
trainData <- cbind(training[,"classe"], preProc)
colnames(trainData)[1] = "classe"

preProc <- subset(validation, TRUE, numericCols)
valData <- cbind(validation[,"classe"], preProc)
colnames(valData)[1] = "classe"

preProc80 <- preProcess(preProc, method = "pca", thresh = 0.8)
preProc80$numComp  # 12 principal components

# look at results of PCA analysis
# prcomp(~ ., subset(training, TRUE, numericCols), scale = TRUE)

# classification tree
rpartNoPCA <- train(classe ~ ., data = trainData, method = "rpart")

confusionMatrix(rpartNoPCA, newdata=trainData)
# cross-validation
predPartNoPCA <- predict(rpartNoPCA, valData)
confusionMatrix(predPartNoPCA, valData$classe)

rpartPCA80 <- train(classe ~ ., data = trainData, method = "rpart",
                  preProcess = "pca",
                  trControl = trainControl(preProcOptions = list(thresh = 0.80)))

confusionMatrix(rpartPCA80, newdata=trainData)
confusionMatrix(rpartPCA80, newdata=valData)

rpartPCA95 <- train(classe ~ ., data = trainData, method = "rpart",
                    preProcess = "pca",
                    trControl = trainControl(preProcOptions = list(thresh = 0.95)))

confusionMatrix(rpartPCA95, newdata=trainData)

# cross-validation
predPartPCA95 <- predict(rpartPCA95, valData)
confusionMatrix(predPartPCA95, valData$classe)


# random forest
#rfPCA80 <- train(classe ~ ., data = trainData, method = "rf",
#                    preProcess = "pca",
#                    trControl = trainControl(preProcOptions = list(thresh = 0.80)))
# note that this took 1/2 hr or so
# achieves 0.95 accuracy
# try setting number of resamples to 10 (think that default is 30)
rfPCA80 <- train(classe ~ ., data = trainData, method = "rf",
                 preProcess = "pca",
                 trControl = trainControl(preProcOptions = list(thresh = 0.80),
                                          number = 10))
# this runs in about 10 minutes, gets nearly same accuracy as above
confusionMatrix(rfPCA80, newdata=trainData)
rfPCA80

# cross-validation
predRfPCA80 <- predict(rfPCA80, valData)
confusionMatrix(predict(rfPCA80, valData), valData$classe)


# try with PCA thr = 0.95
rfPCA95 <- train(classe ~ ., data = trainData, method = "rf",
                 preProcess = "pca",
                 trControl = trainControl(preProcOptions = list(thresh = 0.95),
                                          number = 10))
# this runs in about 18 minutes, slightly better accuracy 0.968
confusionMatrix(rfPCA95, newdata=trainData)
rfPCA95
varImp(rfPCA95)

# cross-validation
predRfPCA95 <- predict(rfPCA95, valData)
confusionMatrix(predict(rfPCA95, valData), valData$classe)

# should also try linear discriminant analysis
ldaPCA95 <- train(classe ~ ., data = trainData, method = "lda",
                  preProcess = "pca",
                  trControl = trainControl(preProcOptions = list(thresh = 0.95)))
# runs in seconds, but only 53% accuracy
confusionMatrix(ldaPCA95, newdata=trainData)
ldaPCA95
# cross-validation
predLdaPCA95 <- predict(ldaPCA95, valData)
confusionMatrix(predLdaPCA95, valData$classe)

# and quadratic discriminant analysis

qdaPCA95 <- train(classe ~ ., data = trainData, method = "qda",
                  preProcess = "pca",
                  trControl = trainControl(preProcOptions = list(thresh = 0.95)))
# runs in seconds, but only 75% accuracy
confusionMatrix(qdaPCA95, newdata=trainData)
qdaPCA95

predQdaPCA95 <- predict(qdaPCA95, valData)
confusionMatrix(predQdaPCA95, valData$classe)

# also try bagging - use method treebag - again, with PCA95
bagPCA95 <- train(classe ~ ., data = trainData, method = "treebag",
                  preProcess = "pca",
                  trControl = trainControl(preProcOptions = list(thresh = 0.95)))
# runs in 5 - 10 minutes
confusionMatrix(bagPCA95, newdata=trainData)
bagPCA95
# cross-validation
predBagPCA95 <- predict(bagPCA95, valData)
confusionMatrix(predBagPCA95, valData$classe)
# accuracy .961, almost as good as rfPCA95, in 1/2 the time

# dendrogram - don't work on rf, only rpart
#library(rattle)
#fancyRpartPlot(rfPCA95)

plot(rfPCA95, log = "y")
varImpPlot(rfPCA95)

answers <- predict(rfPCA95, dfTest)
