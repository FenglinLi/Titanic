#import library
library(plyr)

#Loading data
#setwd("C:/Users/lif8/Documents/GitHub/Titanic/")
setwd("C:/Users/lfl1001/Documents/GitHub/Titanic/")

train <- read.csv(file='train.csv', stringsAsFactors = F)
test  <- read.csv(file='test.csv', stringsAsFactors = F)

#change survived from intger to boolean
train$Survived <- as.logical(train$Survived)
levels(train$Survived) <- c("Not survived", "Survived")
train$FamilySize <- train$SibSp + train$Parch + 1
test$FamilySize <- test$SibSp + test$Parch + 1
#set factor levels for 3=Pclass, 5=Sex, 12=Embarked
for(i in c(3,5,12)) {
  train[,i] <- as.factor(train[,i])
}

#visualing the correclation between PClass and Sex
library(ggplot2)
ggplot(train, aes(x=Age, y=Pclass, color=Survived)) + 
  geom_jitter(position = position_jitter(height = .1)) +
  scale_color_manual(values=c("red", "blue")) + facet_grid(Sex ~ .) +
  ggtitle("Age, Sex, and Class as Survival Factors") + ylab("Pclass")

#Create adjusted family size variable for people sharing cabins but not registered as family members
cabins <- train$Cabin 
n_occur <- data.frame(table(Var1=cabins))
n_occur <- subset(n_occur, nchar(as.character(Var1)) > 1)
sharedCabins <- n_occur$Var1[n_occur$Freq > 1]
train$FamilySizeAdj <- train$FamilySize
print(table(train$FamilySize))

sharedInd <- train$FamilySizeAdj == 1 & train$Cabin %in% sharedCabins
train$FamilySizeAdj[sharedInd] <- 2
rowCount <- sum(sharedInd)
print(c("adjusted rows", rowCount))

print(table(train$FamilySizeAdj))


#break up training set into subset sub_train & sub_test
library(caret)
set.seed(820)
inTrainingSet <- createDataPartition(train$Survived, p = 0.5, list=FALSE)
sub_train <- train[inTrainingSet,]
sub_test <- train[-inTrainingSet,]

#two functions for model fit
modelaccuracy <- function(sub_test, rpred) {
  result_1 <- sub_test$Survived == rpred
  sum(result_1) / length(rpred)
}

checkaccuracy <- function(accuracy) {
  if (accuracy > bestaccuracy) {
    bestaccuracy <- accuracy
    assign("bestaccuracy", accuracy, envir = .GlobalEnv)
    label <- 'better'
  } else if (accuracy < bestaccuracy) {
    label <- 'worse'
  } else {
    label <- 'no change'
  }
  label
}

#first try
library(rpart)
fol <- formula(Survived ~ Age + Sex) 
rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")
accuracy <- modelaccuracy(sub_test, rpred)
bestaccuracy <- accuracy # init base accuracy
print(c("accuracy1", accuracy)) #0.8449        

#2nd try adding Pclass
fol <- formula(Survived ~ Age + Sex + Pclass)
rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")
accuracy <- modelaccuracy(sub_test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy2", accuracy, accuracyLabel)) #0.8382 worse

#3rd try - adding Fare
fol <- formula(Survived ~ Age + Sex + Fare)                 
rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")
accuracy <- modelaccuracy(sub_test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy3", accuracy, accuracyLabel)) #0.8067 worse

#4th try  adding Plcass & Fare
fol <- formula(Survived ~ Age + Sex + Fare + Pcalss)                 
rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")
accuracy <- modelaccuracy(sub_test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy4", accuracy, accuracyLabel)) #0.8067 worse


#5th try - adding SibSp + Parch + Pclass + Fare
fol <- formula(Survived ~ Age + Sex + Pclass + Fare + SibSp + Parch) # 0.838 worse
rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")
print(rmodel)

accuracy <- modelaccuracy(sub_test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy5", accuracy, accuracyLabel)) #0.8382 worse


#check deck letter
sub_train$Deck <- substr(sub_train$Cabin,1,1)
sub_train$Deck[sub_train$Deck==''] = NA
sub_test$Deck <- substr(sub_test$Cabin,1,1)
sub_test$Deck[sub_test$Deck==''] = NA

sub_train$Deck <- as.factor(sub_train$Deck)
sub_test$Deck <- as.factor(sub_test$Deck)

c <- union(levels(sub_train$Deck), levels(sub_test$Deck))
levels(sub_test$Deck) <- c
levels(sub_train$Deck) <- c

fol <- formula(Survived ~ Age + Sex + Pclass + SibSp + Parch + Fare + Deck) 
rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")

accuracy <- modelaccuracy(sub_test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy6", accuracy, accuracyLabel)) #0.8067 worse        


#check FamilySize feature
fol <- formula(Survived ~ Age + Sex + Pclass + FamilySize)           
rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")
print(rmodel)
accuracy <- modelaccuracy(sub_test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy7", accuracy, accuracyLabel)) #0.8719 better


#Visualize
p <- ggplot(aes(x=Pclass,y=factor(FamilySize),color=Survived),data=train) + 
  geom_jitter() + facet_grid(Sex ~ .)
p + ggtitle("Large Family Size >= 5 more likely to not survive") + theme_bw() + 
  geom_hline(yintercept=5) + ylab("Family Size")

mosaicplot(table(FamilySize=train$FamilySize, Survived=train$Survived),
           main="Passenger Survival by Family Size",
           color=c("#fb8072", "#8dd3c7"), cex.axis=1.2)


test$Sex <- as.factor(test$Sex)
test$Pclass <- as.factor(test$Pclass)

fol <- formula(Survived ~ Age + Sex + Pclass + FamilySize)
model <- rpart(fol, method="class", data=train)

library(rpart.plot) 
rpart.plot(model,branch=0,branch.type=2,type=1,extra=102,shadow.col="pink",box.col="gray",split.col="magenta",
           main="Decision tree for model")


#check removing Pclass
fol <- formula(Survived ~ Sex + Age + FamilySize)                   
rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")
accuracy <- modelaccuracy(sub_test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy8", accuracy, accuracyLabel)) #0.8539 worse                   

#cheking traveling alone boolean
fol <- formula(Survived ~ Age + Sex + Pclass + TravelAlone)         
sub_train$TravelAlone <- sub_train$FamilySize == 1
sub_test$TravelAlone <- sub_test$FamilySize == 1

rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")
accuracy <- modelaccuracy(sub_test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy10", accuracy, accuracyLabel)) #0.8539 worse

#checking adding Embarked
fol <- formula(Survived ~ Age + Sex + Pclass + FamilySize + Embarked)   
rmodel <- rpart(fol, method="class", data=sub_train)
rpred <- predict(rmodel, newdata=sub_test, type="class")
accuracy <- modelaccuracy(sub_test, rpred)
accuracyLabel <- checkaccuracy(accuracy)
print(c("accuracy11", accuracy, accuracyLabel)) #0.8584 worse                       

#Print the best
print (c("best accuracy", bestaccuracy))

#Predict for subbmission
fol <- formula(Survived ~ Age + Sex + Pclass + FamilySize)           
rmodel <- rpart(fol, method="class", data=train)
rpred <- predict(rmodel, newdata=test, type='class')

solution <- data.frame(PassengerID = test$PassengerId, Survived = rpred)

solution$Survived =ifelse(solution$Survived=="TRUE",1,0)

write.csv(solution, file = 'r2_submission.csv', row.names = F)