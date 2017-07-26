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

library(rpart)
fol <- formula(Survived ~ Age + Sex) 
rmodel <- rpart(fol, method="class", data=train)














