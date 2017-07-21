#Loading data
setwd("C:/Users/lif8/Documents/GitHub/Titanic/")

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

library(ggplot2)
ggplot(train, aes(x=Age, y=Pclass, color=Survived)) + 
  geom_jitter(position = position_jitter(height = .1)) +
  scale_color_manual(values=c("red", "blue")) + facet_grid(Sex ~ .) +
  ggtitle("Age, Sex, and Class as Survival Factors") + ylab("Pclass")