#Loading data
setwd("C:/Users/lif8/Documents/GitHub/Titanic/")

train <- read.csv(file='train.csv', stringsAsFactors = F)
test  <- read.csv(file='test.csv', stringsAsFactors = F)

#change survived from intger to boolean
train$Survived <- as.logical(train$Survived)