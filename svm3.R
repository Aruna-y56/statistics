ForestFires <- read.csv("Forestfires.csv")
#Exploratory Data Analysis
ForestFires <- read.csv("Forestfires.csv") %>% 
nrow() 
ForestFires <- read.csv("Forestfires.csv") %>% 
colnames()
ForestFires <- read.csv("Forestfires.csv") %>% 
pull(month) %>% 
unique() %>% 
length()
ForestFires <- read.csv("Forestfires.csv") %>% 
pull(day) %>% 
unique() %>% 
length()



View(ForestFires)
FF <- ForestFires[c(-1,-2)]
View(FF)
str(FF)

#Converting to factor
FF$size_category = factor(FF$size_category, levels = c('small', 'large'), labels = c(0,1))
FF[10:28] <- lapply(FF[10:28], factor)
str(FF)

#Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(FF$size_category, SplitRatio = 0.75)
train_set = subset(FF, split == TRUE)
test_set = subset(FF, split == FALSE)

# Feature Scaling
train_set[1:9] = scale(train_set[1:9])
test_set[1:9] = scale(test_set[1:9])
View(test_set)
#Building Model
library(kernlab)
model_vanilladot <- ksvm(size_category ~ . , data= train_set, kernal= 'vanilladot')
model_vanilladot
pred_vanilla <- predict(model_vanilladot, test_set[-29])
pred_vanilla
CM = table(test_set$size_category, pred_vanilla);CM
Accuracy <- sum(diag(CM))/sum(CM);Accuracy

#Model splinedot
model_splinedot <- ksvm(size_category ~ . , data= train_set, kernal= 'splinedot')
model_splinedot
pred_splinedot <- predict(model_splinedot, test_set[-29])
pred_splinedot
CM = table(test_set$size_category, pred_splinedot);CM
Accuracy <- sum(diag(CM))/sum(CM);Accuracy

#Model Polydot
model_polydot <- ksvm(size_category ~ . , data= train_set, kernal= 'polydot')
model_polydot
pred_polydot <- predict(model_polydot, test_set[-29])
pred_polydot
CM = table(test_set$size_category, pred_polydot);CM
Accuracy <- sum(diag(CM))/sum(CM);Accuracy

#Model rbf
model_rbfdot <- ksvm(size_category ~ . , data= train_set, kernal= 'rbfdot')
model_rbfdot
pred_rbfdot <- predict(model_rbfdot, test_set[-29])
pred_rbfdot
CM = table(test_set$size_category, pred_rbfdot);CM
Accuracy <- sum(diag(CM))/sum(CM);Accuracy

#Model rbf
model_stringdot <- ksvm(size_category ~ . , data= train_set, kernal= 'stringdot')
model_stringdot
pred_stringdot <- predict(model_stringdot, test_set[-29])
pred_stringdot
CM = table(test_set$size_category, pred_stringdot);CM
Accuracy <- sum(diag(CM))/sum(CM);Accuracy

#Model rbf
model_besseldot <- ksvm(size_category ~ . , data= train_set, kernal= 'besseldot')
model_besseldot
pred_besseldot <- predict(model_besseldot, test_set[-29])
pred_besseldot
CM = table(test_set$size_category, pred_besseldot);CM
Accuracy <- sum(diag(CM))/sum(CM);Accuracy

#Model rbf
model_laplacedot <- ksvm(size_category ~ . , data= train_set, kernal= 'laplacedot')
model_laplacedot
pred_laplacedot <- predict(model_laplacedot, test_set[-29])
pred_laplacedot
CM = table(test_set$size_category, pred_laplacedot);CM
Accuracy <- sum(diag(CM))/sum(CM);Accuracy

#Most of the model gives same result with 80% accuracy
plot(model_vanilladot)
