library(readr)
install.packages("e1071")
library(e1071)
library(caret)

train_sal <- read.csv("SalaryData_Train")
#Exploratory Data Analysis
train_sal<-read.csv("SalaryData_Train.csv") %>% 
nrow() 
train_sal<-read.csv("SalaryData_Train.csv") %>% 
colnames()
train_sal<-read.csv("SalaryData_Train.csv") %>% 
pull(Salary) %>% 
unique() %>% 
length()
str(train_sal)
train_sal$educationno <- as.factor(train_sal$educationno)

test_sal <- read.csv("SalaryData_Test")
#Exploratory Data Analysis
test_sal <- read.csv("SalaryData_Test") %>% 
nrow() 
test_sal <- read.csv("SalaryData_Test") %>% 
colnames()
test_sal <- read.csv("SalaryData_Test") %>% 
pull(Salary) %>% 
unique() %>% 
length()
str(test_sal)

test_sal$educationno <- as.factor(test_sal$educationno)

Model <- naiveBayes(train_sal$Salary ~ ., data = train_sal)
Model
Model_pred <- predict(Model,test_sal)
mean(Model_pred==test_sal$Salary)confusionMatrix(Model_pred,test_sal$Salary)
