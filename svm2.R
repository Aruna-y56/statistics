library(kernlab)
library(caret)
library(plyr)
library(ggplot2)
library(e1071)
#Data(Train)
# Exploratory Data Analysis

train_sal<-read.csv("SalaryData_Train.csv") %>% 
nrow() 
train_sal<-read.csv("SalaryData_Train.csv") %>% 
colnames()
train_sal<-read.csv("SalaryData_Train.csv") %>% 
pull(Salary) %>% 
unique() %>% 
length()


str(train_sal)


train_sal$educationno<-as.factor(train_sal$educationno)
class(train_sal)
#Data(Test)
test_sal<-read.csv("SalaryData_Test.csv")%>% 
nrow() 
test_sal<-read.csv("SalaryData_Test.csv") %>% 
colnames()
test_sal<-read.csv("SalaryData_Test.csv") %>% 
pull(Salary) %>% 
unique() %>% 
length()
str(test_sal)
test_sal$educationno<-as.factor(test_sal$educationno)
class(test_sal)
#plot and ggplot
ggplot(data=train_sal,aes(x=train_sal$Salary,y=train_sal$age,fill=train_sal$Salary))+
  geom_boxplot()+
  ggtitle("Box Plot")
plot(train_sal$workclass,train_sal$Salary)
plot(train_sal$education,train_sal$Salary)
plot(train_sal$educationno,train_sal$Salary)
plot(train_sal$maritalstatus,train_sal$Salary)
plot(train_sal$occupation,train_sal$Salary)
plot(train_sal$relationship,train_sal$Salary)
plot(train_sal$race,train_sal$Salary)
plot(train_sal$sex,train$Salary)
ggplot(data=train_sal,aes(x=train_sal$Salary,y=train_sal$capitalgain,fill=train_sal$Salary))+
  geom_boxplot()+
  ggtitle("Box Plot")
ggplot(data=train_sal,aes(x=train_sal$Salary,y=train_sal$capitalloss,fill=train_sal$Salary))+
  geom_boxplot()+
  ggtitle("Box Plot")
ggplot(data=train_sal,aes(x=train_sal$Salary,y=train_sal$hoursperweek,fill=train_sal$Salary))+
  geom_boxplot()+
  ggtitle("Box Plot")
plot(train_sal$native,train_sal$Salary)
#Density Plot
ggplot(data=train_sal,aes(x=train_sal$age,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("Age-Density Plot")
ggplot(data=train_sal,aes(x=train_sal$workclass,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("Workclass Density Plot")
ggplot(data=train_sal,aes(x=train_sal$education,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("education Density Plot")
ggplot(data=train_sal,aes(x=train_sal$maritalstatus,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("maritalstatus Density Plot")
ggplot(data=train_sal,aes(x=train_sal$occupation,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("occupation Density Plot")
ggplot(data=train_sal,aes(x=train_sal$sex,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("sex Density Plot")
ggplot(data=train_sal,aes(x=train_sal$relationship,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("Relationship Density Plot")
ggplot(data=train_sal,aes(x=train_sal$race,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("Race Density Plot")
ggplot(data=train_sal,aes(x=train_sal$capitalgain,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("Capitalgain Density Plot")
ggplot(data=train_sal,aes(x=train_sal$hoursperweek,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("Hoursperweek Density Plot")
ggplot(data=train_sal,aes(x=train_sal$native,fill=train_sal$Salary))+
  geom_density(alpha=0.9,color='Violet')
ggtitle("native Density Plot")
model1<-kvsm(train_sal$Salary~.,
             data=train_sal,kernel="vanilladot")
#support vector machine for salary prediction
Salary_prediction<-predict(model1,test_sal)
table(Salary_prediction,test_sal$Salary)
agreement<-Salary_prediction==test_sal$Salary
table(agreement)
prop.table(table(agreement))
#kernel=rfdot
model_rfdot<-kvsm(train_sal$Salary~.,
                  data=train_sal,
                  kernel="rbfdot")
pred_rfdot<-predict(model_rfdot,newdata=test_sal)
mean(pred_rfdot==test_sal$Salary)
#kernel==vaniladot
model_vanilla<-kvsm(train_sal$Salary~.,
                  data=train_sal,
                  kernel="vanilladot")
pred_vanilla<-predict(model_rfdot,newdata=test_sal)
mean(pred_vanilla==test_sal$Salary)


