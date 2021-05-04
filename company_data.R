library(C50)
install.packages("caret")
library(caret)
install.packages("tree")
library(tree)
install.packages("party")
library(party)
mydata1 <- read.csv("Company_Data")
mydata1 <- read.csv("Company_Data") %>% 
nrow() 
mydata1 <- read.csv("Company_Data") %>% 
colnames()
mydata1 <- read.csv("Company_Data") %>% 
pull(ShelveLoc) %>% 
unique() %>% 
length()
View(mydata1)
max(mydata1$Sales, na.rm = TRUE)
sales1 <- cut(mydata1$Sales, breaks=c(0, 10,Inf),labels=paste("sales", 1:2, sep=""),ordered_result = TRUE)
str(sales1)
View(mydata1)
mydata1 <- cbind(mydata1,sales1)
View(mydata1)
mydata1=mydata1[-1]
mydata1<-mydata1[complete.cases(mydata1),]
# Data partion for model building and testing
inTraininglocal1 <- createDataPartition(mydata1$sales1,p=.75, list=F)
training1 <- mydata1[inTraininglocal1,]
View(training1)
testing1 <- mydata1[-inTraininglocal1,]
View(testing1)

model <- C5.0(training1$sales1~.,data = training1,trails = 1000)
summary(model)
pred1 <- predict.C5.0(model,testing1[,-11])
a <- table(testing1$sales1,pred1)
View(a)
sum(diag(a)/sum(a))
plot(model)
model1 <- ctree(training1$sales1~.,data = training1)
plot(model1)
