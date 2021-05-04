install.packages("randomForest")
library(randomForest)
library(caret)
df <- read.csv("Fraud_check")
df <- read.csv("Fraud_check") %>% 
nrow() 
df <- read.csv("Fraud_check") %>% 
colnames()
df <- read.csv("Fraud_check") %>% 
pull(Marital.Status) %>% 
unique() %>% 
length()


View(df)
attach(df)
target <- ifelse(Taxable.Income<=30000,"Risky","Good")
df1 <- cbind(df,target)
View(df1)
df1 <- df1[-3]
intrain <- createDataPartition(df1$target,p=.70, list=F)
training1 <- df1[intrain,]
View(training1)
testing1 <- df1[-intrain,]
View(testing1)
fit.forest <- randomForest(target~.,data=training1, na.action=na.roughfix,importance=TRUE, ntree=2000)
# Training accuracy 
mean(training1$target==predict(fit.forest,training1)) 
# Prediction of train data
pred_train <- predict(fit.forest,training1)
# Confusion Matrix
confusionMatrix(training1$target, pred_train)


# Predicting test data 
pred_test <- predict(fit.forest,newdata=testing1)
mean(pred_test==testing1$target) 


# Confusion Matrix 
library(caret)
confusionMatrix(testing1$target, pred_test)

# Visualization 
plot(fit.forest,lwd=2)
legend("topright", colnames(fit.forest$err.rate),col=1:4,cex=0.8,fill=1:4)
