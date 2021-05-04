install.packages("randomForest")
library(randomForest)
library(caret)
df <- read.csv("Company_Data")
df <- read.csv("Company_Data") %>% 
nrow() 
df <- read.csv("Company_Data") %>% 
colnames()
df <- read.csv("Company_Data") %>% 
pull(ShelveLoc) %>% 
unique() %>% 
length()
View(df)
attach(df)
high <- ifelse(Sales>10,"high","low")
df1 <- cbind(df,high)
View(df1)
df1 <- df1[-1]
#splitting the data into trainning and testing
intrainlocal <- createDataPartition(high,p=.75, list=F)
training1 <- df1[intrainlocal,]
View(training1)
testing1 <- df1[-intrainlocal,]
View(testing1)
fit.forest <- randomForest(high~.,data=training1, na.action=na.roughfix,importance=TRUE, ntree=1000)
# Training accuracy 
mean(training1$high==predict(fit.forest,training1)) #100% accuracy acheived
# Prediction of train data
pred_train <- predict(fit.forest,training1)
# Confusion Matrix
confusionMatrix(training1$high, pred_train)


# Predicting test data 
pred_test <- predict(fit.forest,newdata=testing1)
mean(pred_test==testing1$high) # Accuracy = 94.6 % 


# Confusion Matrix 
library(caret)
confusionMatrix(testing1$high, pred_test)

# Visualization 
plot(fit.forest,lwd=2)
legend("topright", colnames(fit.forest$err.rate),col=1:4,cex=0.8,fill=1:4)

