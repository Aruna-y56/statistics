install.packages("arules")
library(arules)
install.packages("arulesViz")
library(arulesViz)

data <- read.csv(file.choose(),1)
View(data)
summary(data)
A_rules <- apriori(as.matrix(data),parameter=list(support=0.002, confidence = 0.5,minlen=1))
A_rules

inspect(A_rules)                 
inspect(head(sort(A_rules, by = "lift")))
plot(A_rules)

head(quality(A_rules))
plot(A_rules, method = "grouped")
plot(A_rules,method = "graph")


#Different support and confidence levels

A_rules1<- apriori(as.matrix(data),parameter=list(support=0.003, confidence = 0.3,minlen=1))
A_rules1

inspect(A_rules1)                 
inspect(head(sort(A_rules1, by = "lift")))
plot(A_rules1)

head(quality(A_rules1))
plot(A_rules1, method = "grouped")
plot(A_rules1,method = "graph")

#Different support and confidence levels

A_rules2<- apriori(as.matrix(data),parameter=list(support=0.005, confidence = 0.5,minlen=1))
A_rules2

inspect(A_rules2)                 
inspect(head(sort(A_rules2, by = "lift")))
plot(A_rules2)

head(quality(A_rules2))
plot(A_rules2, method = "grouped")
plot(A_rules2,method = "graph")
