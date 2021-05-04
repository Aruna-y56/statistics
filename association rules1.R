install.packages("arules")
library(arules)
install.packages("readxl")
library(readxl)

mydata <- read.csv(file.choose(),1)
View(mydata)
ruless1=mydata[,1:5]
ruless2=mydata[,6:15]
View(ruless1)
View(ruless2)

rules22 <- apriori(as.matrix(ruless2),parameter=list(support=0.02,confidence=0.7,minlen=1)) 
inspect(rules22)
inspect(sort(rules22, by="lift"))
inspect(head(sort(rules22,by="lift")))

# Different Ways of Visualizing Rules
plot(rules22)
plot(rules22,method="grouped")
plot(rules22[1:30],method = "graph") # for good visualization try plotting only few rules


#different support and confidence levels plotted

rules21 <- apriori(as.matrix(ruless2),parameter=list(support=0.002,confidence=0.5,minlen=1)) 
inspect(rules21)
inspect(sort(rules21, by="lift"))
inspect(head(sort(rules21,by="lift")))

# Different Ways of Visualizing Rules
plot(rules21)
plot(rules21,method="grouped")
plot(rules21[1:30],method = "graph")

#different support and confidence levels plotted
rules23 <- apriori(as.matrix(ruless2),parameter=list(support=0.003,confidence=0.4,minlen=1)) 
inspect(rules23)
inspect(sort(rules23, by="lift"))
inspect(head(sort(rules23,by="lift")))

# Different Ways of Visualizing Rules
plot(rules23)
plot(rules23,method="grouped")
plot(rules23[1:30],method = "graph")