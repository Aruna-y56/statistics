wine <- read.csv(file.choose())
View(wine)
pcaObj<-princomp(wine[-1], cor = TRUE, scores = TRUE, covmat = NULL)
summary(pcaObj)
loadings(pcaObj)

plot(pcaObj)
pcaObj$scores
pcaObj$scores[,1:3]
wine<-cbind(wine,pcaObj$scores[,1:3])
View(wine)

clus_data<-wine[,15:17]
View(clus_data)
norm_clus<-scale(clus_data) # Scale function is used to normalize data
dist1<-dist(norm_clus,method = "euclidean") # method for finding the distance
# here I am considering Euclidean distance
fit1<-hclust(dist1,method="complete")
plot(fit1)
groups<-cutree(fit1,3)

membership_1<-as.matrix(groups) # cluster numbering 

View(membership_1)
final1<-cbind(membership_1,wine) # binding column wise with orginal data
View(final1)
normalized_data <- scale(wine[,15:17])
wss = (nrow(normalized_data)-1)*sum (apply(normalized_data, 2, var))
# Determine number of clusters by scree-plot 
for (i in 1:5) wss[i] = sum(kmeans(normalized_data, centers=i)$withinss)
plot(1:5, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")   # Look for an "elbow" in the scree plot #
title(sub = "K-Means Clustering Scree-Plot")
fit <- kmeans(normalized_data, 3) # 5 cluster solution
final2<- data.frame(wine, fit$cluster) # append cluster membership
View(final2)
final3 <- final2[,c(ncol(final2),1:(ncol(final2)-1))]
View(final3)
aggregate(wine, by=list(fit$cluster), FUN=mean)


############################Clustering on the entire Data set###################
# Considering the entire data set for  clustering
# Hierarchial Clustering
# preparing data for clustering 
mydata <- wine[-1]
clus_data1<-mydata

# Normalizing the data 
norm_clus1<-scale(clus_data1) # Scale function is used to normalize data
dist11<-dist(norm_clus1,method = "euclidean") # method for finding the distance
# here I am considering Euclidean distance

# Clustering the data using hclust function --> Hierarchical
fit11<-hclust(dist11,method="complete") # method here is complete linkage

plot(fit11) # Displaying Dendrogram

rect.hclust(fit11, k=7, border="red")
groups1<-cutree(fit11,7) # Cutting the dendrogram for 7 clusters

membership_11<-as.matrix(groups1) # cluster numbering 

View(membership_11)

final11<-cbind(membership_11,wine) # binding column wise with orginal data
View(final11)

