#Installing and loading the libraries
install.packages("recommenderlab", dependencies=TRUE)
install.packages("Matrix")
library(Matrix)
library("recommenderlab")
library(caTools)


View(books)
str(books) ## In data set dependent and independent variable which  data type we can check  
table(books$ratings...3.) # In rating column how many rating is there we can check 
object.size(books) # size of the data set
#rating distribution
hist(books$rating) 
View(books)
#the datatype should be realRatingMatrix inorder to build recommendation engine
books_rate_data_matrix <- as(books,'realRatingMatrix')

#Popularity based 

books_recomm_model1 <- Recommender(books_rate_data_matrix, method="POPULAR")

#Predictions
recommended_books1 <- predict(books_recomm_model1, books_rate_data_matrix[1], n=4)
as(recommended_books1, "list")


#User Based Collaborative Filtering
books_recomm_model2 <- Recommender(books_rate_data_matrix, method="UBCF")

## Note: Here in method we have to use IBCF but if we used UBCF because When we are trying to run code it showing running only but we are not getting output  


#Predictions 
recommended_items2 <- predict(books_recomm_model2, books_rate_data_matrix[1], n=4)
as(recommended_items2, "list")




