
library(rvest)
library(XML)
library(magrittr)
library(syuzhet)
library(tm)
library(wordcloud)


aurl <- "https://www.amazon.in/product-reviews/B07XVMDRZY?ref=ods_ucc_cust_kindle_B07XVMDRZY_nrc_ucc"
amazon_reviews <- NULL
for (i in 1:7){
  murl <- read_html(as.character(paste(aurl,i,sep="=")))  # Use html()
  rev <- murl %>%
    html_nodes(".review-text") %>%
    html_text()
  amazon_reviews <- c(amazon_reviews,rev)
  View (amazon_reviews)
}
write.table(amazon_reviews,"iphone.txt",row.names = F)
write.csv(amazon_reviews, "iphone.csv",row.names = F)


surl_1 <- "https://www.snapdeal.com/product/samsung-galaxy-J3-8gb-4g/676860597612/ratedreviews?page="
surl_2 <- "&sortBy=HELPFUL&ratings=4,5#defRevPDP"
snapdeal_reviews <- NULL
for (i in 1:30){
  surl <- read_html(as.character(paste(surl_1,surl_2,sep=as.character(i))))
  srev <- surl %>%
    html_nodes("#defaultReviewsCard p") %>%
    html_text()
  snapdeal_reviews <- c(snapdeal_reviews,srev)
}
write.csv(snapdeal_reviews, "snapdeal_reviews.csv",row.names = F)
write.table(snapdeal_reviews,"samsung.txt",row.names = FALSE)
getwd()

data <- read.csv(file.choose(), header = TRUE)
reviews <- as.character(data$text)
class(reviews)
s <- get_nrc_sentiment(reviews)
barplot(colSums(s), las = 2.5, col = rainbow(10),
        ylab = 'Count',main= 'Sentiment scores for Amazon reviews')
 
amazon <- read.delim(file.choose(), header = TRUE)
str(amazon)
View(amazon)
corpus <- amazon[-1,]
head(corpus)
class(corpus)
corpus <- Corpus(VectorSource(corpus))
inspect(corpus[1:5])
corpus <- tm_map(corpus,removePunctuation)
corpus <- tm_map(corpus,removeNumbers)
corpus_clean<-tm_map(corpus,stripWhitespace)
cleanset<-tm_map(corpus,removeWords, stopwords('english'))
removeURL <- function(x) gsub('http[[:alnum:]]*','',x)
cleanset <- tm_map(cleanset, content_transformer(removeURL))
cleanset<-tm_map(cleanset,removeWords, c('can','phone'))

cleanset<-tm_map(cleanset,removeWords, c('phones'))
cleanset <- tm_map(cleanset,stripWhitespace)
tdm <- TermDocumentMatrix(cleanset)
tdm
tdm <- as.matrix(tdm)
library(wordcloud)

w <- sort(rowSums(tdm), decreasing = TRUE) # Sort words in decreasing order.
set.seed(123)
wordcloud(words = names(w), freq = w, 
          max.words = 250,random.order = F,
          min.freq =  3, 
          colors = brewer.pal(8, 'Dark2'),
          scale = c(5,0.3),
          rot.per = 0.6)

str(snap)
View(snap)
corpus <- snap[-1,]
head(corpus)
class(corpus)
corpus <- Corpus(VectorSource(corpus))
inspect(corpus[1:5])
corpus <- tm_map(corpus,removePunctuation)
corpus <- tm_map(corpus,removeNumbers)
corpus_clean<-tm_map(corpus,stripWhitespace)
cleanset<-tm_map(corpus,removeWords, stopwords('english'))
removeURL <- function(x) gsub('http[[:alnum:]]*','',x)
cleanset <- tm_map(cleanset, content_transformer(removeURL))
cleanset<-tm_map(cleanset,removeWords, c('can','phone'))

cleanset<-tm_map(cleanset,removeWords, c('phones'))
cleanset <- tm_map(cleanset,stripWhitespace)
tdm <- TermDocumentMatrix(cleanset)
tdm
tdm <- as.matrix(tdm)
library(wordcloud)

w <- sort(rowSums(tdm), decreasing = TRUE) # Sort words in decreasing order.
set.seed(123)
wordcloud(words = names(w), freq = w, 
          max.words = 250,random.order = F,
          min.freq =  3, 
          colors = brewer.pal(8, 'Dark2'),
          scale = c(5,0.3),
          rot.per = 0.6)

