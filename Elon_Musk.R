library(tidyverse)
library(wordcloud)
library(RColorBrewer)
library(qdap)
library(tm)
library(gridExtra)
library(dendextend)
library(ggthemes)
library(RWeka)
library(tidytext)
library(textdata)
library(lubridate)
library(anytime)

theme_func <- function() {
  theme_minimal() +
    theme(
      text = element_text(family = "serif", color = "gray25"),
      plot.subtitle = element_text(size = 12,hjust = 0.5,color = "gray45"),
      plot.caption = element_text(color = "gray30"),
      plot.background = element_rect(fill = "gray95"),
      plot.margin = unit(c(5, 10, 5, 10), units = "mm"),
      plot.title = element_text(hjust = 0.5),
      strip.text = element_text(color = "white")
    )
}

data <- read_csv("elonmusk_tweets.csv")
str(data)
data<-data[!data$text=='',]
data$text[1:5]
clean_corpus <- function(corpus) {
  
  corpus <- tm_map(corpus, content_transformer(tolower))
  
  removeURL <- function(x) gsub("http[^[:space:]]*", "", x) 
  corpus <- tm_map(corpus, content_transformer(removeURL)) 
   
  removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x) 
  corpus <- tm_map(corpus, content_transformer(removeNumPunct))
  
  corpus <- tm_map(corpus, stripWhitespace)
  
  corpus <- tm_map(corpus,removeWords, words = c(stopwords("en"),"brt","amp","s"))
  
  corpus <- tm_map(corpus, stemDocument)
  return(corpus)
}
term_frequency <- rowSums(tweets_m)


term_frequency <- sort(term_frequency,decreasing = TRUE)
barplot(term_frequency[1:15], col = "tan", las = 2)
tdm2 <- removeSparseTerms(tweets_tdm, sparse=0.96)
m2 <- as.matrix(tdm2)
# cluster terms
distMatrix <- dist(scale(m2))
hc <- hclust(distMatrix)

hcd <- as.dendrogram(hc)


hcd_colored <- branches_attr_by_labels(hcd, c("teslamotor", "car","launch","rocket"), "red")

plot(hcd_colored, main = "Cluster Dendrogram")
findAssocs(tweets_tdm, "stock", 0.4)
findAssocs(tweets_tdm, "spacex", 0.2)
findAssocs(tweets_tdm, "falcon", 0.2)
tokenizer <- function(x) {
  NGramTokenizer(x, Weka_control(min = 2, max = 2))
}

# Create unigram_dtm
unigram_dtm <- DocumentTermMatrix(clean_tweets)

# Create bigram_dtm
bigram_dtm <- DocumentTermMatrix(
  clean_tweets, 
  control = list(tokenize = tokenizer)
)

# Create bigram_dtm_m
bigram_dtm_m <- as.matrix(bigram_dtm)

# Create freq
freq <- colSums(bigram_dtm_m)

# Create bi_words
bi_words <- names(freq)


# Plot a wordcloud
par(bg="black") 
wordcloud(bi_words, freq,max.words=500,random.order=FALSE,c(4,0.4), col=terrain.colors(length(bi_words) , alpha=0.9) , rot.per=0.3)
data_tibble <- data %>%
  unnest_tokens(output = "words", input = text, token = "words")
#Remove stop words from tibble
tweets_tibble_clean <- data_tibble %>%
  anti_join(stop_words, by=c("words"="word"))

word_freq <- tweets_tibble_clean %>% 
  # Inner join to bing lexicon by words = word
  inner_join(get_sentiments("bing"), by = c("words" = "word")) %>% 
  
  # Count by words and sentiment, weighted by count
  count(words, sentiment) %>%
  # Spread sentiment, using n as values
  spread(sentiment, n, fill = 0) %>%
  
  # Mutate to add a polarity column
  mutate(polarity = positive - negative)%>%
  filter(abs(polarity) >= 9) %>%
  mutate(
    pos_or_neg = ifelse(polarity > 0, "positive", "negative")
  )
ggplot(word_freq, aes(x = reorder(words, polarity),y =  polarity, fill = pos_or_neg)) +
  geom_col() + 
  labs(
    x = "words",
    title = "Sentiment Word Frequency"
  )+
  theme_func()+
  
  # Rotate text and vertically justify
  theme(axis.text.x = element_text(angle = 55))
data %>%
  unnest_tokens(output = "words", input = text, token = "words")%>%
  anti_join(stop_words, by=c("words"="word"))%>%
  count(words) %>%
  inner_join(get_sentiments("nrc"), by = c("words"="word")) %>%
  group_by(sentiment) %>%
  filter(!grepl("positive|negative", sentiment)) %>%
  top_n(5, n) %>%
  ungroup() %>%
  mutate(word = reorder(words, n)) %>%
  ggplot(aes(word, n,fill=sentiment)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ sentiment, scales = "free") +
  theme_func()+
  labs(y = "count",title = "Top Words under Each Sentiment")

