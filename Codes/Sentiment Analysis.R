setwd("C:/Users/V/Downloads")
memory.limit(size = 500000)

library(tm)
#df = read.delim("amazon_reviews.txt",header = TRUE, sep = "\t", dec = ".", quote = "")

df = read.csv("amazon_reviews.csv")

names(df) = "reviews"

df = df[!grepl("#####", df$reviews),] # Remove #### rows
df = df[!(df$reviews ==""), ] # Remove Blank Rows

df1 = df

df1 = as.data.frame(df[, c(1)])
  
names(df1) = "reviews"

reviews = list(df1$reviews)

df1$reviews = as.character(df1$reviews)

# Text Pre-Processing

# remove at people
df1$reviews = gsub("@\\w+", "", df1$reviews)

# remove punctuation
df1$reviews = gsub("[[:punct:]]", "", df1$reviews)

# remove numbers
df1$reviews = gsub("[[:digit:]]", "", df1$reviews)

# remove html links
df1$reviews = gsub("http\\w+", "", df1$reviews)

# remove unnecessary spaces
df1$reviews = gsub("[ \t]{2,}", "", df1$reviews)
df1$reviews = gsub("^\\s+|\\s+$", "", df1$reviews)
df1$reviews = gsub("'-'","",df1$reviews)

#convert to lower case
df1$reviews = tolower(df1$reviews)

#### Sentiment Analysis
# library(Rstem)
# library(sentiment)

# install.packages(Rstem, repos = "http://www.omegahat.net/R")

library(sentimentr)

# classify sentiment
senti = sentiment(df1$reviews, by = NULL)

str(senti)

# classify emotion
emo = emotion(df1$reviews)
str(emo)

emo_by = emotion_by(df1$reviews)
str(emo_by)

senti$ave_sentiment = round(senti$ave_sentiment, digits=2)

summary(senti$ave_sentiment)

# Polarity Class

senti$polarity[senti$sentiment == 0] = "Neutral"
 
senti$polarity[(senti$sentiment < -1)] = "Very Negative"

senti$polarity[senti$sentiment > 1] = "Very Positive"

senti$polarity[(senti$sentiment < 0) & (senti$sentiment >= -1)] = "Negative"

senti$polarity[(senti$sentiment > 0) & (senti$sentiment <= 1)] = "Positive"

library(ggplot2)

# plot distribution of sentiment scores
qplot(senti$sentiment, geom="histogram",binwidth=0.1, main="Review Sentiment Histogram", 
      colour = I("red"), xlab ="Sentiment Score", ylab ="Number of Reviews")

# plot distribution of polarity
ggplot(senti, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill = polarity)) +
  labs(x="Polarity Categories", y="Number of Reviews")

# plot distribution of polarity
ggplot(senti, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill = polarity)) +
  labs(x="Polarity Categories", y="Number of Reviews")

#Separate the text by emotions and visualize the words 
#with a comparison cloud separating text by emotion
emos = levels(factor(senti$polarity))
nemo = length(emos)
emo.docs = rep("", nemo)
for (i in 1:nemo)
{
  tmp = reviews[emotion == emos[i]]
  emo.docs[i] = paste(tmp, collapse=" ")
}
  

