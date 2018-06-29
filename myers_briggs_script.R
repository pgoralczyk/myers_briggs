library(ggplot2)
library(ggrepel)
library(reshape2)
library(tidytext)
library(dplyr)

dmb <- read.csv("mbti_1.csv", stringsAsFactors = FALSE)


remove_numbers <- function(d) {
  d <- mutate(d,
              is_number = ifelse(is.na(as.numeric(word)), 0, 1))
  d <- filter(d, is_number == FALSE) %>% select(-is_number)
}


#create an ID for each person
dmb$id <- 1:nrow(dmb)

# dmb <- mutate(dmb, 
#               IE = substr(type, 1, 1),
#               TF = substr(type, 3, 3))

#transforming data to long format (grain: ID, word). Dropping stop words.
data(stop_words)
words_to_drop = data_frame(word = c('â', 'ã', '1w2', '4w5', '5w4', '5w6', '6w7',
                                    '9w1', '6w5', '9w1', 'www.youtube.com', 'http', 'https', 
                                    'infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 
                                    'infp', 'enfp', 'isfp', 'istp', 'isfj', 'istj', 
                                    'estp', 'esfp', 'estj', 'esfj' ))



dmb_long <- dmb %>%
  unnest_tokens(word, posts) %>%
  anti_join(stop_words) %>%
  anti_join(words_to_drop)

dmb_long <- remove_numbers(dmb_long) %>%
  filter(nchar(word) > 3)



#lets restructure data a bit:
#we'll create a wide data frame, one row per person
#with a long list of column indicating whether a person used this word or not
dmb_wide <- group_by(dmb_long, word) %>%
  mutate(n = n_distinct(id),
         one = 1) %>%
  filter(n > 100) %>%
  unique() %>%
  select(id_ = id, type_ = type, word, one) %>%
  dcast(id_ + type_ ~ word, value.var = 'one')


dmb_wide[is.na(dmb_wide)] <- 0

head(dmb_wide[, 1:10])

#

#let's start with a simpler task.

dmb_wide <- mutate(dmb_wide,
                   type_ie = ifelse(substr(type_, 1, 1) == 'E', 1, 0))

m1 <- glm(type_ie ~ .,
    data = dmb_wide[, 3:ncol(dmb_wide)],
    family = binomial())

ncol(dmb_wide)

#we'll try to separate Introverts from Extraverts from Introverts




##
dmb_word_rank <- dmb_long %>%
  group_by(TF, word) %>%
  summarise(word_count = n()) %>%
  group_by(TF) %>%
  mutate(freq = word_count / sum(word_count)) %>%
  filter(word_count > 1000) %>%
  arrange(TF, desc(freq)) %>%
  mutate(word_rank = seq_along(word_count),
         freq_st = (freq - mean(freq))/sd(freq)) %>%
  ungroup()

top_words <- filter(dmb_word_rank, word_rank <= 20)$word %>% unique()

filter(dmb_word_rank, word %in% top_words) %>%
  dcast(word ~ TF, value.var = 'freq_st') %>%
  ggplot(aes(x = T,
             y = F)) +
  geom_text_repel(aes(label = word), size = 3) +
  geom_abline(intercept = 0, slope = 1)

dmb_word_rank %>%
  dcast(word ~ TF, value.var = 'freq') %>%
  mutate(freq_diff = T - F ) %>%
  arrange(desc(abs(freq_diff))) %>%
  head(20)

?seq_along
