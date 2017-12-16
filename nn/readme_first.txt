*Glove embeddings link: http://nlp.stanford.edu/data/glove.twitter.27B.zip
it includes 200d,100d,50d and 25d embeddings vectors.100d and 50d is already included.200d causes memory problems.

*create "Glove embeddings" and "cleaned twits" file

##############################################################
Basic classifier is simple nn.It uses one vector per tweet by weighted (weights are coming from tf-idf) avaraging words of this tweet.

CNN is cnn nets.Truncute the tweets,one vector per word in tweet. 

lstm is rnn nets.Uses keras tokenizer class, embedding layer and rnn.No code for submission because expected accuracy level is not reached. 

