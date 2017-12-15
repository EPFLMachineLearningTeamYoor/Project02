import string
import re


# before running this script , please run the datacleaning script and set the folder to read from to train_pos_full and train_neg_full to generate the vocabullary 
# for the respective files


def loadText(fileName):

	with open(fileName) as f:
		content = f.readlines()
		content = [x.strip() for x in content] 
		return content	



def stripWords(tweets,vocab):


	filtered_tweets=[]
	for tweet in tweets:
	
		tweet_words = tweet.split()
		resultwords  = [word for word in tweet_words if word.lower() in vocab]
		result = ' '.join(resultwords)
		filtered_tweets.append(result)

	return filtered_tweets


def writeResult(tweets,outputFile):

	result = open(outputFile, 'w')
	for item in vocab:
	 	result.write("%s\n" % item)
	

positive_tweets = loadText('../data/train_pos_full.txt')
positive_vocab  = set(loadText('../vocabulary/train_pos_full_vocab.txt')) # change vocab based on previous folder
content = stripWords(positive_tweets,positive_vocab)
writeResult(content,'../cleandata/train_pos_clean.txt')
