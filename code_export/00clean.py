import re
import random
import enchant
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from nltk.metrics import edit_distance
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import multiprocessing
from tqdm import tqdm

# LOADING DATASETS
pos_file, neg_file = 'data/train_pos_full.txt', 'data/train_neg_full.txt'

no_tqdm = lambda x : x

# a method that takes raw text and generates raw text out of it/ op parameter is used to either 
#to keep redundant elements or remove them
def tokenize_tweets(raw_tweets,op,tqdm_=no_tqdm):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tknzr.tokenize(raw_tweets)
    words = [w.lower() for w in tqdm_(tokens)]
    if op == "vocab":
        vocab = set(words)
    elif op == "raw":
        vocab = words
    return vocab

# a method that reads a file and removes all non-ascii characters from it 
# returns raw_text,list [tweets]
def loadFile(filename):
    f = open(filename)
    raw_tweets = f.read()
    remove_nonunicode = re.sub(r'[^\x00-\x7F]',' ', raw_tweets)
    content = remove_nonunicode.splitlines()
    return remove_nonunicode,content

# STEMMING WORDS
#Word stemming is the process to find the stem of a word. To be more concrete as an example consider the following words eating,eats both are variations of the word eat. In our approach we know that tweets may include many repeated characters within one word, where people stress on their emotions. Thus we convert these words to words having only 2 repetions of these characters and we check if the reduced version is valid (found in dictionary)

# a method that finds a group of repeated letters and replaces them by 2 instances of that letter
def repl(matchObj):	
    char = matchObj.group(1)
    return "%s%s" % (char, char)

# checks if the word has repeated characters if true then we invoke the previous method
def replaceRepeatingCharacters(tweets):
    pattern = re.compile(r"(\w)\1+")
    corrected_words = pattern.sub(repl,tweets)
    return corrected_words

# check if current token is already correct(found in dictionary) 
#if not then call above function
# if word has changed => there os a sequence of repeating characters and hence we reccur
# otherwise the word is returned
def certifyToken(token):
    if wordnet.synsets(token): #if dictionary knows the token we return it
        return token
    reduced_token = replaceRepeatingCharacters(token) # remove repeating characters if any
    if reduced_token != token: # if the reduced word is different we may remove further
        return certifyToken(reduced_token)
    else:
        return reduced_token

# CORRECTING MISSPELLED WORDS

#In this block, we focus on correcting misspelled words in tweets. This is achieved by using an english dictionary provided by enchant library, which checks if a word is correct or wrong.If wrong it provides us with suggestions for relevant replacement. We then compute the levensteihn distance between our token and the provided suggestion.
#The first method constructs a python dictionary mapping every levensteihn distance to a count denoting number of suggestions that lie that far from the word. 
#The second method replaces all wrong words by their respective suggestions based on a threshold we set.

# a method that checks wether the token is known by the dictionary provided by enchant library
# if yes returns it otherwise the dictionary provides a sey of suggestions 
# we compute levensthein distance to with the first suggestion if the distance is less than a certain threshold
# then we return the suggestion otherwise we keep the token as is
def correct_misspelled_word(word_dict,token):
    levenshtein_thresh = 5
    if word_dict.check(token):
        return token
    else:
        suggestions = word_dict.suggest(token)
        if suggestions:
            levenshtein_dist = edit_distance(token, suggestions[0])
            if levenshtein_dist <=  levenshtein_thresh:
                return suggestions[0]
            else:
                return token
                
        else:
            return token 

# CORRECTING AND STEMMING ALL TEXT

# this method is used to construct our vocabullary that we use to build our occurrence matrix
# it finds the stem of every token
def stemm_all_tokens(tokens,tqdm_=no_tqdm):
    correct_tweets = [certifyToken(token) for token in tqdm_(tokens)]
    return correct_tweets

word_dict = enchant.Dict("en_US")
def correct_global(token):
    global word_dict
    return correct_misspelled_word(word_dict,token)

# this method is used to construct our vocabullary that we use to build our occurrence matrix
# it corrects every misspelled words
def correct_all_tokens(word_dict_,tokens, pool):
    return list(tqdm(pool.imap(correct_global, tokens), total=len(tokens)))
    #r = pool.map(correct_global, tokens)
    #return r

# this method uses cache to correct tweets
# voc2corrvoc: token 'beingcreative' -> 'being creative'
def correct_tweet_cached(text, voc2corrvoc):
    tokens = tokenize_tweets(text, "raw")
    return ' '.join([voc2corrvoc[t] if t in voc2corrvoc else t for t in tokens]).lower()

# this method uses a map to either stem or correct every token in a tweet
def correct_tweet(text,op,tqdm_=no_tqdm):
    tokens = list(tokenize_tweets(text,"raw"))
    word_dict = enchant.Dict("en_US")
    if op == "spelling":
        correct_tokens = lambda token : correct_misspelled_word(word_dict,token)  if token.isalpha() else token
    elif op =="stemming":
        correct_tokens = lambda token : certifyToken(token)  if token.isalpha() else token
    corrected_tokens = [correct_tokens(token) for token in tqdm_(tokens)]
    corrected_tweet = ' '.join(corrected_tokens)
    return corrected_tweet  

# this method takes a list of tweets and either corrects or stems the tweets by using a map function
def correct_all_tweets(tweets,op, tqdm_=no_tqdm):
    correct_tweets = [correct_tweet(x,op) for x in tqdm_(tweets)]
    return correct_tweets

# this method takes a list of tweets and corrects them using cache
# voc2corrvoc: token 'beingcreative' -> 'being creative'
def correct_all_tweets_cached(tweets, voc2corrvoc):
    correct_tweets = [correct_tweet_cached(x, voc2corrvoc) for x in tqdm(tweets)]
    return correct_tweets

# STRIP INITIAL CHARACTERS

def strip_token(token):
    contains_chars = re.search('[a-zA-Z]', token)
    if contains_chars is not None:
        regex = re.compile('[^a-zA-Z]')
        stripped = regex.sub('',token)
        return stripped
    else:
        return token

def correct_vocab(tokens, tqdm_=no_tqdm):
    correct_tweets = [strip_token(token) for token in tqdm_(tokens)]
    return correct_tweets

def strip_tokens(text, tqdm_=no_tqdm):
    tokens = list(tokenize_tweets(text,"raw"))
    corrected_tokens = [strip_token(token) for token in tqdm_(tokens)]
    corrected_tweet = ' '.join(corrected_tokens)
    return corrected_tweet

def strip_text(tweets,tqdm_=no_tqdm):
    correct_tweets = [strip_tokens(x) for x in tqdm_(tweets)]
    return correct_tweets

# GENERATE CLEAN TWEETS

def writeToFile(data,outputSet):
    result = open(outputSet, 'w')
    if (type(data) is np.ndarray or type(data) is list):
        for item in data:
            result.write("%s\n" % item)
        result.close()

pool = multiprocessing.Pool(processes=8)
print('Loading files...')
positive_raw,positive_tweets = loadFile(pos_file)
negative_raw,negative_tweets = loadFile(neg_file)

print('Tokenizing tweets...')
positive_vocab =  tokenize_tweets(positive_raw,"vocab",tqdm)
negative_vocab =  tokenize_tweets(negative_raw,"vocab",tqdm)
word_dict = enchant.Dict("en_US")

print('Stripping text...')
filtered_tweets_pos = strip_text(positive_tweets,tqdm)
filtered_tweets_neg = strip_text(negative_tweets,tqdm)

print('Correcting vocabulary...')
filtered_vocab_pos = correct_vocab(positive_vocab,tqdm)
filtered_vocab_neg = correct_vocab(negative_vocab,tqdm)

print('Computing unique tweets...')
unique_tweets_pos = np.unique(filtered_tweets_pos)
unique_tweets_neg = np.unique(filtered_tweets_neg)

print('Obtaining correct vocabulary...')
filtered_vocab = list(set(filtered_vocab_pos + filtered_vocab_neg))
correct_tweet_vocab = correct_all_tokens(word_dict, filtered_vocab, pool)

print('Initializing correctors')
corrector = dict(zip(filtered_vocab, correct_tweet_vocab))

print('Splitting correct dictionary')
correct_tweet_vocab_pos = [corrector[w] for w in tqdm(filtered_vocab_pos)]
correct_tweet_vocab_neg = [corrector[w] for w in tqdm(filtered_vocab_neg)]

print('Obtaining unique correct vocabulary...')
correct_tweet_vocab_pos_unique = np.unique(correct_tweet_vocab_pos)
correct_tweet_vocab_neg_unique = np.unique(correct_tweet_vocab_neg)

print('Correcting tweets...')
correct_tweets_pos = correct_all_tweets_cached(unique_tweets_pos, corrector)
correct_tweets_neg = correct_all_tweets_cached(unique_tweets_neg, corrector)

print('Writing results to files...')
writeToFile(correct_tweets_pos,'data/clean_train_pos.txt')
writeToFile(correct_tweet_vocab_pos_unique,'data/clean_train_vocab_pos.txt')
writeToFile(correct_tweets_neg,'data/clean_train_neg.txt')
writeToFile(correct_tweet_vocab_neg_unique,'data/clean_train_vocab_neg.txt')

pool.terminate()
