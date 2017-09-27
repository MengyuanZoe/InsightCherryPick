import pandas as pd
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import models, corpora, similarities
from collections import defaultdict
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def sentiment_analysis(parsed_comments):
	"""
	Sentiment analysis on each comment in dataframe parsed_comments.
	Input: dataframe containing parsed comments
	Output: add 4 new columns of sentiment scores into the input dataframe
	"""
	parsed_comments['compound'] = 0.0
	parsed_comments['pos'] = 0.0
	parsed_comments['neu'] = 0.0
	parsed_comments['neg'] = 0.0

	sid = SentimentIntensityAnalyzer()
	for index, row in parsed_comments.iterrows():
		text = (' '.join(row['full_text_ngramT'])).replace('_',' ')
		ss = sid.polarity_scores(text)
		parsed_comments.set_value(index,'compound',ss['compound'])
		parsed_comments.set_value(index,'pos',ss['pos'])
		parsed_comments.set_value(index,'neu',ss['neu'])
		parsed_comments.set_value(index,'neg',ss['neg'])

	return parsed_comments


def text_vectorization(parsed_comments_200k, parsed_comments, dim=50):
	"""
	Vectorization text in each comment in dataframe parsed_comments.
	Input: dataframe containing full parsed comments for training corpus; 
			dataframe containing labeled parsed comments
			number of dimensions of targeted vector space 
	Output: add $(dim) new columns of vector element into the labeled dataframe parsed_comments
	"""
	documents = []
	for index, row in parsed_comments_200k.iterrows():
		documents.append(row['full_text_ngramT'])
	stopwords = nltk.corpus.stopwords.words('english')
	texts = [[token for token in tokens if token not in stopwords] for tokens in documents]
	# remove words that appear only once
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1

	texts = [[token for token in text if frequency[token] > 1] for text in texts]
	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=dim) # initialize an LSI transformation
	corpus_lsi = lsi[corpus_tfidf]

	list_comp = [[item[1] for item in row] if len(row)==dim else [0]*dim for row in corpus_lsi]
	tfidffeature = np.asarray(list_comp)
	parsed_comments_200k[['tfidf_{}'.format(i) for i in range(0,dim)]] = pd.DataFrame(index=parsed_comments_200k.index, data=[x for x in tfidffeature])

	#parsed_comments left join with parsed_commenets_200k
	#result = pd.concat([parsed_comments,parsed_comments_200k],axis = 1, join='inner')
	#result = pd.merge(parsed_comments,parsed_comments_200k,on = index,how='inner')
	parsed_comments1 = parsed_comments_200k.loc[parsed_comments.index.tolist()]
	cols_to_use = parsed_comments1.columns.difference(parsed_comments.columns)
	result = pd.merge(parsed_comments, parsed_comments1[cols_to_use], left_index=True, right_index=True, how='outer')

	return result


def categorical_features(parsed_comments):
	"""
	Add categorical features for each comment in input dataframe parsed_comments.
	Input: dataframe containing parsed comments
	Output: add 4 columns of categorical features:
		number of emojis
		has ! or not
		has ? or not
		has @ or not
	"""
	parsed_comments['has_exag'] = 0
	parsed_comments['has_ques'] = 0
	parsed_comments['has_at'] = 0
	parsed_comments['num_emoji'] = 0

	for index, row in parsed_comments.iterrows():
		if parsed_comments['usernameT'][index] != []:
			parsed_comments.set_value(index,'has_at',1)
		if '!' in parsed_comments['punctuation'][index]:
			parsed_comments.set_value(index,'has_exag',1)
		if '?' in parsed_comments['punctuation'][index]:
			parsed_comments.set_value(index,'has_ques',1)
		parsed_comments.set_value(index,'num_emoji',len(parsed_comments['emojiT'][index]))

	return parsed_comments

def clean_features(features):
	"""
	Clean up features including throw out rows with all 0.0 feature values (non-english etc)
	Input: features dataframe
	Output: cleaned up feature dataframe
	"""
	features = features[np.sqrt(np.square(features.loc[0:,'compound':]).sum(axis=1)) != 0.0]
	features['num_emoji'] = features['num_emoji'].copy()/100.
	return features

