import pandas as pd
import numpy as np
import parse_comments_utils

#libraries for connect to database
import datetime
from sqlalchemy import (create_engine, Column, Integer, String, Float, Text,
                        BigInteger, DateTime, ForeignKey, Boolean)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import config
import psycopg2


def load_process_comments(connection_sting):
	"""
	Load data and write into dataframes.
	Input: connection string to connect to the database
	Output:
		data_200k: entire original dataset
		data: labeled portion of original dataset
		parsed_comments_200k: entire processed dataset
		parsed_commnets: labeled portion of processed dataset
	Note: on ~220K comments, this function to load and parse comments takes about 10min to run.
	"""
	# connect:
	con = psycopg2.connect(connection_sting)

	# query:
	sql_query = """
	SELECT comment_id, instagram_id, comment_text, label FROM comment;
	"""

	#execute sql to read data from database, assign to dataframe "data_200k"
	data_200k = pd.read_sql_query(sql_query,con)
	#dataframe "data" is labeled part of the entire dataset
	data = data_200k[np.isfinite(data_200k['label'])].copy()
	data.label = data.label.astype(np.int64)
	
	#create a dataframe to store post-processing text elements and labels
	parsed_comments_200k = pd.DataFrame(columns = ['original_comment','full_text_ngramT','text_onlyT','emojiT','translated_emojiT','usernameT','hashtagT','punctuation','language','label','purchase_intent'])
	parsed_comments_200k['original_comment'] = data_200k['comment_text']
	parsed_comments_200k['label'] = data_200k['label']
	parsed_comments_200k['purchase_intent'] = parsed_comments_200k['label']
	parsed_comments_200k['purchase_intent'].replace({3:1,-2:0,-1:0,1:0,2:0,4:0},inplace=True)
	parsed_comments_200k['full_text_ngramT'] = np.empty((len(parsed_comments_200k), 0)).tolist()
	parsed_comments_200k['text_onlyT'] = np.empty((len(parsed_comments_200k), 0)).tolist()
	parsed_comments_200k['emojiT'] = np.empty((len(parsed_comments_200k), 0)).tolist()
	parsed_comments_200k['translated_emojiT'] = np.empty((len(parsed_comments_200k), 0)).tolist()
	parsed_comments_200k['usernameT'] = np.empty((len(parsed_comments_200k), 0)).tolist()
	parsed_comments_200k['hashtagT']  = np.empty((len(parsed_comments_200k), 0)).tolist()
	parsed_comments_200k['punctuation'] = ''
	parsed_comments_200k['language'] = ''

	#use functions from parse_comments_utils.py to extract text elements
	for index, row in parsed_comments_200k.iterrows():
		tokens = parse_comments_utils.preprocess(row['original_comment'])
		text_only,language = parse_comments_utils.get_text_only(tokens)
		parsed_comments_200k.set_value(index,'language',language)
		if language == 'en':
			parsed_comments_200k.set_value(index,'usernameT',parse_comments_utils.get_at_tokens(tokens))
			hashes = parse_comments_utils.get_hashtag_tokens(tokens)
			parsed_comments_200k.set_value(index,'hashtagT',hashes)
			emojis = parse_comments_utils.get_emoji_tokens(tokens)
			translated = parse_comments_utils.translate_emoji_ngram(emojis)
			parsed_comments_200k.set_value(index,'emojiT',emojis)
			parsed_comments_200k.set_value(index,'translated_emojiT',translated)
			parsed_comments_200k.set_value(index,'punctuation',''.join(parse_comments_utils.get_punctuation_tokens(tokens)))
			text_new = parse_comments_utils.delete_repeatings_text_only(parse_comments_utils.stemmed_text_only(text_only))
			parsed_comments_200k.set_value(index,'text_onlyT',text_new)
			parsed_comments_200k.set_value(index,'full_text_ngramT',text_new + translated + hashes)

	#save the labeled portion as a seperate dataframe from parsed_comments_200k
	parsed_comments = parsed_comments_200k[np.isfinite(parsed_comments_200k['label'])].copy()
	parsed_comments.label = parsed_comments.label.astype(np.int64)
	parsed_comments.purchase_intent = parsed_comments.purchase_intent.astype(np.int64)

	return data_200k,data,parsed_comments_200k,parsed_comments
