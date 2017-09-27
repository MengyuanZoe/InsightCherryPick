"""
Functions for comment pre-processing
inluding: extract emojis, text, punctuations, @ and #; 
          tokenization, stemming and getting rid of repeating characters such as "neeeed"
"""

import string
import re
import emoji  #see: https://github.com/carpedm20/emoji/
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import langid
import string
import nltk
from gensim import models, corpora, similarities
from collections import defaultdict

#define emoticons, not actually used, but can be easily added
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

#define how to identify each component
regex_str = [
    #emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
url_re = re.compile(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')


def tokenize(s):
    """
    Tokenize a string.
    input: string s
    output: a list of tokens
    """
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    """
    Preprocess a string, including tokenize the non-emoticon part and change to lower cases.
    input: string s, flag for lowercase
    output: a list of tokens
    """
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
 
def get_hashtag_tokens(tokens):
    """
    Extract words as tokens after # sign.
    input: a list of tokens
    output: a list of tokens that start with #
    """
    hashtag_tokens = []
    hashtag_tokens = hashtag_tokens +[term.replace('#','').lower() for term in tokens 
              if term.startswith('#')]
    return hashtag_tokens

def get_at_tokens(tokens):
    """
    Extract words as tokens after @ sign.
    input: a list of tokens
    output: a list of tokens that start with @
    """
    at_tokens = []
    at_tokens = at_tokens +[term.replace('@','') for term in tokens 
              if term.startswith('@')]
    return at_tokens

def text_has_emoji(text):
    """
    Check if a list of tokens has emoji.
    input: token list 
    output: 1 if input has emojis, 0 is not
    """
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return 1
    return 0

def get_emoji_tokens(tokens):
    """
    Extract emojis from a list of tokens.
    input: a list of tokens
    output: a list of emojis from the input token list
    """
    emoji_tokens = []
    emoji_tokens = emoji_tokens + [term for term in tokens
                                  if text_has_emoji(term)]
    return emoji_tokens

def translate_emoji_ngram(tokens):
    """
    Translate emoji from unicode to a string with emoji shortcodes.
    input: a list of unicode emoji tokens
    output: a list of translated emoji names, e.g. red_heart
    """
    emojitrans_tokens = [emoji.demojize(term, ('','')).lower() 
                                            for term in tokens]
    return emojitrans_tokens

def get_punctuation_tokens(tokens):
    """
    Extract punctuations from a list of tokens.
    input: a list of tokens
    output: a list of punctuations from the input
    """
    punct_tokens = []
    punct_tokens = punct_tokens + [term for term in tokens
                                   if term in string.punctuation]
    return punct_tokens

def isNormal(myStr):
    """
    Screen out non-normal (normal: alphabets, numbers, punctuations) characters.
    input: string
    ouput: normal characters fron the input string
    Note: this is used to get rid of some unrecognizable leftover of emoji unicodes.
    """ 
    myChars = [ord(x) for x in myStr]
    return all(x < 127 and x > 31 for x in myChars)

def get_text_only(tokens):
    """
    Extract only text component from tokenzied comment.
    input: a list of tokens
    output: text compom=nents from the input token list
    """
    terms_only = [term for term in tokens 
                  if term.replace('#','').lower() not in get_hashtag_tokens(tokens) and
                  term.replace('@','') not in get_at_tokens(tokens) and
                  term not in get_emoji_tokens(tokens) and
                  term not in get_punctuation_tokens(tokens) and
                  term not in url_re.findall(term) and
                  isNormal(term)]
    language = langid.classify(' '.join(terms_only))[0]
    return terms_only, language

def stemmed_text_only(tokens):
    """
    Word stemming using NLTK Snowball stemmer.
    input: a list of word tokens
    output: stemmed word tokens
    """
    snowball_stemmer = nltk.stem.SnowballStemmer('english')
    stemmed = []
    stemmed = stemmed + [snowball_stemmer.stem(term) for term in tokens]
    return stemmed

# Repeating words like hurrrryyyyyy
rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);
def rpt_repl(match):
    return match.group(1)+match.group(1)


def delete_repeatings_text_only(tokens):
    """
    Delect >2 repeatings characters in word.
    input: word tokens
    output: word tokens with at most 2 repeating characters in a word
    example: neeeeeed -> need, wowwww -> woww
    """
    new_tokens = []
    for term in tokens:
        new_tokens.append(re.sub(rpt_regex, rpt_repl, term))
    return new_tokens


