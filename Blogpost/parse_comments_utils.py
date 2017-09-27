import string
import re
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import langid
import string
import nltk
from gensim import models, corpora, similarities
from collections import defaultdict


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
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
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
 
def get_hashtag_tokens(tokens):
    hashtag_tokens = []
    hashtag_tokens = hashtag_tokens +[term.replace('#','').lower() for term in tokens 
              if term.startswith('#')]
    return hashtag_tokens

def get_at_tokens(tokens):
    at_tokens = []
    at_tokens = at_tokens +[term.replace('@','') for term in tokens 
              if term.startswith('@')]
    return at_tokens

def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return 1
    return 0

def get_emoji_tokens(tokens):
    emoji_tokens = []
    emoji_tokens = emoji_tokens + [term for term in tokens
                                  if text_has_emoji(term)]
    return emoji_tokens

def translate_emoji_ngram(tokens):
    emojitrans_tokens = [emoji.demojize(term, ('','')).lower() 
                                            for term in tokens]
    return emojitrans_tokens

def get_punctuation_tokens(tokens):
    punct_tokens = []
    punct_tokens = punct_tokens + [term for term in tokens
                                   if term in string.punctuation]
    return punct_tokens

def isNormal(myStr):
    myChars = [ord(x) for x in myStr]
    return all(x < 127 and x > 31 for x in myChars)

def get_text_only(tokens):
    terms_only = [term for term in tokens 
                  if term.replace('#','').lower() not in get_hashtag_tokens(tokens) and
                  term.replace('@','') not in get_at_tokens(tokens) and
                  term not in get_emoji_tokens(tokens) and
                  term not in get_punctuation_tokens(tokens) and
                  term not in url_re.findall(term) and
                  isNormal(term)]
    language = langid.classify(' '.join(terms_only))[0]
    #terms_only_new = []
    #if language == 'en':
    #    terms_only_new = [term for term in terms_only
    #              if isNormal(term)] #get rid of some remaining parts of unidentifiable emojis etc
    return terms_only, language

def stemmed_text_only(tokens):
    snowball_stemmer = nltk.stem.SnowballStemmer('english')
    stemmed = []
    stemmed = stemmed + [snowball_stemmer.stem(term) for term in tokens]
    return stemmed

# Repeating words like hurrrryyyyyy
rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);
def rpt_repl(match):
    return match.group(1)+match.group(1)


def delete_repeatings_text_only(tokens):
    new_tokens = []
    for term in tokens:
        new_tokens.append(re.sub(rpt_regex, rpt_repl, term))
    return new_tokens

def text2features(text):
    tokens = preprocess(text)
    text_only,language = get_text_only(tokens)
    #if language == 'en':
    usernameT = get_at_tokens(tokens)
    has_username = 0
    if usernameT != []:
        has_username = 1
    hashes = get_hashtag_tokens(tokens)
    emojis = get_emoji_tokens(tokens)
    num_emoji = len(emojis)
    translated = translate_emoji_ngram(emojis)
    punctuation = ''.join(get_punctuation_tokens(tokens))
    has_exag = 0
    if '!' in punctuation:
        has_exag = 1
    has_ques = 0
    if '?' in punctuation:
        has_ques = 1
    text_onlyT = delete_repeatings_text_only(stemmed_text_only(text_only))
    full_text_ngramT = text_onlyT + translated + hashes


    #tfidf features
    dictionary = corpora.Dictionary.load('/Users/mengyuan/Documents/Insight/InsightCherryPick/Models/comments_200k.dict')
    corpus = corpora.MmCorpus('/Users/mengyuan/Documents/Insight/InsightCherryPick/Models/comments_200k.mm')
    tfidf = tfidf_model = models.TfidfModel.load('/Users/mengyuan/Documents/Insight/InsightCherryPick/Models/tfidf_200k.tfidf_model')
    lsi = models.LsiModel.load('/Users/mengyuan/Documents/Insight/InsightCherryPick/Models/model_50.lsi')
    tfidfraw = lsi[tfidf[dictionary.doc2bow(full_text_ngramT)]]
    tfidffeatures = [item[1] for item in tfidfraw] #this is a list
    #sentiment features
    sentence = (' '.join(full_text_ngramT)).replace('_',' ')
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
    sentifeatures = [ss['compound'],ss['pos'],ss['neu'],ss['neg']]#[item.value for item in ss]
    #categorical features
    categoricalfeatures = [has_exag,has_ques,has_username,num_emoji/100.]

    return categoricalfeatures + sentifeatures + tfidffeatures


