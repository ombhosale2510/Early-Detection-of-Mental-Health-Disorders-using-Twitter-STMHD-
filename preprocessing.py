# ENGG*6600 ST: Deep Learning Term Project W24
# Mental Illness Classification with RNNs
# Dataset: https://zenodo.org/records/6409736
# 
# Ben Chapman-Kish (bchapm02@uoguelph.ca)
# John Quinto (jquinto@uoguelph.ca)
# Om Bhosale (obhosale@uoguelph.ca)
# Parya Abadeh (pabadeh@uoguelph.ca)
# 
# Pre-processing file

import os
import json
import pandas as pd

import re
import string
import wordninja
from collections import OrderedDict
import dateutil.parser

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from emot.emo_unicode import UNICODE_EMOJI
from emot.emo_unicode import EMOTICONS_EMO

def convert_emojis(text: str) -> str:
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
    return text

def replace_emoticons(text: str) -> str:
    for emot in EMOTICONS_EMO:
        text = text.replace(emot, "_".join(EMOTICONS_EMO[emot].replace(",","").replace(":","").split()))
    return text

# taken from https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e

# Initialize the lemmatizer
word_net_lemmatizer = WordNetLemmatizer()
tweet_tokenizer = TweetTokenizer()
 
# This is a helper function to map NTLK position tags
def _get_wordnet_pos(tag: str) -> str:
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Lemmatize and tokenize the sentence
def lemmatize(text: str) -> str:
    word_pos_tags = nltk.pos_tag(tweet_tokenizer.tokenize(text)) # Get position tags
    a = [word_net_lemmatizer.lemmatize(tag[0], _get_wordnet_pos(tag[1])) for tag in word_pos_tags] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def clean_text(orig_text: str) -> str:
    # This function should clean up the text of each tweet to remove extra whitespace, punctuation, etc and convert emojis
    text = orig_text

    # convert pictograms into a textual representation
    text = convert_emojis(text)
    text = replace_emoticons(text)

    # basic text cleaning to remove special characters and URLs
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'$\w*', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http', '', text)
    text = re.sub(r'\@\w+|\#\…', '', text)
    text = re.sub(r'…', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = ' '.join(OrderedDict((w,w) for w in text.split()).keys())

    # more basic text cleaning to remove punctuation, remove/insert whitespace where necessary, and convert text case

    text.translate(str.maketrans('', '', string.punctuation))
    text = wordninja.split(text)
    text = " ".join(text)
    text = text.lower().strip()

    # remove stopwords
    text = " ".join(word for word in str(text).split() if word not in stopwords.words('english'))
    
    # TODO: fix typos, replace abbreviations/short forms, and expand contractions
    # TODO: perform stemming (slicing the end or the beginning of words with the intention of removing affixes)
    
    text = lemmatize(text)

    return text

def filter_tweet(text: str) -> bool:
    # This function should filter out any tweets that are not relevant or long enough to use for this project, or that are just empty
    
    # TODO: implement more of a filter than just checking for the empty string
    return len(text) == 0

NEGATIVE_PATH = "./data_sample/neg"
ALL_DISORDERS = {
    "adhd": "./data_sample/adhd",
    "anxiety": "./data_sample/anxiety",
    "bipolar": "./data_sample/bipolar",
    "depression": "./data_sample/depression",
    "mdd": "./data_sample/mdd",
    "ocd": "./data_sample/ocd",
    "ppd": "./data_sample/ppd",
    "ptsd": "./data_sample/ptsd"
}

def create_tweets_df(disorders: list[str]) -> pd.DataFrame:
    if len(disorders) < 1:
        raise ValueError("Must provide at least one disorder in addition to the negative group")
    for disorder in disorders:
        if disorder not in ALL_DISORDERS:
            raise ValueError(f"The disorder '{disorder}' does not have a corresponding dataset")

        elif not os.path.exists(ALL_DISORDERS[disorder]):
            raise FileNotFoundError(f"The dataset for disorder '{disorder}' cannot be found")

    tweets_dicts = []

    for user_id in os.listdir(f"{NEGATIVE_PATH}"):
        if not user_id.isdigit():
            continue # skip system files such as .DS_Store

        user_tweets_file = f"{NEGATIVE_PATH}/{user_id}/tweets.json"

        with open(user_tweets_file, 'r') as f:
            tweets_json = json.load(f)

            for day_of_tweets, tweets_of_day in tweets_json.items():
                for tweet in tweets_of_day:
                    tweet_text_raw = tweet["text"]
                    tweet_text_cleaned = clean_text(tweet_text_raw)

                    if filter_tweet(tweet_text_cleaned):
                        continue # skip tweets that don't have alphanumeric characters

                    tweets_dicts.append({
                        "tweet_id": tweet["tweet_id"],
                        "user_id": user_id,
                        "tweet_text_raw": tweet_text_raw,
                        "tweet_text": tweet_text_cleaned,
                        "tweet_day": day_of_tweets,
                        "before_anchor_tweet": None,
                        "pre_covid_anchor": None,
                        "disorder_discourse": bool(tweet["disorder_flag"]),
                        "disorder_name": None,
                        "has_disorder": False
                    })

    for disorder in disorders:
        disorder_path = ALL_DISORDERS[disorder]

        for coviddir in ["precovid", "postcovid"]:
            for user_id in os.listdir(f"{disorder_path}/{coviddir}"):
                if not user_id.isdigit():
                    continue # skip system files such as .DS_Store

                user_anchor_tweet_file = f"{disorder_path}/{coviddir}/{user_id}/anchor_tweet.json"

                with open(user_anchor_tweet_file, 'r') as f:
                    tweet_json = json.load(f)
                    anchor_tweet_date = dateutil.parser.parse(tweet_json["anchor_tweet_date"])

                user_tweets_file = f"{disorder_path}/{coviddir}/{user_id}/tweets.json"

                with open(user_tweets_file, 'r') as f:
                    tweets_json = json.load(f)

                    for day_of_tweets, tweets_of_day in tweets_json.items():
                        tweet_date = dateutil.parser.parse(day_of_tweets)

                        for tweet in tweets_of_day:
                            tweet_text_raw = tweet["text"]
                            tweet_text_cleaned = clean_text(tweet_text_raw)
                            
                            # TODO: we need to perform vectorization on the text now for training inputs

                            if filter_tweet(tweet_text_cleaned):
                                continue

                            tweets_dicts.append({
                                "tweet_id": tweet["tweet_id"],
                                "user_id": user_id,
                                "tweet_text_raw": tweet_text_raw,
                                "tweet_text": tweet_text_cleaned,
                                "tweet_day": day_of_tweets,
                                "before_anchor_tweet": (tweet_date < anchor_tweet_date),
                                "pre_covid_anchor": (coviddir == "precovid"),
                                "disorder_discourse": bool(tweet["disorder_flag"]),
                                "disorder_name": disorder,
                                "has_disorder": True
                            })

    tweets_df = pd.DataFrame(tweets_dicts, columns=[
        "tweet_id", "user_id", "tweet_text_raw", "tweet_text", "tweet_day", "before_anchor_tweet", "pre_covid_anchor", "disorder_discourse", "disorder_name", "has_disorder"
    ])
    tweets_df.set_index("tweet_id", inplace=True)

    return tweets_df

if __name__ == "__main__":
    tweets_df = create_tweets_df(["depression"])
    print(tweets_df[["tweet_text_raw", "tweet_text"]])
