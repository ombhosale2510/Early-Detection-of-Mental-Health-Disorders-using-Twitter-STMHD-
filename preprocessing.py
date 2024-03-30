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
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_poisson_deviance,
)

import re
import string
import wordninja
import html
from collections import OrderedDict

import nltk
nltk.download('stopwords', quiet=True)
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt')

from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from nltk.tokenize import TweetTokenizer
#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize

from emot.emo_unicode import UNICODE_EMOJI
from emot.emo_unicode import EMOTICONS_EMO


def process_pictograms(text: str, preserve: bool = True) -> str:
    if preserve:
        for emot in UNICODE_EMOJI:
            text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
        for emot in EMOTICONS_EMO:
            text = text.replace(emot, "_".join(EMOTICONS_EMO[emot].replace(",","").replace(":","").split()))
    
    else:
        for emot in UNICODE_EMOJI:
            text = text.replace(emot, "")
        for emot in EMOTICONS_EMO:
            text = text.replace(emot, "")
    
    return text

def clean_text(orig_text: str, preserve_pictograms: bool = True) -> str:
    # This function should clean up the text of each tweet to remove extra whitespace, punctuation, etc and convert emojis
    text = orig_text.strip() # first pass of whitespace trimming

    text = process_pictograms(text, preserve_pictograms)

    # these are taken directly from github.com/luisroberto-maker/MDD-and-PDD_Paper/blob/main/Pre-processing.ipynb
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

    text = html.unescape(text)

    text.translate(str.maketrans('', '', string.punctuation))
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = wordninja.split(text)
    text = " ".join(text)
    text = text.lower()
    text = " ".join(word for word in str(text).split() if word not in stopwords.words('english'))
    
    # TODO: any other text or punctuation that should be stripped?

    return text.strip()

def filter_tweet(text: str) -> bool:
    # This function should filter out any tweets that are not relevant or long enough to use for this project, or that are just empty
    
    # TODO: implement more of a filter than just checking for the empty string
    return len(text) == 0

def identify_anchor_tweet(anchor_text: str, candidate_text: str) -> bool:
    anchor_text = clean_text(anchor_text, False)
    candidate_text = clean_text(candidate_text, False)

    anchor_stripped = " ".join(anchor_text.split()).strip().lower()
    candidate_stripped = " ".join(candidate_text.split()).strip().lower()

    return anchor_stripped == candidate_stripped

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
                        "anchor_tweet": False,
                        "pre_covid_anchor": None,
                        "disorder_flag": tweet["disorder_flag"],
                        "disorder": "negative"
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
                    anchor_tweet_text = tweet_json["anchor_tweet"]
                    anchor_tweet_date = tweet_json["anchor_tweet_date"]

                user_tweets_file = f"{disorder_path}/{coviddir}/{user_id}/tweets.json"

                with open(user_tweets_file, 'r') as f:
                    tweets_json = json.load(f)

                    for day_of_tweets, tweets_of_day in tweets_json.items():
                        for tweet in tweets_of_day:
                            tweet_text_raw = tweet["text"]
                            tweet_text_cleaned = clean_text(tweet_text_raw)

                            is_anchor_tweet = False
                            if identify_anchor_tweet(anchor_tweet_text, tweet_text_raw):
                                is_anchor_tweet = True

                            if filter_tweet(tweet_text_cleaned):
                                continue

                            tweets_dicts.append({
                                "tweet_id": tweet["tweet_id"],
                                "user_id": user_id,
                                "tweet_text_raw": tweet_text_raw,
                                "tweet_text": tweet_text_cleaned,
                                "tweet_day": day_of_tweets,
                                "anchor_tweet": is_anchor_tweet,
                                "pre_covid_anchor": (coviddir == "precovid"),
                                "disorder_flag": tweet["disorder_flag"],
                                "disorder": disorder
                            })

    tweets_df = pd.DataFrame(tweets_dicts, columns=[
        "tweet_id", "user_id", "tweet_text_raw", "tweet_text", "tweet_day", "anchor_tweet", "pre_covid_anchor", "disorder_flag", "disorder"
    ])
    tweets_df.set_index("tweet_id", inplace=True)

    return tweets_df

if __name__ == "__main__":
    tweets_df = create_tweets_df(["depression"])
    print(tweets_df[tweets_df.anchor_tweet.eq(True)])

