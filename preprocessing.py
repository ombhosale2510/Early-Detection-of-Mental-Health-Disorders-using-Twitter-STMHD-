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
import re
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

def clean_text(orig_text: str) -> str:
    # This function should clean up the text of each tweet to remove extra whitespace, punctuation, etc and convert emojis
    text = orig_text.strip() # first pass of whitespace trimming

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http', '', text)

    text = convert_emojis(text)
    text = replace_emoticons(text)
    
    # TODO: any other text or punctuation that should be stripped?

    return text.strip()

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
                    tweet_text_cleaned = clean_text(tweet["text"])

                    if filter_tweet(tweet_text_cleaned):
                        continue # skip tweets that don't have alphanumeric characters

                    tweets_dicts.append({
                        "tweet_id": tweet["tweet_id"],
                        "tweet_text": tweet_text_cleaned,
                        "tweet_day": day_of_tweets,
                        "pre_covid": None,
                        "disorder_flag": tweet["disorder_flag"],
                        "disorder": "negative"
                    })

    for disorder in disorders:
        disorder_path = ALL_DISORDERS[disorder]

        for coviddir in ["precovid", "postcovid"]:
            for user_id in os.listdir(f"{disorder_path}/{coviddir}"):
                if not user_id.isdigit():
                    continue # skip system files such as .DS_Store

                user_tweets_file = f"{disorder_path}/{coviddir}/{user_id}/tweets.json"

                with open(user_tweets_file, 'r') as f:
                    tweets_json = json.load(f)

                    for day_of_tweets, tweets_of_day in tweets_json.items():
                        for tweet in tweets_of_day:
                            tweet_text_cleaned = clean_text(tweet["text"])

                            tweets_dicts.append({
                                "tweet_id": tweet["tweet_id"],
                                "tweet_text": tweet_text_cleaned,
                                "tweet_day": day_of_tweets,
                                "pre_covid": (coviddir == "precovid"),
                                "disorder_flag": tweet["disorder_flag"],
                                "disorder": disorder
                            })

    tweets_df = pd.DataFrame(tweets_dicts, columns=["tweet_id", "tweet_text", "tweet_day", "pre_covid", "disorder_flag", "disorder"])
    tweets_df.set_index("tweet_id", inplace=True)

    return tweets_df

if __name__ == "__main__":
    print(create_tweets_df(["depression"]))

