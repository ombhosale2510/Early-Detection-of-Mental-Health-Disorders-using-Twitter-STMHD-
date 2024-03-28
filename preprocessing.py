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

def clean_text(orig_text: str) -> str:
    # TODO: implement this
    return orig_text

NEGATIVE_DATADIR = "./data_sample/neg"
DEPRESSION_DATADIR = "./data_sample/depression"
ALL_DISORDERS = {
    "depression": DEPRESSION_DATADIR
}

tweets_dicts = []

for user_id in os.listdir(f"{NEGATIVE_DATADIR}"):
    if not user_id.isdigit():
        continue # skip system files such as .DS_Store

    user_tweets_file = f"{NEGATIVE_DATADIR}/{user_id}/tweets.json"

    with open(user_tweets_file, 'r') as f:
        tweets_json = json.load(f)

        for day_of_tweets, tweets_of_day in tweets_json.items():
            for tweet in tweets_of_day:
                tweet_text_cleaned = clean_text(tweet["text"])

                tweets_dicts.append({
                    "tweet_id": tweet["tweet_id"],
                    "tweet_text": tweet_text_cleaned,
                    "tweet_day": day_of_tweets,
                    "disorder_flag": tweet["disorder_flag"],
                    "pre_covid": None,
                    "mental_illness": "negative"
                })

for disorder, disorder_path in ALL_DISORDERS.items():
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

if __name__ == "__main__":
    print(tweets_df)

