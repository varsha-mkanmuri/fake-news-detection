import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

import os
import sys


def get_isot_dataset():

    file_path = 'data'
    file_true_news = 'True.csv'
    file_fake_news = 'Fake.csv'

    df_true = pd.read_csv(os.path.join(os.getcwd(), file_path, file_true_news))

    df_fake = pd.read_csv(os.path.join(os.getcwd(), file_path, file_fake_news))

    df_true['label'] = 0  # True News
    df_fake['label'] = 1  # Fake News

    shuff_df_true = df_true.sample(frac=1, random_state=42).copy()
    shuff_df_true.to_csv(os.path.join(os.getcwd(), 'syn_fake_data', 'shuffle_true_data_new.csv'), index=False)

    df = pd.concat([df_true,df_fake])

    df_data = df.copy()
    df_data = df_data.reset_index(drop=True)

    shuffled_df = df_data.sample(frac=1, random_state=42).copy()

    shuffled_df['text'] = shuffled_df['title'] + ' ' + shuffled_df['text']



    return df_true, df_fake, shuffled_df
