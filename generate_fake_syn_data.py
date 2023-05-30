from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import torchvision
import pandas as pd
import time
import os

import read_data as read_data

df_true, df_fake, shuffled_df = read_data.get_isot_dataset()

file_path = 'syn_fake_data'
file_name = 'fake_syn_data_new_500.csv'

shuff_df_true = pd.read_csv(os.path.join(os.getcwd(), file_path, 'shuffle_true_data.csv'))


model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


fake_syn_data = pd.DataFrame(columns=['title', 'text', 'subject', 'date', 'label'])

# This will generate 500 synthetic fake news examples and store the data
# in file -  fake_syn_data_new_500.csv - in the syn_fake_data folder.
# This will take a lot of time to run, depending on RAM availability, GPU etc.
org_start = time.time()
for i in range(500):
    print("starting: ", i)
    prompt1 = shuff_df_true['title'].iloc[i]

    input_ids1 = tokenizer(prompt1, return_tensors="pt").input_ids

    start = time.time()

    gen_tokens = model.generate(
        input_ids1,
        do_sample=True,
        temperature=0.9,
        max_length=150,
    )
    gen_text1 = tokenizer.batch_decode(gen_tokens)[0]

    end = time.time()

    print("total time elapsed: {} seconds".format(end - start))

    df_syn_fake = {'title': shuff_df_true['title'].iloc[i], 'text': gen_text1, 'subject': 'fake', 'date': 'fake', 'label': 1}
    fake_syn_data = fake_syn_data.append(df_syn_fake, ignore_index=True)

    fake_syn_data.to_csv(os.path.join(os.getcwd(), file_path, file_name), index=False)

fin_end = time.time()

print("total time elapsed: {} minutes".format((fin_end - org_start) / 60))

print("Done, Finished Generating Fake synthetic data from Real News titles")