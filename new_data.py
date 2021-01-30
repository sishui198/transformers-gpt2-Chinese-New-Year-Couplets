from transformers import BertTokenizer
import numpy as np

tokenizer = BertTokenizer(vocab_file="vocab.txt")

with open("data/train_data-v2.txt", "r") as f:
    data = f.readlines()
    data = [i[:-1] for i in data]

inputs = tokenizer(data, return_tensors="tf", padding=True, truncation=True)
input_ids = inputs["input_ids"]

print(input_ids[:2])
exit()
np.savez("train_data-cn", input_ids)
