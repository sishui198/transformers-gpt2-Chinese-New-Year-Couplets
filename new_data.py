from transformers import BertTokenizer
import numpy as np

tokenizer = BertTokenizer(vocab_file="vocab-cn-v3.txt")

with open("data/train_data-v2.txt", "r") as f:
    data = f.readlines()
    data = [i[:-1] for i in data]

inputs = tokenizer(data, return_tensors="tf", padding=True, truncation=True, add_special_tokens=False)
input_ids = inputs["input_ids"]
s = tokenizer.get_vocab()["[CLS]"]
d = tokenizer.get_vocab()["[SEP]"]

new_ans = []
for i in input_ids:
    ans_ls = list(i.numpy())
    ans_ls.insert(0, s)
    try:
        z_index = ans_ls.index(0)
        ans_ls.insert(z_index, d)
    except:
        ans_ls.insert(-1, d)
    new_ans.append(ans_ls)

input_ids = np.array(new_ans)
print(input_ids)
print(input_ids.shape)
np.savez("train_data-cn", input_ids)
