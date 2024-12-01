from transformers import AutoTokenizer
import torch
from transformers import AutoConfig
from math import sqrt

model_ckpt = "bert-base-uncased"
text = "time flies like an arrow"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs.input_ids)

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())

query = key = value = inputs_embeds

dim_k = key.size(-1)
print(f'dim_k shape is {dim_k}')
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
scores.size()