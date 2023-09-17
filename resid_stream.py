import torch
from transformer_lens import HookedTransformer
from matplotlib import pyplot as plt
import math

from utils import tokenize_and_highlight
from sentences import get_interesting_sentences

torch.set_grad_enabled(False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HookedTransformer.from_pretrained('gpt2-small', device=device)

layer = 0
head = 0
s, bitoken = list(get_interesting_sentences(model.tokenizer))[0]

print(s)
w_q = model.blocks[layer].attn.W_Q[head,:,:]    # 768 x 64
w_k = model.blocks[layer].attn.W_K[head,:,:]    # 768 x 64
b_q = model.blocks[layer].attn.b_Q[head,:]      # 64
b_k = model.blocks[layer].attn.b_K[head,:]      # 64

tokens, highlighted = tokenize_and_highlight(model.tokenizer, s, bitoken)
ttokens = torch.tensor([tokens], device=device)
logits, cache = model.run_with_cache(ttokens)

layernorm = model.blocks[layer].ln1
resid = layernorm(cache['resid_pre', layer])[0, :, :]      # n_tokens x 768
attn_scores = cache['attn_scores', layer, 'attn'][0, head, :, :] # n_tokens x n_tokens
q = cache['q', layer, 'attn'][0, head, :, :]    # n_tokens x 64
k = cache['k', layer, 'attn'][0, head, :, :]    # n_tokens x 64
q2 = torch.matmul(resid, w_q) + b_q          # n_tokens x 64
k2 = torch.matmul(resid, w_k) + b_k          # n_tokens x 64
print()
print()
print()
#print(q)
#print(q2)
attn_scores2 = torch.matmul(q2, k2.transpose(0,1)) / 8 # n_tokens x n_tokens
print(attn_scores - attn_scores2)
