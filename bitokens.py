import torch
from transformer_lens import HookedTransformer
from matplotlib import pyplot as plt
import math

from corpus import get_bitokens, get_sentences_containing
from utils import tokenize_and_highlight

torch.set_grad_enabled(False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HookedTransformer.from_pretrained('gpt2-small', device=device)

bitokens = ['shear', 'binomial', 'tangent', 'detergent', 'leaped', 'parlor', 'textiles', 'fastened', 'lagoon', 'inventories', 'painters']

for bitoken in bitokens:
    best_score = -10000000000
    best_s = None
    for s in get_sentences_containing(bitoken):
        tokens, highlighted = tokenize_and_highlight(model.tokenizer, s, bitoken)
        # Want at least 4 tokens after the highlighted ones. Otherwise the shorter the better.
        if True in highlighted and highlighted.index(True) + 4 < len(highlighted):
            score = -len(tokens)
            if score > best_score:
                best_score = score
                best_s = s

    if best_s is None:
        continue
    s = best_s
    print(s)
    tokens, highlighted = tokenize_and_highlight(model.tokenizer, s, bitoken)
    ttokens = torch.tensor([tokens], device=device)
    logits, cache = model.run_with_cache(ttokens)
    fig, ax = plt.subplots(model.cfg.n_layers, model.cfg.n_heads, squeeze=False)
    for layer in range(model.cfg.n_layers):
        attn = cache['pattern', layer, 'attn'][0,:,:,:].cpu()
        for head in range(model.cfg.n_heads):
            colors = torch.zeros((len(tokens), len(tokens), 3))
            for i in range(len(tokens)):
                for j in range(len(tokens)):
                    value = math.sqrt(attn[head,i,j])
                    if highlighted[j] and j<len(tokens)-1 and highlighted[j+1]:
                        if i == j or i == j+1:
                            colors[i,j,0] = value
                            colors[i,j,1] = value
                            colors[i,j,2] = value
                        else:
                            colors[i,j,0] = value
                            colors[i,j,1] = min(value, math.sqrt(attn[head,i,j+1]))
                    elif highlighted[j]:
                        colors[i,j,1] = value
                    else:
                        colors[i,j,2] = value
            ax[layer, head].set_axis_off()
            ax[layer, head].imshow(colors)
    fig.suptitle(s)
    plt.show()
