import torch
from transformer_lens import HookedTransformer
from matplotlib import pyplot as plt
import math

from utils import tokenize_and_highlight
from sentences import get_interesting_sentences

torch.set_grad_enabled(False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HookedTransformer.from_pretrained('gpt2-small', device=device)

for s, bitoken in get_interesting_sentences(model.tokenizer):
    print(s)
    tokens, highlighted = tokenize_and_highlight(model.tokenizer, s, bitoken)

    first_token_index = highlighted.index(True)
    print(f'First token index: {first_token_index}')

    ttokens = torch.tensor([tokens], device=device)
    logits, cache = model.run_with_cache(ttokens)
    fig, ax = plt.subplots(model.cfg.n_layers, 2, squeeze=False)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attnstuff = cache['attn_scores', layer, 'attn'][0,head,:,:]
            kstuff = cache['k', layer, 'attn'][0, :, head, :]
            qstuff = cache['q', layer, 'attn'][0, :, head, :]
            print(attnstuff.shape)
            print(kstuff.shape)
            print(qstuff.shape)
            # Calculate the attention values again from k and q
            attnstuff2 = torch.matmul(qstuff, kstuff.transpose(0,1)) / 8
            #print(attnstuff - attnstuff2)
            ax[head][0].imshow(torch.maximum(torch.tensor(-5), attnstuff.cpu()))
            ax[head][1].imshow(attnstuff2.cpu())
            print(attnstuff - attnstuff2)
        fig.suptitle(s)
        plt.show()
