import torch
from transformer_lens import HookedTransformer

from corpus import get_bitokens, get_sentences_containing

torch.set_grad_enabled(False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HookedTransformer.from_pretrained('gpt2-small', device=device)

# Get the bitokens
bitokens = get_bitokens(model.tokenizer)

# Print the most common ones
for bitoken, count in bitokens.most_common(50):
    print(f'{bitoken:20}', f'{count:4}', [model.tokenizer.decode([t]) for t in model.tokenizer.encode(' ' + bitoken)])
