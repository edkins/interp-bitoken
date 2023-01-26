from corpus import get_sentences_containing
from utils import tokenize_and_highlight

bitokens = ['shear', 'binomial', 'tangent', 'detergent', 'leaped', 'parlor', 'textiles', 'fastened', 'lagoon', 'inventories', 'painters']

def get_interesting_sentences(tokenizer):
    for bitoken in bitokens:
        best_score = -10000000000
        best_s = None
        for s in get_sentences_containing(bitoken):
            tokens, highlighted = tokenize_and_highlight(tokenizer, s, bitoken)
            # Want at least 4 tokens after the highlighted ones. Otherwise the shorter the better.
            if True in highlighted and highlighted.index(True) + 4 < len(highlighted):
                score = -len(tokens)
                if score > best_score:
                    best_score = score
                    best_s = s
        if best_s is None:
            continue
        yield best_s, bitoken
