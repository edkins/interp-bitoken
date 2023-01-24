import nltk.corpus
import random
import re
import more_itertools
from typing import Callable
from collections import Counter

re_word = re.compile(r'^[a-zA-Z0-9].*$')
re_acceptable_word = re.compile(r'^[a-z]+$')

def word_join(words: list[str]) -> str:
    word_spaces = []
    for w in words:
        if len(word_spaces) > 0 and re_word.match(w):
            word_spaces.append(' ')
        word_spaces.append(w)
    return ''.join(word_spaces)

def get_bitokens(tokenizer) -> Counter[str]:
    c = nltk.corpus.brown
    print("Reading words")
    words = Counter()
    for w in c.words():
        if re_acceptable_word.match(w):
            words[w] += 1

    print("Reading monotokens")
    monotokens = set()
    for w in words.keys():
        ts = tokenizer.encode(' ' + w)
        if len(ts) == 1:
            monotokens.add(ts[0])
    print("Reading bitokens")
    bitokens = Counter()
    for w,count in words.items():
        if re_acceptable_word.match(w):
            ts = tokenizer.encode(' ' + w)
            if len(ts) == 2 and ts[0] in monotokens:
                bitokens[w] += count
    print(f"Done reading {len(bitokens)} bitokens")
    return bitokens

def get_sentences_containing(word: str) -> list[str]:
    c = nltk.corpus.brown
    result = []
    for s in c.sents():
        if word in s:
            result.append(word_join(s))
    return result
