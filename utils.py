def tokenize_and_highlight(tokenizer, text, bitoken) -> tuple[list[int], list[bool]]:
    tokens = tokenizer.encode(text)
    bitoken_tokens = tokenizer.encode(' ' + bitoken)
    # A token is "highlighted" if it is part of the bitoken
    highlighted = [False] * len(tokens)
    for i in range(len(tokens) - len(bitoken_tokens) + 1):
        if tokens[i:i+len(bitoken_tokens)] == bitoken_tokens:
            for j in range(len(bitoken_tokens)):
                highlighted[i+j] = True
    return tokens, highlighted
