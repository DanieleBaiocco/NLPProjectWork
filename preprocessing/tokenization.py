class ExtendedBertTokenizer:
    """Extended tokenizer that wraps a base tokenizer."""

    def __init__(self, base_tokenizer):
        self.tokenizer = base_tokenizer

    def tokenize_with_label_extension(self, text, labels, copy_previous_label=False, extension_label='X'):
        tok_text = self.tokenizer.tokenize(text)
        for i in range(len(tok_text)):
            if '##' in tok_text[i]:
                if copy_previous_label:
                    labels.insert(i, labels[i - 1])
                else:
                    labels.insert(i, extension_label)
        return tok_text, labels

    def convert_tokens_to_ids(self, tokens):
      return self.tokenizer.convert_tokens_to_ids(tokens)