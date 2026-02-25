import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(specials):
            self.word_to_id[token] = i
            self.id_to_word[i] = token

            
        unique = set()
        for text in texts:
            for word in text.split(" "):
                if word not in self.word_to_id:
                    unique.add(word)  

        for i, word in enumerate(sorted(list(unique))):
            new_id = i + 4
            self.word_to_id[word] = new_id
            self.id_to_word[new_id] = word
            
        self.vocab_size = len(self.word_to_id)
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        result = []
        for part in text.split(" "):
            if part in self.word_to_id:
                result.append(self.word_to_id[part])
            else:
                result.append(1)
            
        return result
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = [self.id_to_word.get(idx, self.unk_token) for idx in ids]
        return " ".join(words)
        pass
