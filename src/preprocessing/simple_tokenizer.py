import re
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        for punct in '.,?!':
            text = text.replace(punct, f' {punct} ')
        tokens = re.split(r'\s+', text.strip())
        return [t for t in tokens if t]