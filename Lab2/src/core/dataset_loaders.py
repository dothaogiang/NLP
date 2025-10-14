import os

def load_raw_text_data(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()