from datasets import load_dataset
from transformers import AutoTokenizer

class VioDatasetLoader:
    def __init__(self, tokenizer_name='bert-base-uncased', dataset_name='ag_news', split='train', text_column='text', label_column='label'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset_name = dataset_name
        self.split = split
        self.text_column = text_column
        self.label_column = label_column
        self.dataset = self._load_and_tokenize_dataset()

    def _load_and_tokenize_dataset(self):
        raw_dataset = load_dataset(self.dataset_name, split=self.split)
        tokenized_dataset = raw_dataset.map(self._tokenize_function, batched=True)
        return tokenized_dataset

    def _tokenize_function(self, example):
        return self.tokenizer(example[self.text_column], truncation=True, padding='max_length')

    def get_dataset(self):
        return self.dataset

if __name__ == "__main__":
    loader = VioDatasetLoader()
    dataset = loader.get_dataset()
    print(dataset[0])
