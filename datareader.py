import pandas as pd
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import random

class IMDB_DATASETS(Dataset):
    def __init__(self, DATAROOT = 'IMDB-Dataset.csv', TOKENIZER = "textattack/bert-base-uncased-imdb", MAX_LENGTH = 168, MODE = 'train', SPLIT = 0.9, RANDOM_MASK_INPUT = 0.15):
        self.DATA              = pd.read_csv(DATAROOT)
        self.DATA              = [{'REVIEW': ROW['review'], 'SENTIMENT': (0 if ROW['sentiment'] == 'negative' else 1)} for idx, ROW in self.DATA.iterrows()]
        self.MODE              = MODE
        self.MAX_LENGTH        = MAX_LENGTH
        self.TOKENIZER         = BertTokenizer.from_pretrained(TOKENIZER)
        self.RANDOM_MASK_INPUT = RANDOM_MASK_INPUT
        if MODE == 'train':
            self.data = self.DATA[:int(len(self.DATA) * SPLIT)]
        elif MODE == 'test':
            self.data = self.DATA[int(len(self.DATA) * SPLIT):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, INDEX):
        ENCODED = self.TOKENIZER(
            self.data[INDEX]['REVIEW'],
            padding        = 'max_length',
            truncation     = True,
            max_length     = self.MAX_LENGTH,
            return_tensors = 'pt'
        )
        # mask input tokens randomly
        if self.MODE == 'train' and self.RANDOM_MASK_INPUT > 0:
            INPUT_IDS = ENCODED['input_ids']
            ATTENTION_MASK = ENCODED['attention_mask']
            # Randomly mask some input tokens
            for i in range(INPUT_IDS.size(1)):
                if random.random() < self.RANDOM_MASK_INPUT:
                    INPUT_IDS[0][i] = self.TOKENIZER.mask_token_id
            ENCODED['input_ids']      = INPUT_IDS
            ENCODED['attention_mask'] = ATTENTION_MASK
        LABEL = self.data[INDEX]['SENTIMENT']
        return ENCODED['input_ids'], ENCODED['attention_mask'], LABEL

# TEST DATASET
if __name__ == "__main__":
    DATASETS = IMDB_DATASETS(RANDOM_MASK_INPUT = 0.0)
    TRAINLOADER = DataLoader(DATASETS, batch_size=2, shuffle=True)

    for BATCH_IDX, (INPUT_IDS, ATTENTION_MASK, LABEL) in enumerate(TRAINLOADER):
        print(INPUT_IDS.shape, ATTENTION_MASK.shape, LABEL.shape)