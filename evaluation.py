import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from model import Transformer
from datareader import IMDB_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm

import pickle

BATCH_SIZE              = 160
MAX_SEQ_LENGTH          = 128
RANDOM_MASK_INPUT       = 0.05
POSITION_ENCODING_TYPE  = 'Learned'
MODEL_SAVE_PATH         = './records/Learned-True-0.05/models/best_acc.pth'
LAYERNORM               = True
# './records/Sinusoidal-True-0/models/best_acc.pth' loss 1.6684 acc 0.8170
# './records/Sinusoidal-False-0/models/best_acc.pth' loss 1.6219 acc 0.8116
# './records/Learned-True-0/models/best_acc.pth' loss 0.3849 acc 0.8242
# './records/Learned-False-0/models/best_acc.pth' loss 0.4130 acc 0.8216
# './records/None-True-0/models/best_acc.pth' loss 0.3849 acc 0.8288
# './records/None-False-0/models/best_acc.pth' loss 0.3956 acc 0.8184
# Learned 0.15 Test Loss: 0.4301, Test Accuracy: 0.8070
# Learned 0.1 Test Loss: 0.4136, Test Accuracy: 0.8188
# Learned 0.05 Test Loss: 0.4209, Test Accuracy: 0.8246
TEST_DATASETS  = IMDB_DATASETS(DATAROOT = 'IMDB-Dataset.csv', MODE = 'test', MAX_LENGTH = MAX_SEQ_LENGTH, RANDOM_MASK_INPUT = RANDOM_MASK_INPUT)
TEST_LOADER    = DataLoader(TEST_DATASETS, batch_size = BATCH_SIZE, shuffle = False)

MODEL = Transformer(
    VOCAB_SIZE = 30522,  # BERT's vocab size
    EMBEDDING_DIMENSIONS = 768,
    NUMBER_OF_HEADS = 12,
    FFN_DIM = 3072,
    NUM_LAYERS = 6,
    DROP_RATE = 0.1,
    POSITION_ENCODING = POSITION_ENCODING_TYPE,
    LAYERNORM = LAYERNORM,
    MAX_SEQ_LEN = MAX_SEQ_LENGTH,
    NUM_LAYER = 12,
    NUM_CLASSES = 2
)
MODEL = MODEL.cuda()
MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH))
MODEL.eval()
CRITERION = nn.CrossEntropyLoss()
ONE_EPOCH_LOSS, ONE_EPOCH_ACC = 0.0, 0.0
with torch.no_grad():
    for BATCH_IDX, (INPUTS, ATTENTION_MASKS, LABELS) in enumerate(tqdm(TEST_LOADER)):
        INPUTS, ATTENTION_MASKS, LABELS = INPUTS.cuda(), ATTENTION_MASKS.cuda(), LABELS.cuda()
        INPUTS = INPUTS.squeeze(1)
        OUTPUTS = MODEL(INPUTS, ATTENTION_MASKS)
        LOSS = CRITERION(OUTPUTS, LABELS)
        ONE_EPOCH_LOSS += LOSS.item() * INPUTS.size(0)
        ONE_EPOCH_ACC += (OUTPUTS.argmax(dim=-1) == LABELS).sum().item()
    ONE_EPOCH_LOSS /= len(TEST_DATASETS)
    ONE_EPOCH_ACC /= len(TEST_DATASETS)
print(f"Test Loss: {ONE_EPOCH_LOSS:.4f}, Test Accuracy: {ONE_EPOCH_ACC:.4f}")