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

EPOCHS                  = 20
BATCH_SIZE              = 160
LEARNING_RATE           = 1e-4
END_LEARNING_RATE       = 1e-5
MAX_SEQ_LENGTH          = 128
RANDOM_MASK_INPUT       = 0.1
POSITION_ENCODING_TYPE  = 'Learned'
# "None", "Sinusoidal", "Learned"
LAYERNORM               = True
# "POSITION_ENCODING_TYPE"-"LAYERNORM"-"RANDOM_MASK_INPUT"
RECORDS_PREFIX          = f"{POSITION_ENCODING_TYPE}-{LAYERNORM}-{RANDOM_MASK_INPUT}"
RECORDS_NAME            = f"{RECORDS_PREFIX}.pkl"
RECORDS_PATH            = f'./records/{RECORDS_PREFIX}'
RECORDS_MODELS_PATH     = os.path.join(RECORDS_PATH, 'models')
os.makedirs(RECORDS_PATH, exist_ok = True)
os.makedirs(RECORDS_MODELS_PATH, exist_ok = True)
def SET_SEED(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SET_SEED(42)
TRAIN_DATASETS = IMDB_DATASETS(DATAROOT = 'IMDB-Dataset.csv', MODE = 'train', MAX_LENGTH = MAX_SEQ_LENGTH, RANDOM_MASK_INPUT = RANDOM_MASK_INPUT)
TEST_DATASETS  = IMDB_DATASETS(DATAROOT = 'IMDB-Dataset.csv', MODE = 'test', MAX_LENGTH = MAX_SEQ_LENGTH, RANDOM_MASK_INPUT = RANDOM_MASK_INPUT)
TRAIN_LOADER   = DataLoader(TRAIN_DATASETS, batch_size = BATCH_SIZE, shuffle = True)
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
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = optim.AdamW(MODEL.parameters(), lr = LEARNING_RATE)
SCHEDULER = optim.lr_scheduler.CosineAnnealingLR(OPTIMIZER, T_max = EPOCHS, eta_min = END_LEARNING_RATE)

RECORDS = {
    "TRAIN_LOSSES"  : [],
    "TEST_LOSSES"   : [],
    "TRAIN_ACCS"    : [],
    "TEST_ACCS"     : [],
    "TRAIN_BATCH_LOSSES" : [],
    "TEST_BATCH_LOSSES"  : [],
    "TRAIN_BATCH_ACCS"   : [],
    "TEST_BATCH_ACCS"    : []
}
MIN_LOSS = float('inf')
MAX_ACC  = 0.0

for EPOCH in range(1, EPOCHS + 1):
    MODEL.train()
    ONE_EPOCH_LOSS, ONE_EPOCH_ACC = 0.0, 0.0
    for BATCH_IDX, (INPUTS, ATTENTION_MASKS, LABELS) in enumerate(tqdm(TRAIN_LOADER)):
        INPUTS, ATTENTION_MASKS, LABELS = INPUTS.cuda(), ATTENTION_MASKS.cuda(), LABELS.cuda()
        INPUTS = INPUTS.squeeze(1)
        OPTIMIZER.zero_grad()
        OUTPUTS = MODEL(INPUTS, ATTENTION_MASKS)
        LOSS = CRITERION(OUTPUTS, LABELS)
        LOSS.backward()
        OPTIMIZER.step()
        ONE_EPOCH_LOSS += LOSS.item() * INPUTS.size(0)
        ONE_EPOCH_ACC += (OUTPUTS.argmax(dim=-1) == LABELS).sum().item()
        RECORDS["TRAIN_BATCH_LOSSES"].append(LOSS.item())
        RECORDS["TRAIN_BATCH_ACCS"].append((OUTPUTS.argmax(dim=-1) == LABELS).sum().item() / INPUTS.size(0))
        if BATCH_IDX % 100 == 0:
            tqdm.write(f'Batch [{BATCH_IDX}/{len(TRAIN_LOADER)}], Loss: {LOSS.item():.4f}, Accuracy: {(OUTPUTS.argmax(dim=-1) == LABELS).sum().item() / INPUTS.size(0):.4f}')
    ONE_EPOCH_LOSS /= len(TRAIN_DATASETS)
    ONE_EPOCH_ACC /= len(TRAIN_DATASETS)
    RECORDS["TRAIN_LOSSES"].append(ONE_EPOCH_LOSS)
    RECORDS["TRAIN_ACCS"].append(ONE_EPOCH_ACC)
    tqdm.write(f'Train Epoch [{EPOCH}/{EPOCHS}], Loss: {ONE_EPOCH_LOSS:.4f}, Accuracy: {ONE_EPOCH_ACC:.4f}')

    MODEL.eval()
    ONE_EPOCH_LOSS, ONE_EPOCH_ACC = 0.0, 0.0
    with torch.no_grad():
        for BATCH_IDX, (INPUTS, ATTENTION_MASKS, LABELS) in enumerate(tqdm(TEST_LOADER)):
            INPUTS, ATTENTION_MASKS, LABELS = INPUTS.cuda(), ATTENTION_MASKS.cuda(), LABELS.cuda()
            INPUTS = INPUTS.squeeze(1)
            OUTPUTS = MODEL(INPUTS, ATTENTION_MASKS)
            LOSS = CRITERION(OUTPUTS, LABELS)
            ONE_EPOCH_LOSS += LOSS.item() * INPUTS.size(0)
            ONE_EPOCH_ACC += (OUTPUTS.argmax(dim=-1) == LABELS).sum().item()
            RECORDS["TEST_BATCH_LOSSES"].append(LOSS.item())
            RECORDS["TEST_BATCH_ACCS"].append((OUTPUTS.argmax(dim=-1) == LABELS).sum().item() / INPUTS.size(0))

    ONE_EPOCH_LOSS /= len(TEST_DATASETS)
    ONE_EPOCH_ACC /= len(TEST_DATASETS)
    RECORDS["TEST_LOSSES"].append(ONE_EPOCH_LOSS)
    RECORDS["TEST_ACCS"].append(ONE_EPOCH_ACC)
    tqdm.write(f'Test Epoch [{EPOCH}/{EPOCHS}], Loss: {ONE_EPOCH_LOSS:.4f}, Accuracy: {ONE_EPOCH_ACC:.4f}')
    if ONE_EPOCH_LOSS < MIN_LOSS:
        MIN_LOSS = ONE_EPOCH_LOSS
        torch.save(MODEL.state_dict(), os.path.join(RECORDS_MODELS_PATH, f'best_loss.pth'))
    if ONE_EPOCH_ACC > MAX_ACC:
        MAX_ACC = ONE_EPOCH_ACC
        torch.save(MODEL.state_dict(), os.path.join(RECORDS_MODELS_PATH, f'best_acc.pth'))
    SCHEDULER.step()
with open(os.path.join(RECORDS_PATH, RECORDS_NAME), 'wb') as f:
    pickle.dump(RECORDS, f)
    

