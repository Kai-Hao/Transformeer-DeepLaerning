import torch
import torch.nn as nn
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, EMBEDDING_DIMENSIONS = 768, NUMBER_OF_HEADS = 12, DROP_RATE = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.QUERY       = nn.Linear(EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS)
        self.KEY         = nn.Linear(EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS)
        self.VALUE       = nn.Linear(EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS)

        self.NUM_HEADS   = NUMBER_OF_HEADS
        self.DEPTH       = EMBEDDING_DIMENSIONS // NUMBER_OF_HEADS
        self.DROPOUT     = nn.Dropout(DROP_RATE)

        self.OUTPUT      = nn.Linear(EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS)

    def forward(self, INPUT, ATTENTION_MASK = None):
        BATCH_SIZE, SEQ_LEN, _ = INPUT.shape

        # LINEAR PROJECTIONS
        Q = rearrange(self.QUERY(INPUT), 'b s (h d) -> b h s d', h = self.NUM_HEADS)
        K = rearrange(self.KEY(INPUT)  , 'b s (h d) -> b h s d', h = self.NUM_HEADS)
        V = rearrange(self.VALUE(INPUT), 'b s (h d) -> b h s d', h = self.NUM_HEADS)

        # SCALED DOT-PRODUCT ATTENTION
        ATTENTION_MAP = torch.matmul(Q, rearrange(K, 'b h s d -> b h d s')) / (self.DEPTH ** 0.5)
        # ATTENTION MAP SHAPE: (BATCH_SIZE, NUMBER_OF_HEADS, SEQ_LEN, SEQ_LEN)
        ATTENTION_MASK = ATTENTION_MASK.unsqueeze(dim = 1)
        ATTENTION_MAP = ATTENTION_MAP.masked_fill(ATTENTION_MASK == 0, float('-inf'))
        ATTENTION_MAP = torch.nn.functional.softmax(ATTENTION_MAP, dim = -1)
        ATTENTION_MAP = self.DROPOUT(ATTENTION_MAP)
        ATTENTION_OUT = torch.matmul(ATTENTION_MAP, V)
        # ATTENTION OUT SHAPE: (BATCH_SIZE, NUMBER_OF_HEADS, SEQ_LEN, DEPTH)
        ATTENTION_OUT = rearrange(ATTENTION_OUT, 'b h s d -> b s (h d)', h = self.NUM_HEADS)
        # ATTENTION OUT SHAPE: (BATCH_SIZE, SEQ_LEN, EMBEDDING_DIMENSIONS)
        ATTENTION_OUT = self.OUTPUT(ATTENTION_OUT)
        return ATTENTION_OUT

class FeedForwardNetwork(nn.Module):
    def __init__(self, EMBEDDING_DIMENSIONS = 768, FF_DIM = 3072, DROP_RATE = 0.1):
        super(FeedForwardNetwork, self).__init__()
        self.LINEAR1 = nn.Linear(EMBEDDING_DIMENSIONS, FF_DIM)
        self.RELU    = nn.GELU()
        self.LINEAR2 = nn.Linear(FF_DIM, EMBEDDING_DIMENSIONS)
        self.DROPOUT = nn.Dropout(DROP_RATE)

    def forward(self, X):
        X = self.LINEAR1(X)
        X = self.RELU(X)
        X = self.DROPOUT(X)
        X = self.LINEAR2(X)
        X = self.DROPOUT(X)
        return X

class TransformerBlock(nn.Module):
    def __init__(self, EMBEDDING_DIMENSIONS = 768, NUMBER_OF_HEADS = 12, FF_DIM = 3072, DROP_RATE = 0.1, LAYERNORM = True):
        super(TransformerBlock, self).__init__()
        self.SELF_ATTENTION = MultiHeadSelfAttention(EMBEDDING_DIMENSIONS, NUMBER_OF_HEADS, DROP_RATE)
        self.FEED_FORWARD   = FeedForwardNetwork(EMBEDDING_DIMENSIONS, FF_DIM, DROP_RATE)
        self.LAYER_NORM1    = nn.LayerNorm(EMBEDDING_DIMENSIONS) if LAYERNORM else nn.Identity()
        self.LAYER_NORM2    = nn.LayerNorm(EMBEDDING_DIMENSIONS) if LAYERNORM else nn.Identity()

    def forward(self, X, ATTENTION_MASK = None):
        ATTENTION_OUT = self.SELF_ATTENTION(X, ATTENTION_MASK)
        X             = self.LAYER_NORM1(X + ATTENTION_OUT)
        FF_OUT        = self.FEED_FORWARD(X)
        X             = self.LAYER_NORM2(X + FF_OUT)
        return X

# Embedding is use vector to represent words in a continuous vector space.
class EmbeddingLayer(nn.Module):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIMENSIONS = 768, POSITION_ENCODING = 'None', MAX_SEQ_LEN = 512):
        super(EmbeddingLayer, self).__init__()
        self.EMBEDDING = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSIONS)
        self.EMBEDDING_TYPE = POSITION_ENCODING
        if POSITION_ENCODING == 'Learned':
            self.POSITIONAL_ENCODING = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, EMBEDDING_DIMENSIONS))
        elif POSITION_ENCODING == 'Sinusoidal':
            self.POSITIONAL_ENCODING = torch.zeros(MAX_SEQ_LEN, EMBEDDING_DIMENSIONS)
            position = torch.arange(0, MAX_SEQ_LEN).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, EMBEDDING_DIMENSIONS, 2).float() * -(torch.log(torch.tensor(10000.0)) / EMBEDDING_DIMENSIONS))
            self.POSITIONAL_ENCODING[:, 0::2] = torch.sin(position * div_term)
            self.POSITIONAL_ENCODING[:, 1::2] = torch.cos(position * div_term)
            self.POSITIONAL_ENCODING = self.POSITIONAL_ENCODING.unsqueeze(0)
        elif POSITION_ENCODING == 'None':
            pass
        else:
            raise ValueError("POSITION_ENCODING must be 'Learned', 'Sinusoidal', or 'None'.")

    def forward(self, INPUT):
        EMBEDDED_INPUT = self.EMBEDDING(INPUT)
        if self.EMBEDDING_TYPE == 'Learned':
            return EMBEDDED_INPUT + self.POSITIONAL_ENCODING[:, :EMBEDDED_INPUT.size(1), :]
        elif self.EMBEDDING_TYPE == 'Sinusoidal':
            return EMBEDDED_INPUT + self.POSITIONAL_ENCODING[:, :EMBEDDED_INPUT.size(1), :].to(EMBEDDED_INPUT.device)
        else:
            return EMBEDDED_INPUT

class Transformer(nn.Module):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIMENSIONS = 768, NUMBER_OF_HEADS = 12, FFN_DIM = 3072, NUM_LAYERS = 6, DROP_RATE = 0.1, POSITION_ENCODING = 'None', LAYERNORM = True, MAX_SEQ_LEN = 512, NUM_LAYER = 12, NUM_CLASSES = 2):
        super(Transformer, self).__init__()
        self.EMBEDDING_LAYER = EmbeddingLayer(VOCAB_SIZE, EMBEDDING_DIMENSIONS, POSITION_ENCODING, MAX_SEQ_LEN)
        self.TRANSFORMER_BLOCKS = nn.ModuleList([
            TransformerBlock(EMBEDDING_DIMENSIONS, NUMBER_OF_HEADS, FFN_DIM, DROP_RATE, LAYERNORM) for _ in range(NUM_LAYERS)
        ])
        self.LAYER_NORM = nn.LayerNorm(EMBEDDING_DIMENSIONS) if LAYERNORM else nn.Identity()
        self.CLASSIFIER = nn.Linear(EMBEDDING_DIMENSIONS, NUM_CLASSES)
    def forward(self, INPUT, ATTENTION_MASK = None):
        EMBEDDED_INPUT = self.EMBEDDING_LAYER(INPUT)
        for BLOCK in self.TRANSFORMER_BLOCKS:
            EMBEDDED_INPUT = BLOCK(EMBEDDED_INPUT, ATTENTION_MASK)
        OUTPUT = self.LAYER_NORM(EMBEDDED_INPUT)
        OUTPUT = rearrange(OUTPUT, 'b s d -> b s 1 d')
        OUTPUT = self.CLASSIFIER(OUTPUT[:, 0, :])
        return OUTPUT.squeeze()

# TEST FUNCTIONS:
if __name__ == "__main__":
    # TEST MultiHeadSelfAttention 
    MHSA_BLOCK = MultiHeadSelfAttention()
    INPUT = torch.randn(2, 10, 768)  # (BATCH_SIZE, SEQ_LEN, EMBEDDING_DIMENSIONS)
    ATTENTION_MASK = torch.ones(2, 10, 10)  # (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    OUTPUT = MHSA_BLOCK(INPUT, ATTENTION_MASK)
    print("MultiHeadSelfAttention Input Shape:", INPUT.shape)  # Expected: (2, 10, 768)
    print("MultiHeadSelfAttention Output Shape:", OUTPUT.shape)  # Expected: (2, 10, 768)

    # TEST FeedForwardNetwork
    FFN_BLOCK = FeedForwardNetwork()
    INPUT = torch.randn(2, 10, 768)  # (BATCH_SIZE, SEQ_LEN, EMBEDDING_DIMENSIONS)
    OUTPUT = FFN_BLOCK(INPUT)
    print("FeedForwardNetwork Input Shape:", INPUT.shape)  # Expected: (2, 10, 768)
    print("FeedForwardNetwork Output Shape:", OUTPUT.shape)  # Expected: (2, 10, 768)

    # TEST TransformerBlock
    TRANSFORMER_BLOCK = TransformerBlock()
    INPUT = torch.randn(2, 10, 768)  # (BATCH_SIZE, SEQ_LEN, EMBEDDING_DIMENSIONS)
    ATTENTION_MASK = torch.ones(2, 10, 10)  # (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    OUTPUT = TRANSFORMER_BLOCK(INPUT, ATTENTION_MASK)
    print("TransformerBlock Input Shape:", INPUT.shape)  # Expected: (2, 10, 768)
    print("TransformerBlock Output Shape:", OUTPUT.shape)  # Expected: (2, 10, 768)

    # TEST EmbeddingLayer
    EMBEDDING_LAYER = EmbeddingLayer(VOCAB_SIZE=30522)  # BERT's vocab size
    INPUT = torch.randint(0, 30522, (2, 10))  # (BATCH_SIZE, SEQ_LEN)
    OUTPUT = EMBEDDING_LAYER(INPUT)
    print("EmbeddingLayer Input Shape:", INPUT.shape)  # Expected: (2, 10)
    print("EmbeddingLayer Output Shape:", OUTPUT.shape)  # Expected: (2, 10, 768)

    # TEST Transformer
    TRANSFORMER_MODEL = Transformer(VOCAB_SIZE=30522)
    INPUT = torch.randint(0, 30522, (2, 10))  # (BATCH_SIZE, SEQ_LEN)
    ATTENTION_MASK = torch.ones(2, 10, 10)  # (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    OUTPUT = TRANSFORMER_MODEL(INPUT, ATTENTION_MASK)
    print("Transformer Input Shape:", INPUT.shape)  # Expected: (2, 10)
    print("Transformer Output Shape:", OUTPUT.shape)  # Expected: (2, 2) for binary classification


