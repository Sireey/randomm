class Config:
    def __init__(self):
        self.vocab_size = 8766  # Update this to the size of the Telugu vocabulary
        self.embed_dim = 768
        self.num_heads = 12
        self.seq_len = 512
        self.attention_dropout = 0.1
        self.residual_dropout = 0.1
        self.emb_dropout = 0.1
        self.mlp_ratio = 4
        self.mlp_dropout = 0.1
        self.depth = 12
        self.eos_token_id = 3  # Add EOS token ID if necessary