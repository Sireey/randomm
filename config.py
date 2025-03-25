class Config:
    def __init__(self, tokenizer=None):
        self.embed_dim = 768
        self.num_heads = 12
        self.seq_len = 512
        self.attention_dropout = 0.1
        self.residual_dropout = 0.1
        self.emb_dropout = 0.1
        self.mlp_ratio = 4
        self.mlp_dropout = 0.1
        self.depth = 12

        # Dynamically set vocab size and special token IDs
        if tokenizer:
            self.vocab_size = tokenizer.vocab_size
            self.bos_token_id = tokenizer.bos_id
            self.eos_token_id = tokenizer.eos_id
        else:
            # Defaults in case tokenizer is not passed
            self.vocab_size = 8766
            self.bos_token_id = 2
            self.eos_token_id = 3
