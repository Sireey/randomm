import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
import timm


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your config (example)


class ModelConfig:
    def __init__(self, tokenizer=None):
        self.embed_dim = 768
        self.num_heads = 12
        self.seq_len = 1024
        self.attention_dropout = 0.1
        self.residual_dropout = 0.1
        self.mlp_ratio = 4
        self.mlp_dropout = 0.1
        self.emb_dropout = 0.1
        self.vocab_size = 25000

        # Set vocab_size and eos_token_id based on tokenizer if provided
        if tokenizer:

            self.eos_token_id = tokenizer.eos_id
        else:
            # Default values (GPT2)
            self.vocab_size = 50257
            self.eos_token_id = 50256


class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.c_attn = nn.Linear(
            self.embed_dim, self.head_size * self.n_heads * 3, bias=True)
        self.scale = self.head_size ** -0.5

        # Register mask buffer that will be moved to the right device automatically
        self.register_buffer('mask', torch.tril(
            torch.ones(1, 1, self.seq_len, self.seq_len)))

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        qk_t = (q @ k.transpose(-2, -1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ v
        attention = attention.permute(0, 2, 1, 3).contiguous().view(b, t, c)

        out = self.c_proj(attention)
        out = self.resid_dropout(out)

        return out


class GPT2CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_size ** -0.5

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, q, k, v):
        b, t, c = q.shape

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(b, q.size(1), self.n_heads,
                   self.head_size).permute(0, 2, 1, 3)
        k = k.view(b, k.size(1), self.n_heads,
                   self.head_size).permute(0, 2, 1, 3)
        v = v.view(b, v.size(1), self.n_heads,
                   self.head_size).permute(0, 2, 1, 3)

        qk_t = (q @ k.transpose(-2, -1)) * self.scale
        qk_t = F.softmax(qk_t, dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ v
        attention = attention.permute(0, 2, 1, 3).contiguous().view(b, t, c)

        out = self.c_proj(attention)
        out = self.resid_dropout(out)

        return out


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout

        self.c_fc = nn.Linear(self.embed_dim, self.embed_dim * self.mlp_ratio)
        self.c_proj = nn.Linear(
            self.embed_dim * self.mlp_ratio, self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config)

    def forward(self, x, enc_out):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), enc_out, enc_out)
        x = x + self.mlp(self.ln_3(x))
        return x


class VisionGPT2Model(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        self.config = config

        # Set device with priority to GPU if available
        if device:
            self.device = device
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        vit = timm.create_model('vit_base_patch16_224',
                                pretrained=True, num_classes=0)
        self.patch_embed = vit.patch_embed
        num_patches = self.patch_embed.num_patches

        self.cls_token = vit.cls_token
        embed_len = num_patches + vit.num_prefix_tokens
        self.pos_embed = vit.pos_embed
        self.pos_drop = nn.Dropout(p=0.)

        self.blocks = nn.ModuleList([vit.blocks[i]
                                    for i in range(config.depth)])

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_dim),
            wpe=nn.Embedding(config.seq_len, config.embed_dim),
            drop=nn.Dropout(config.emb_dropout),
            h=nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f=nn.LayerNorm(config.embed_dim)
        ))
        self.lm_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Tie weights

        # Initialize embedding weights properly
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=0.02)

        # Move model to device
        self.to(self.device)

    def _pos_embed(self, x):
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def pretrained_layers_trainable(self, trainable=False):
        layers = [
            self.cls_token, self.patch_embed, self.pos_embed, self.blocks,
            self.transformer.wte, self.transformer.wpe,
            self.transformer.ln_f, self.lm_head
        ]
        gpt_layers = [[
            self.transformer.h[i].ln_1, self.transformer.h[i].ln_2,
            self.transformer.h[i].attn, self.transformer.h[i].mlp
        ] for i in range(self.config.depth)]
        for l in gpt_layers:
            layers.extend(l)

        for layer in layers:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable

        total_frozen_params = sum(
            [p.numel() for p in self.parameters() if not p.requires_grad])
        print(f'{total_frozen_params=}')

    def unfreeze_last_gpt_block(self):
        """Unfreezes the last GPT2 block."""
        for param in self.transformer.h[-1].parameters():
            param.requires_grad = True
        print("Unfreezing last GPT2 block.")

    def unfreeze_last_n_gpt_blocks(self, n):
        """Unfreezes the last 'n' GPT2 blocks."""
        for block in self.transformer.h[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Unfreezing last {n} GPT2 blocks.")

    def unfreeze_gpt_layers(self):
        """Unfreezes all GPT2 layers."""
        for block in self.transformer.h:
            for param in block.parameters():
                param.requires_grad = True
        print("Unfreezing all GPT2 layers.")

    def unfreeze_word_embeddings(self):
        """Unfreezes word embedding layer (important for custom tokenizers)."""
        for param in self.transformer.wte.parameters():
            param.requires_grad = True
        for param in self.lm_head.parameters():
            param.requires_grad = True
        print("Unfreezing word embedding layers.")

    @classmethod
    def from_pretrained(cls, config, device=None):
        # Add device parameter to from_pretrained
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        model = VisionGPT2Model(config, device=device)
        sd = model.state_dict()
        keys = sd.keys()
        ignore_matches = ['blocks.', 'cross_attn.', 'ln_3',
                          'cls_token', 'pos_embed', 'patch_embed.', '.attn.mask']
        vit_keys = [key for key in keys if any(
            match in key for match in ignore_matches)]
        gpt_keys = [key for key in keys if key not in vit_keys]

        try:
            # Add warning message for potential large download
            print("Downloading pretrained GPT2 weights (this might take a while)...")
            gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
            print("Download complete.")

            # Move the HF model to the same device
            gpt2_small = gpt2_small.to(device)

            sd_hf = gpt2_small.state_dict()
            hf_keys = sd_hf.keys()
            hf_keys = [k for k in hf_keys if not k.endswith(
                '.attn.masked_bias')]
            hf_keys = [k for k in hf_keys if not k.endswith('.attn.bias')]
            transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                          'mlp.c_fc.weight', 'mlp.c_proj.weight']

            for k in hf_keys:
                if any(match in k for match in ignore_matches):
                    continue

                # Skip embedding and output layers with shape mismatch (vocabulary differences)
                if 'wte.weight' in k or 'lm_head.weight' in k:
                    continue

                if any(k.endswith(w) for w in transposed):
                    if sd_hf[k].shape[::-1] == sd[k].shape:
                        with torch.no_grad():
                            sd[k].copy_(sd_hf[k].t())
                else:
                    if sd_hf[k].shape == sd[k].shape:
                        with torch.no_grad():
                            sd[k].copy_(sd_hf[k])

            model.load_state_dict(sd)

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Continuing with randomly initialized weights.")

        return model

    def forward(self, image, input_ids, attention_mask=None, labels=None):
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # ViT Processing
        image_features = self.patch_embed(image)
        image_features = self._pos_embed(image_features)
        for block in self.blocks:
            image_features = block(image_features)

        # GPT Embedding
        token_embeddings = self.transformer.wte(input_ids)
        positions = torch.arange(0, input_ids.size(1), device=self.device)
        pos_embeddings = self.transformer.wpe(positions)
        input_embedded = self.transformer.drop(
            token_embeddings + pos_embeddings)

        # GPT Blocks with cross-attention
        for block in self.transformer.h:
            input_embedded = block(input_embedded, image_features)

        input_embedded = self.transformer.ln_f(input_embedded)

        lm_logits = self.lm_head(input_embedded)

        if labels is not None:
            labels_copy = labels.clone()

            # Fix invalid negative labels
            labels_copy[(labels_copy < 0) & (labels_copy != -100)] = -100

            # Loss computation
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels_copy.view(-1))
            return loss

        return lm_logits  # Return logits for last token if no labels

    def generate(self, image, sequence, max_tokens=50, temperature=1.0, deterministic=False):
        """
        Generate text conditioned on image and prompt sequence

        Args:
            image: Image tensor [batch_size, channels, height, width]
            sequence: Token IDs tensor [batch_size, seq_len]
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            deterministic: If True, use argmax sampling instead of temperature

        Returns:
            Tensor of generated token IDs
        """
        # Cache the processed image to avoid recomputing for each token
        self.eval()  # Set model to evaluation mode

        # Ensure inputs are on the correct device
        image = image.to(self.device)
        sequence = sequence.to(self.device)

        # Process the image through ViT (can be cached for multiple generations)
        with torch.no_grad():
            image_features = self.patch_embed(image)
            image_features = self._pos_embed(image_features)

            # Process image through ViT blocks
            for block in self.blocks:
                image_features = block(image_features)

        # Generate tokens one by one
        with torch.no_grad():
            for _ in range(max_tokens):
                # Ensure sequence is the right shape
                if len(sequence.shape) == 1:
                    sequence = sequence.unsqueeze(0)

                # Forward pass with current sequence
                try:
                    # Explicitly pass labels=None to ensure correct method signature
                    logits = self.forward(image, sequence, labels=None)
                except Exception as e:
                    print(f"Forward pass error: {e}")
                    break

                # Handle different logits shapes
                if logits is None:
                    break

                # Ensure logits are 3D
                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(1)

                # Get probabilities for the next token
                try:
                    # Always take the last token's logits
                    next_token_logits = logits[:, -1, :] / temperature
                    probs = F.softmax(next_token_logits, dim=-1)

                    # Sample or take argmax
                    if deterministic:
                        next_token = torch.argmax(probs, dim=-1, keepdim=True)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)

                    # Append new token to sequence
                    sequence = torch.cat([sequence, next_token], dim=1)

                    # Stop if EOS token is generated
                    if next_token.item() == self.config.eos_token_id:
                        break

                except Exception as e:
                    print(f"Generation step error: {e}")
                    print(f"Logits shape: {logits.shape}")
                    break

        # Move result to CPU before returning
        return sequence.cpu()
