
import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample(boundaries: torch.Tensor, hidden: torch.Tensor, null_group: torch.Tensor):
    B, L = boundaries.shape
    _, _, D = hidden.shape

    boundaries = boundaries.to(dtype=torch.long).clone()  # [B, L]

    # Number of segments per example and across the batch
    seg_counts = boundaries.sum(dim=1)                    # [B]
    S = int(seg_counts.max().item())

    # If no segments at all in the batch, return a single null segment
    if S == 0:
        # shape [1, B, D]
        return null_group.expand(1, B, D).to(hidden.dtype).to(hidden.device)

    # Build [B, L, S] template of segment indices 0..S-1
    seg_ids = torch.arange(S, device=boundaries.device).view(1, 1, S)        # [1,1,S]
    seg_ids = seg_ids.expand(B, L, S)                                        # [B,L,S]

    # Segment index for each token position: 0,0,0,1,1,2,... (per-example)
    # cumulative_num_boundaries counts boundaries up to and including pos i
    cumulative = boundaries.cumsum(dim=1)                                    # [B,L]
    real_segment_index = cumulative - boundaries                             # [B,L]

    # One-hot membership mask: token at (b, l) belongs to segment k iff k == real_segment_index[b,l]
    membership = (real_segment_index.unsqueeze(-1) == seg_ids).to(hidden.dtype)  # [B,L,S]

    # Normalize over L so each segment’s weights sum to 1
    denom = membership.sum(dim=1, keepdim=True).clamp_min(1e-9)              # [B,1,S]
    weights = membership / denom                                             # [B,L,S]

    # Weighted average over tokens -> [S, B, D]
    shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, weights)
    return shortened_hidden


######## CLIP pretraining uses the old downsampling code below ###########
##########################################################################
# def final(foo,
#           upsample):
#     """
#         Input:
#             B x L x S
#     """
#     autoregressive = foo != 0
#     lel = 1 - foo

#     lel[autoregressive] = 0

#     dim = 2 if upsample else 1

#     lel = lel / (lel.sum(dim=dim, keepdim=True) + 1e-9)

#     return lel

# def common(boundaries, upsample=False):
#     boundaries = boundaries.clone()

#     n_segments = boundaries.sum(dim=-1).max().item()

#     if upsample:
#         n_segments += 1

#     if n_segments == 0:
#         return None

#     tmp = torch.zeros_like(
#         boundaries
#     ).unsqueeze(2) + torch.arange(
#         start=0,
#         end=n_segments,
#         device=boundaries.device
#     )

#     hh1 = boundaries.cumsum(1)

#     if not upsample:
#         hh1 -= boundaries

#     foo = tmp - hh1.unsqueeze(-1)

#     return foo

# def downsample(boundaries, hidden, null_group):
#     """
#         Downsampling

#         - The first element of boundaries tensor is always 0 and doesn't matter
#         - 1 starts a new group
#         - We append an extra "null" group at the beginning
#         - We discard last group because it won't be used (in terms of upsampling)

#         Input:
#             boundaries: B x L
#             hidden: L x B x D
#         Output:
#             shortened_hidden: S x B x D
#     """

#     foo = common(boundaries, upsample=False)  # B x L x S

#     if foo is None:
#         return null_group.repeat(1, hidden.size(1), 1)
#     else:
#         bar = final(foo=foo, upsample=False)  # B x L x S

#         bar = bar.to(hidden.dtype)  # ensure same dtype

#         shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, bar)
#         shortened_hidden = torch.cat(
#             [null_group.repeat(1, hidden.size(1), 1), shortened_hidden], dim=0
#         )

#         return shortened_hidden

@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float) -> torch.Tensor:
    return alpha * (tensor1 + tensor2)

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm, activation_function):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output

class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
    ):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        del activation_function

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(self.d_model, 3 * n_head * d_head)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.pre_lnorm:
            w_head_q, w_head_k, w_head_v = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)

        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)      # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', rr_head_q, r_head_k)       # bsz x n_head x qlen x klen
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))
        else:
            pass
        
        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        pre_lnorm,
        activation_function,
    ):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm,
            activation_function,
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output

########### This is the version used for CLIP pretraining ###########
#####################################################################
#####################################################################
class BoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner, activation_function,
                 temp, prior, bp_type, threshold=0.5,
                 image_size=None, patch_size=None, embed_dim=None):
        super().__init__()

        self.temp = temp
        self.prior = prior
        self.bp_type = bp_type
        self.threshold = threshold
        self.compression_rate = prior
        self.embed_dim = embed_dim
        if image_size is not None and patch_size is not None:
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_patches = (image_size // patch_size) ** 2

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, hidden):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]

        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)

        if self.bp_type == 'gumbel':
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=boundary_probs,
            )

            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (
                hard_boundaries - soft_boundaries.detach() + soft_boundaries
            )
        elif self.bp_type in ['entropy', 'unigram']:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        return soft_boundaries, hard_boundaries

    def calc_loss(self, preds):
        # B x T
        total_count = preds.size(-1)
        target_count = preds.sum(dim=-1)
        binomial = torch.distributions.binomial.Binomial(
            total_count=total_count,
            probs=torch.Tensor([self.prior]).to(preds.device)
        )
        loss_boundaries = -binomial.log_prob(target_count).mean() / total_count
        return loss_boundaries

class DTPViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=(2, 8, 0),
                 num_heads=12,
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 temp=1.0,
                 compression_rate=0.5,
                 bp_type='gumbel',
                 threshold=0.5,
                 num_classes=1000,
                 activation_function='gelu',
                 flop_measure: bool = False,
        ):

        super().__init__()
        self.flop_measure = flop_measure
        self.prior = compression_rate
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_len = self.num_patches

        # patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(drop_rate)

        # positional embedding
        self.pos_emb = PositionalEmbedding(embed_dim)
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        
        def create_decoder_layers(n_layers):
            layers = nn.ModuleList(
                [
                    RelPartialLearnableDecoderLayer(
                        n_head=num_heads,
                        d_model=embed_dim,
                        d_head=embed_dim // num_heads,
                        d_inner=int(embed_dim * mlp_ratio),
                        dropout=drop_rate,
                        dropatt=attn_drop_rate,
                        pre_lnorm=False,
                        activation_function=activation_function,
                    )
                    for _ in range(n_layers)
                ]
            )

            return layers

        # pre-pooling block
        self.pre_blocks = create_decoder_layers(depth[0])

        # post-pooling block
        self.short_blocks = create_decoder_layers(depth[1])

        # boundary predictor
        self.boundary_predictor = BoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function=activation_function,
            temp=temp,
            prior=compression_rate,
            bp_type=bp_type,
            threshold=threshold
        )

        # layer normalization
        self.down_ln = nn.LayerNorm(embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

        # final projection
        self.num_classes = num_classes
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward_after_pooling_with_attn_masks(self, core_input: torch.Tensor, layers, attention_mask: torch.Tensor):
        """
        Process input with relative attention and padding-aware masking.
        """
        T, _, _ = core_input.size()

        # Compute position embeddings
        pos_seq = torch.arange(T - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype)
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.dropout(pos_emb)

        core_out = core_input
        for layer in layers:
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=attention_mask)
        return core_out

    def encode(self, x: torch.Tensor, return_loss: bool = False):
        """
        Encode input image to feature sequence without final pooling.
        Returns:
            features OR (features, boundary_loss, avg_boundaries, boundary_ratio)
        """
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)                  # B x C x H' x W'
        x = x.flatten(2).transpose(1, 2)         # B x L x C
        x = self.dropout(x)                      # B x L x C

        # Positional embedding (for pre-blocks)
        pos_seq = torch.arange(self.seq_len - 1, -1, -1.0,
                            device=x.device, dtype=x.dtype)
        r = self.pos_emb(pos_seq)                # L x 1 x C

        # Pre-pooling transformer blocks
        x = x.transpose(0, 1)                    # L x B x C
        for block in self.pre_blocks:
            x = block(x, r, self.r_w_bias, self.r_r_bias)

        # boundary prediction
        if self.flop_measure:
            # Simulate hard boundaries for FLOP measurement
            L = x.size(0)
            num_tokens_to_keep = max(1, int(L * self.prior))
            indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep).round().long()
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, indices] = 1
        else:
            _, hard_boundaries = self.boundary_predictor(x)  # B x L

        # Downsampling (Dynamic Token Pooling)
        hidden = self.down_ln(x)               # L x B x D
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        )                                        # S x B x D

        # attention mask for post-pooling transformer layers
        S = shortened_hidden.size(0)
        pad_mask = shortened_hidden.abs().sum(-1).eq(0)       # S x B (1 where padded, 0 where regular)

        attn_mask = pad_mask.transpose(0, 1).unsqueeze(1)     # B x 1 x S
        attn_mask = attn_mask.expand(B, S, S)                 # B x S x S

        # post-pooling transformer blocks
        shortened_hidden = self.forward_after_pooling_with_attn_masks(
            shortened_hidden,
            self.short_blocks,
            attention_mask=attn_mask
        )

        # return features and optional loss
        features = shortened_hidden  # S x B x D

        if return_loss and not self.flop_measure:
            # Binomial boundary loss (no need for mask since all sequences have the same number of tokens)
            boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return features

    def forward(self, x, return_loss=False):
        """
        Full forward pass including pooling to class logits.
        """
        features_out = self.encode(x, return_loss=return_loss)

        if return_loss and not self.flop_measure:
            # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
            x, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
        else:
            x = features_out

        # pool across sequence dimension with mean pooling
        pad_mask = x.abs().sum(-1).eq(0).float()           # S x B
        valid_mask = 1.0 - pad_mask                        # S x B
        valid_mask_exp = valid_mask.unsqueeze(-1)          # S x B x 1

        x = x * valid_mask_exp                             # Mask padded tokens
        sum_x = x.sum(dim=0)                               # B x D
        valid_counts = valid_mask.sum(dim=0).clamp(min=1e-6).unsqueeze(-1)  # B x 1
        x = sum_x / valid_counts                           # B x D (masked mean)

        logits = self.head(x)

        if return_loss and not self.flop_measure:
            return logits, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return logits

class HierarchicalDTPViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=(2, 4, 4),
                 num_heads=12,
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 temp=1.0,
                 compression_rate=(0.5, 0.5),  # compression at stage 1 and 2
                 bp_type='gumbel',
                 threshold=0.5,
                 num_classes=1000,
                 activation_function='gelu',
                 flop_measure: bool = False,
        ):
        super().__init__()
        self.flop_measure = flop_measure
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_len = self.num_patches

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(drop_rate)

        # Positional embedding
        self.pos_emb = PositionalEmbedding(embed_dim)
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))

        # Helper to create decoder layers
        def create_decoder_layers(n_layers):
            return nn.ModuleList(
                [
                    RelPartialLearnableDecoderLayer(
                        n_head=num_heads,
                        d_model=embed_dim,
                        d_head=embed_dim // num_heads,
                        d_inner=int(embed_dim * mlp_ratio),
                        dropout=drop_rate,
                        dropatt=attn_drop_rate,
                        pre_lnorm=False,
                        activation_function=activation_function,
                    )
                    for _ in range(n_layers)
                ]
            )

        # Transformer blocks for each stage
        self.pre_blocks = create_decoder_layers(depth[0])    # before 1st pooling
        self.mid_blocks = create_decoder_layers(depth[1])    # between 1st and 2nd pooling
        self.final_blocks = create_decoder_layers(depth[2])  # after 2nd pooling

        # Two-stage boundary predictors
        self.bp1 = BoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function=activation_function,
            temp=temp,
            prior=compression_rate[0],
            bp_type=bp_type,
            threshold=threshold
        )

        self.bp2 = BoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function=activation_function,
            temp=temp,
            prior=compression_rate[1],
            bp_type=bp_type,
            threshold=threshold
        )

        # Layer norm
        self.down_ln = nn.LayerNorm(embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

        # Final classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_after_pooling_with_attn_masks(self, core_input: torch.Tensor, layers, attention_mask: torch.Tensor):
        """
        Process input with relative attention and padding-aware masking.
        """
        T, _, _ = core_input.size()
        pos_seq = torch.arange(T - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype)
        pos_emb = self.dropout(self.pos_emb(pos_seq))

        core_out = core_input
        for layer in layers:
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=attention_mask)
        return core_out

    def _make_attn_mask(self, seq: torch.Tensor):
        """
        seq: S x B x D
        return: B x S x S mask (True=mask)
        """
        S, B, _ = seq.size()
        pad_mask = seq.abs().sum(-1).eq(0)  # S x B
        attn_mask = pad_mask.transpose(0, 1).unsqueeze(1).expand(B, S, S)  # B x S x S
        return attn_mask

    def _downsample_stage(self, x, boundary_predictor):
        """
        One stage of boundary prediction + downsampling
        """
        B = x.size(1)
        L = x.size(0)
        hidden = self.down_ln(x)

        # boundary prediction
        if self.flop_measure:
            num_tokens_to_keep = max(1, int(L * boundary_predictor.prior))
            indices = torch.arange(0, L, step=max(1, L // num_tokens_to_keep), device=x.device)
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, indices] = 1
        else:
            _, hard_boundaries = boundary_predictor(x)  # B x L

        # downsample
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        )  # S x B x D

        return shortened_hidden, hard_boundaries

    def encode(self, x: torch.Tensor, return_loss: bool = False):
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # B x C x H' x W'
        x = x.flatten(2).transpose(1, 2)  # B x L x C
        x = self.dropout(x).transpose(0, 1)  # L x B x C

        # Stage 0: pre-blocks
        pos_seq = torch.arange(self.seq_len - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        r = self.pos_emb(pos_seq)  # L x 1 x C
        for block in self.pre_blocks:
            x = block(x, r, self.r_w_bias, self.r_r_bias)

        # Stage 1: 1st downsampling
        x, hard_boundaries1 = self._downsample_stage(x, self.bp1)
        attn_mask1 = self._make_attn_mask(x)

        # Stage 1 blocks
        x = self.forward_after_pooling_with_attn_masks(x, self.mid_blocks, attention_mask=attn_mask1)

        # Stage 2: 2nd downsampling
        x, hard_boundaries2 = self._downsample_stage(x, self.bp2)
        attn_mask2 = self._make_attn_mask(x)

        # Stage 2 blocks (final)
        x = self.forward_after_pooling_with_attn_masks(x, self.final_blocks, attention_mask=attn_mask2)

        features = x  # S x B x D

        if return_loss and not self.flop_measure:
            loss1 = self.bp1.calc_loss(hard_boundaries1)
            loss2 = self.bp2.calc_loss(hard_boundaries2)
            boundary_loss = loss1 + loss2
            avg_boundaries_per_batch1 = hard_boundaries1.sum(dim=1).float().mean().item()
            avg_boundaries_per_batch2 = hard_boundaries2.sum(dim=1).float().mean().item()

            boundary_ratio1 = avg_boundaries_per_batch1 / hard_boundaries1.size(1)
            boundary_ratio2 = avg_boundaries_per_batch2 / hard_boundaries2.size(1)

            # only report the second boundary ratio
            # this is not really that helpful, but we keep it for consistency
            cumulative_avg_boundaries_per_batch = avg_boundaries_per_batch2

            # compute the cumulative boundary ratio (e.g., 0.5 * 0.5 = 0.25)
            # NOTE: this is really important to monitor the cumulative compression ratio!
            cumulative_boundary_ratio = boundary_ratio1 * boundary_ratio2 

            return features, boundary_loss, cumulative_avg_boundaries_per_batch, cumulative_boundary_ratio
        else:
            return features

    def forward(self, x, return_loss=False):
        """
        Full forward pass including pooling to class logits.
        """
        features_out = self.encode(x, return_loss=return_loss)

        if return_loss and not self.flop_measure:
            # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
            x, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
        else:
            x = features_out

        # pool across sequence dimension with mean pooling
        pad_mask = x.abs().sum(-1).eq(0).float()           # S x B
        valid_mask = 1.0 - pad_mask                        # S x B
        valid_mask_exp = valid_mask.unsqueeze(-1)          # S x B x 1

        x = x * valid_mask_exp                             # Mask padded tokens
        sum_x = x.sum(dim=0)                               # B x D
        valid_counts = valid_mask.sum(dim=0).clamp(min=1e-6).unsqueeze(-1)  # B x 1
        x = sum_x / valid_counts                           # B x D (masked mean)

        logits = self.head(x)

        if return_loss and not self.flop_measure:
            return logits, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return logits


class SoftBoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner, activation_function,
                 temp, prior_upper_bound, prior_lower_bound, bp_type, threshold=0.5,
                 image_size=None, patch_size=None, embed_dim=None):
        super().__init__()

        self.temp = temp
        self.prior_upper_bound = prior_upper_bound
        self.prior_lower_bound = prior_lower_bound
        self.bp_type = bp_type
        self.threshold = threshold
        self.embed_dim = embed_dim
        if image_size is not None and patch_size is not None:
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_patches = (image_size // patch_size) ** 2

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, hidden):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]

        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)

        if self.bp_type == 'gumbel':
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=boundary_probs,
            )

            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (
                hard_boundaries - soft_boundaries.detach() + soft_boundaries
            )
        elif self.bp_type in ['entropy', 'unigram']:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        return soft_boundaries, hard_boundaries

    def calc_loss(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Penalize boundary rates only if outside the [prior_lower_bound, prior_upper_bound] interval.
        preds: B x T hard boundary tensor (0/1)
        """
        # Compute boundary rate per batch
        boundary_rate = preds.float().mean(dim=1)  # B

        # Compute penalty only outside the interval
        upper_violation = (boundary_rate - self.prior_upper_bound).clamp(min=0)
        lower_violation = (self.prior_lower_bound - boundary_rate).clamp(min=0)

        # Loss = mean squared deviation outside interval
        loss = (lower_violation + upper_violation).mean()
        return loss

    def calc_stats(self, preds, gt):
        # B x T
        preds, gt = preds.bool(), gt.bool()
        TP = ((preds == gt) & preds).sum().item()
        FP = ((preds != gt) & preds).sum().item()
        FN = ((preds != gt) & (~preds)).sum().item()

        acc = (preds == gt).sum().item() / gt.numel()

        if TP == 0:
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

        stats = {"acc": acc, "precision": precision, "recall": recall}

        return stats


class SoftDTPViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=(2, 8, 0),
                 num_heads=12,
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 temp=1.0,
                 compression_rate=(0.4, 0.6),
                 bp_type='gumbel',
                 threshold=0.5,
                 num_classes=1000,
                 activation_function='gelu',
                 flop_measure: bool = False,
        ):

        super().__init__()
        self.flop_measure = flop_measure
        self.prior = compression_rate
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_len = self.num_patches

        # patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(drop_rate)

        # positional embedding
        self.pos_emb = PositionalEmbedding(embed_dim)
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        
        def create_decoder_layers(n_layers):
            layers = nn.ModuleList(
                [
                    RelPartialLearnableDecoderLayer(
                        n_head=num_heads,
                        d_model=embed_dim,
                        d_head=embed_dim // num_heads,
                        d_inner=int(embed_dim * mlp_ratio),
                        dropout=drop_rate,
                        dropatt=attn_drop_rate,
                        pre_lnorm=False,
                        activation_function=activation_function,
                    )
                    for _ in range(n_layers)
                ]
            )

            return layers

        # pre-pooling block
        self.pre_blocks = create_decoder_layers(depth[0])

        # post-pooling block
        self.short_blocks = create_decoder_layers(depth[1])

        self.prior_lower_bound = compression_rate[0]
        self.prior_upper_bound = compression_rate[1]

        # boundary predictor
        self.boundary_predictor = SoftBoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function=activation_function,
            prior_lower_bound=self.prior_lower_bound,
            prior_upper_bound=self.prior_upper_bound,
            temp=temp,
            bp_type=bp_type,
            threshold=threshold
        )

        # layer normalization and null token
        self.down_ln = nn.LayerNorm(embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

        # final projection
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward_after_pooling_with_attn_masks(self, core_input: torch.Tensor, layers, attention_mask: torch.Tensor):
        """
        Process input with relative attention and padding-aware masking.
        """
        T, _, _ = core_input.size()

        # Compute position embeddings
        pos_seq = torch.arange(T - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype)
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.dropout(pos_emb)

        core_out = core_input
        for layer in layers:
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=attention_mask)
        return core_out

    def encode(self, x: torch.Tensor, return_loss: bool = False):
        """
        Encode input image to feature sequence without final pooling.
        Returns:
            features OR (features, boundary_loss, avg_boundaries, boundary_ratio)
        """
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)                  # B x C x H' x W'
        x = x.flatten(2).transpose(1, 2)         # B x L x C
        x = self.dropout(x)                      # B x L x C

        # Positional embedding (for pre-blocks)
        pos_seq = torch.arange(self.seq_len - 1, -1, -1.0,
                            device=x.device, dtype=x.dtype)
        r = self.pos_emb(pos_seq)                # L x 1 x C

        # Pre-pooling transformer blocks
        x = x.transpose(0, 1)                    # L x B x C
        for block in self.pre_blocks:
            x = block(x, r, self.r_w_bias, self.r_r_bias)

        # boundary prediction
        if self.flop_measure:
            # Simulate hard boundaries for FLOP measurement
            L = x.size(0)
            num_tokens_to_keep = max(1, int(L * (self.prior_lower_bound + self.prior_upper_bound) / 2))
            indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep).round().long()
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, indices] = 1
        else:
            _, hard_boundaries = self.boundary_predictor(x)  # B x L

        # Downsampling (Dynamic Token Pooling)
        hidden = self.down_ln(x)               # L x B x D
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        )                                        # S x B x D

        # attention mask for post-pooling transformer layers
        S = shortened_hidden.size(0)
        pad_mask = shortened_hidden.abs().sum(-1).eq(0)       # S x B (1 where padded, 0 where regular)

        attn_mask = pad_mask.transpose(0, 1).unsqueeze(1)     # B x 1 x S
        attn_mask = attn_mask.expand(B, S, S)                 # B x S x S

        # post-pooling transformer blocks
        shortened_hidden = self.forward_after_pooling_with_attn_masks(
            shortened_hidden,
            self.short_blocks,
            attention_mask=attn_mask
        )

        # return features and optional loss
        features = shortened_hidden  # S x B x D

        if return_loss and not self.flop_measure:
            # Binomial boundary loss (no need for mask since all sequences have the same number of tokens)
            boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return features

    def forward(self, x, return_loss=False):
        """
        Full forward pass including pooling to class logits.
        """
        features_out = self.encode(x, return_loss=return_loss)

        if return_loss and not self.flop_measure:
            # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
            x, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
        else:
            x = features_out

        # pool across sequence dimension with mean pooling
        pad_mask = x.abs().sum(-1).eq(0).float()           # S x B
        valid_mask = 1.0 - pad_mask                        # S x B
        valid_mask_exp = valid_mask.unsqueeze(-1)          # S x B x 1

        x = x * valid_mask_exp                             # Mask padded tokens
        sum_x = x.sum(dim=0)                               # B x D
        valid_counts = valid_mask.sum(dim=0).clamp(min=1e-6).unsqueeze(-1)  # B x 1
        x = sum_x / valid_counts                           # B x D (masked mean)

        logits = self.head(x)

        if return_loss and not self.flop_measure:
            return logits, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return logits
