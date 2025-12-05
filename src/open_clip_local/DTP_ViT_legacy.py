
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import ResidualAttentionBlock
from .pos_embed import get_2d_sincos_pos_embed

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

# class DTPViT(nn.Module):
#     def __init__(self,
#                  image_size=224,
#                  patch_size=16,
#                  in_chans=3,
#                  embed_dim=768,
#                  depth=(2, 8, 0),
#                  num_heads=12,
#                  mlp_ratio=4.0,
#                  drop_rate=0.1,
#                  attn_drop_rate=0.1,
#                  temp=1.0,
#                  compression_rate=0.5,
#                  bp_type='gumbel',
#                  threshold=0.5,
#                  num_classes=1000,
#                  activation_function='gelu',
#                  flop_measure: bool = False,
#         ):

#         super().__init__()
#         self.flop_measure = flop_measure
#         self.prior = compression_rate
#         self.embed_dim = embed_dim
#         self.num_patches = (image_size // patch_size) ** 2
#         self.seq_len = self.num_patches

#         # patch embedding
#         self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.dropout = nn.Dropout(drop_rate)

#         # positional embedding
#         self.pos_emb = PositionalEmbedding(embed_dim)
#         self.r_w_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
#         self.r_r_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        
#         def create_decoder_layers(n_layers):
#             layers = nn.ModuleList(
#                 [
#                     RelPartialLearnableDecoderLayer(
#                         n_head=num_heads,
#                         d_model=embed_dim,
#                         d_head=embed_dim // num_heads,
#                         d_inner=int(embed_dim * mlp_ratio),
#                         dropout=drop_rate,
#                         dropatt=attn_drop_rate,
#                         pre_lnorm=False,
#                         activation_function=activation_function,
#                     )
#                     for _ in range(n_layers)
#                 ]
#             )

#             return layers

#         # pre-pooling block
#         self.pre_blocks = create_decoder_layers(depth[0])

#         # post-pooling block
#         self.short_blocks = create_decoder_layers(depth[1])

#         # boundary predictor
#         self.boundary_predictor = BoundaryPredictor(
#             d_model=embed_dim,
#             d_inner=int(embed_dim * mlp_ratio),
#             activation_function=activation_function,
#             temp=temp,
#             prior=compression_rate,
#             bp_type=bp_type,
#             threshold=threshold
#         )

#         # layer normalization
#         self.down_ln = nn.LayerNorm(embed_dim)
#         self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         nn.init.normal_(self.null_token, std=0.02)

#         # final projection
#         self.num_classes = num_classes
#         self.head = nn.Linear(embed_dim, num_classes)
    
#     def forward_after_pooling_with_attn_masks(self, core_input: torch.Tensor, layers, attention_mask: torch.Tensor):
#         """
#         Process input with relative attention and padding-aware masking.
#         """
#         T, _, _ = core_input.size()

#         # Compute position embeddings
#         pos_seq = torch.arange(T - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype)
#         pos_emb = self.pos_emb(pos_seq)
#         pos_emb = self.dropout(pos_emb)

#         core_out = core_input
#         for layer in layers:
#             core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=attention_mask)
#         return core_out

#     def encode(self, x: torch.Tensor, return_loss: bool = False):
#         """
#         Encode input image to feature sequence without final pooling.
#         Returns:
#             features OR (features, boundary_loss, avg_boundaries, boundary_ratio)
#         """
#         B = x.size(0)

#         # Patch embedding
#         x = self.patch_embed(x)                  # B x C x H' x W'
#         x = x.flatten(2).transpose(1, 2)         # B x L x C
#         x = self.dropout(x)                      # B x L x C

#         # Positional embedding (for pre-blocks)
#         pos_seq = torch.arange(self.seq_len - 1, -1, -1.0,
#                             device=x.device, dtype=x.dtype)
#         r = self.pos_emb(pos_seq)                # L x 1 x C

#         # Pre-pooling transformer blocks
#         x = x.transpose(0, 1)                    # L x B x C
#         for block in self.pre_blocks:
#             x = block(x, r, self.r_w_bias, self.r_r_bias)

#         # boundary prediction
#         if self.flop_measure:
#             # Simulate hard boundaries for FLOP measurement
#             L = x.size(0)
#             num_tokens_to_keep = max(1, int(L * self.prior))
#             indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep).round().long()
#             hard_boundaries = torch.zeros(B, L, device=x.device)
#             hard_boundaries[:, indices] = 1
#         else:
#             _, hard_boundaries = self.boundary_predictor(x)  # B x L

#         # Downsampling (Dynamic Token Pooling)
#         hidden = self.down_ln(x)               # L x B x D
#         shortened_hidden = downsample(
#             boundaries=hard_boundaries,
#             hidden=hidden,
#             null_group=self.null_token
#         )                                        # S x B x D

#         # attention mask for post-pooling transformer layers
#         S = shortened_hidden.size(0)
#         pad_mask = shortened_hidden.abs().sum(-1).eq(0)       # S x B (1 where padded, 0 where regular)

#         attn_mask = pad_mask.transpose(0, 1).unsqueeze(1)     # B x 1 x S
#         attn_mask = attn_mask.expand(B, S, S)                 # B x S x S

#         # post-pooling transformer blocks
#         shortened_hidden = self.forward_after_pooling_with_attn_masks(
#             shortened_hidden,
#             self.short_blocks,
#             attention_mask=attn_mask
#         )

#         # return features and optional loss
#         features = shortened_hidden  # S x B x D

#         if return_loss and not self.flop_measure:
#             # Binomial boundary loss (no need for mask since all sequences have the same number of tokens)
#             boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
#             avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
#             boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
#             return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
#         else:
#             return features

#     def forward(self, x, return_loss=False):
#         """
#         Full forward pass including pooling to class logits.
#         """
#         features_out = self.encode(x, return_loss=return_loss)

#         if return_loss and not self.flop_measure:
#             # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
#             x, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
#         else:
#             x = features_out

#         # pool across sequence dimension with mean pooling
#         pad_mask = x.abs().sum(-1).eq(0).float()           # S x B
#         valid_mask = 1.0 - pad_mask                        # S x B
#         valid_mask_exp = valid_mask.unsqueeze(-1)          # S x B x 1

#         x = x * valid_mask_exp                             # Mask padded tokens
#         sum_x = x.sum(dim=0)                               # B x D
#         valid_counts = valid_mask.sum(dim=0).clamp(min=1e-6).unsqueeze(-1)  # B x 1
#         x = sum_x / valid_counts                           # B x D (masked mean)

#         logits = self.head(x)

#         if return_loss and not self.flop_measure:
#             return logits, boundary_loss, avg_boundaries_per_batch, boundary_ratio
#         else:
#             return logits


class HierarchicalDTPViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=768,
                 depth=(2, 2, 6, 2),
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 temp=1.0,
                 compression_rate=(0.25, 0.25, 0.25),
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
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, embed_dim)
        self.dropout = nn.Dropout(drop_rate)

        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))

        def create_decoder_layers(n_layers, n_head):
            layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=n_head,
                        dim_feedforward=int(embed_dim * mlp_ratio),
                        dropout=drop_rate,
                        activation=activation_function,
                        batch_first=False,
                        norm_first=True
                    )
                    for _ in range(n_layers)
                ]
            )

            return layers

        # Transformer blocks for each stage
        self.pre_blocks = create_decoder_layers(depth[0], n_head=num_heads[0])
        self.mid_blocks_1 = create_decoder_layers(depth[1], n_head=num_heads[1])
        self.mid_blocks_2 = create_decoder_layers(depth[2], n_head=num_heads[2])
        self.final_blocks = create_decoder_layers(depth[3], n_head=num_heads[3])

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

        self.bp3 = BoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function=activation_function,
            temp=temp,
            prior=compression_rate[2],
            bp_type=bp_type,
            threshold=threshold
        )

        # Layer norm
        self.down_ln = nn.LayerNorm(embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

        # Final classification head
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes

    def forward_after_pooling_with_attn_masks(self, core_input: torch.Tensor, layers, attention_mask: torch.Tensor):
        """
        Process input with relative attention and padding-aware masking.
        """
        T, B, D = core_input.size()
        core_out = core_input
        for layer in layers:
            core_out = layer(core_out, src_key_padding_mask=attention_mask)
        return core_out
    

    def _make_attn_mask(self, shortened_hidden: torch.Tensor):
        """
        seq: S x B x D
        return: B x S x S mask (True=mask)
        """
        S = shortened_hidden.size(0)
        pad_mask = shortened_hidden.abs().sum(-1).eq(0)       # S x B (1 where padded, 0 where regular)
        attn_mask = pad_mask.transpose(0, 1)                  # (B, S)  True=PAD
        return attn_mask

    def _downsample_stage(self, x: torch.Tensor, boundary_predictor: BoundaryPredictor):
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

        # ✅ Patch embedding already gives (B, L, C)
        x = self.patch_embed(x)                  # (B, L, C)
        x = self.dropout(x)                      # (B, L, C)

        # Positional embedding (for pre-blocks)
        L = x.size(1)
        pos = self.pos_emb[:, 1:1 + L, :].to(device=x.device, dtype=x.dtype)   # (1, L, C)
        x = x + pos                                                             # (B, L, C)
        # Pre-pooling transformer blocks
        x = x.transpose(0, 1)                    # (L, B, C)
        for block in self.pre_blocks:
            x = block(x)

        x, hard_boundaries1 = self._downsample_stage(x, self.bp1)
        attn_mask1 = self._make_attn_mask(x)

        x = self.forward_after_pooling_with_attn_masks(x, self.mid_blocks_1, attention_mask=attn_mask1)

        x, hard_boundaries2 = self._downsample_stage(x, self.bp2)
        attn_mask2 = self._make_attn_mask(x)

        x = self.forward_after_pooling_with_attn_masks(x, self.mid_blocks_2, attention_mask=attn_mask2)

        x, hard_boundaries3 = self._downsample_stage(x, self.bp3)
        attn_mask3 = self._make_attn_mask(x)

        x = self.forward_after_pooling_with_attn_masks(x, self.final_blocks, attention_mask=attn_mask3)
        features = x  # S x B x D

        if return_loss and not self.flop_measure:
            loss1 = self.bp1.calc_loss(hard_boundaries1)
            loss2 = self.bp2.calc_loss(hard_boundaries2)
            loss3 = self.bp3.calc_loss(hard_boundaries3)
            boundary_loss = loss1 + loss2 + loss3
            avg_boundaries_per_batch1 = hard_boundaries1.sum(dim=1).float().mean().item()
            avg_boundaries_per_batch2 = hard_boundaries2.sum(dim=1).float().mean().item()
            avg_boundaries_per_batch3 = hard_boundaries3.sum(dim=1).float().mean().item()

            boundary_ratio1 = avg_boundaries_per_batch1 / hard_boundaries1.size(1)
            boundary_ratio2 = avg_boundaries_per_batch2 / hard_boundaries2.size(1)
            boundary_ratio3 = avg_boundaries_per_batch3 / hard_boundaries3.size(1)

            # only report the second boundary ratio
            # this is not really that helpful, but we keep it for consistency
            cumulative_avg_boundaries_per_batch = avg_boundaries_per_batch2

            # compute the cumulative boundary ratio (e.g., 0.5 * 0.5 = 0.25)
            # NOTE: this is really important to monitor the cumulative compression ratio!
            cumulative_boundary_ratio = boundary_ratio1 * boundary_ratio2 * boundary_ratio3

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


class XL_Baseline(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 num_heads=12,
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 temp=1.0,
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

        # blocks
        self.blocks = create_decoder_layers(12)

        # layer normalization
        self.down_ln = nn.LayerNorm(embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

        # final projection
        self.num_classes = num_classes
        self.head = nn.Linear(embed_dim, num_classes)

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

        x = x.transpose(0, 1)                    # L x B x C
        for block in self.blocks:
            x = block(x, r, self.r_w_bias, self.r_r_bias)

        # return features
        return x

    def forward(self, x, return_loss=False):
        """
        Full forward pass including pooling to class logits.
        """
        features_out = self.encode(x, return_loss=return_loss)

        # pool across sequence dimension with mean pooling
        pad_mask = features_out.abs().sum(-1).eq(0).float()           # S x B
        valid_mask = 1.0 - pad_mask                        # S x B
        valid_mask_exp = valid_mask.unsqueeze(-1)          # S x B x 1

        x = features_out * valid_mask_exp                             # Mask padded tokens
        sum_x = x.sum(dim=0)                               # B x D
        valid_counts = valid_mask.sum(dim=0).clamp(min=1e-6).unsqueeze(-1)  # B x 1
        x = sum_x / valid_counts                           # B x D (masked mean)

        logits = self.head(x)
        return logits
    



###############################################################################################################
##################################### VIT backbone ########################################
###############################################################################################################
class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int = 3, embed_dim: int = 768):
        """
        Patch Embedding Layer
        Args:
            image_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_chans (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Dimension of the embedding space.
        """
        super().__init__()
        self.img_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """
            input: [batch size, # channels, height, width]
            output: [batch size, # patches, embed_dim]
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    """
    A single Transformer block with multi-head self-attention and MLP layers.
    Args:
        dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        drop (float): Dropout rate applied after attention and MLP layers.
        attn_drop (float): Dropout rate applied within attention layers.
        activation_function (str): Activation function used in MLP layers ('gelu' or 'relu').
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0., attn_drop=0., activation_function='gelu'):
        
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=False)
        self.drop_path = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)

        act_fn = nn.GELU() if activation_function == 'gelu' else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_fn,
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


##################################### VIT backbone ########################################
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
                 sinusoidal_pos_emb: bool = True,
                 flop_measure: bool = False,
        ):

        super().__init__()
        self.flop_measure = flop_measure
        self.prior = compression_rate
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.seq_len = self.num_patches
        self.sinusoidal_pos_emb = sinusoidal_pos_emb
        if self.sinusoidal_pos_emb:
            print("🥶🥶🥶🥶🥶Using sinusoidal 2D positional embeddings.")
        else:
            print("😹😹😹😹😹Using learnable positional embeddings.")

        # patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, embed_dim)
        self.dropout = nn.Dropout(drop_rate)

        # positional embedding
        if not self.sinusoidal_pos_emb:
            # self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
            # nn.init.trunc_normal_(self.pos_emb, std=0.02)
            scale = embed_dim ** -0.5
            self.pos_emb = nn.Parameter(
                scale * torch.randn(1, 
                                self.grid_size[0] * self.grid_size[1], 
                                embed_dim)
                        )
        else:
            # FIXME: try sinusoidal 2d pos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                    'currently sin cos 2d pos embedding only supports square input'
            self.pos_emb = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1], embed_dim), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(embed_dim, self.grid_size[0], cls_token=False)
            self.pos_emb.data.copy_(torch.from_numpy(pos_embed_type).float())

        
        def create_decoder_layers(n_layers):
            layers = nn.ModuleList(
                [
                    # nn.TransformerEncoderLayer(
                    #     d_model=embed_dim,
                    #     nhead=num_heads,
                    #     dim_feedforward=int(embed_dim * mlp_ratio),
                    #     dropout=drop_rate,
                    #     activation=activation_function,
                    #     batch_first=False,
                    #     norm_first=True
                    # )
                    ResidualAttentionBlock(
                        d_model=embed_dim,
                        n_head=num_heads,
                        mlp_ratio=mlp_ratio
                    ) # follow CLIP ViT design (GFLOP drops from 2.19 to 1.5)
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

        # final projection
        self.num_classes = num_classes
        self.head = nn.Linear(embed_dim, num_classes)

        print(f"Initialized DTPViT with compression rate: {compression_rate}")
    
    def forward_after_pooling_with_attn_masks(self, core_input, layers, attention_mask):
        T, B, D = core_input.size()

        if not self.sinusoidal_pos_emb: # only add pos emb when not using sin cos
            # take patch pos emb only (drop cls slot)
            patch_pos = self.pos_emb[:, 1:, :]          # (1, N, D), N = original num_patches

            # interpolate to new length T
            patch_pos = patch_pos.transpose(1, 2)       # (1, D, N)
            patch_pos: torch.Tensor = F.interpolate(patch_pos, size=T, mode="linear", align_corners=False)
            patch_pos = patch_pos.transpose(1, 2)       # (1, T, D)
            pos_emb: torch.Tensor = self.dropout(patch_pos)
            pos_emb = pos_emb.transpose(0, 1)  # (T, 1, D)
            core_out = core_input + pos_emb
        else:
            core_out = core_input

        for layer in layers:
            # core_out = layer(core_out, src_key_padding_mask=attention_mask)
            core_out = layer(core_out)
        return core_out

    def encode(self, x: torch.Tensor, return_loss: bool = False):
        B = x.size(0)
        x = self.patch_embed(x)                  # (B, L, C)
        x = self.dropout(x)                      # (B, L, C)

        # Positional embedding (for pre-blocks)
        L = x.size(1)
        # pos = self.pos_emb[:, 1:1 + L, :].to(device=x.device, dtype=x.dtype)   # (1, L, C)
        # x = x + pos                                                             # (B, L, C)
        
        # FIXME: just fixed unnecessary slicing
        x = x + self.pos_emb.to(device=x.device, dtype=x.dtype)

        # Pre-pooling transformer blocks
        x = x.transpose(0, 1)                    # (L, B, C)
        for block in self.pre_blocks:
            x = block(x)

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

        attn_mask = pad_mask.transpose(0, 1)                  # (B, S)  True=PAD

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
        

####################################
### more baselines #################
####################################

class PatchMergeAvg(nn.Module):
    """
    Fixed 2x2 spatial downsampling WITHOUT channel concat/projection.
    Input:  x (B, L, C) with L = H*W
    Output: (B, (H/2)*(W/2), C)
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, f"Expected L=H*W, got L={L}, H*W={H*W}"
        assert H % 2 == 0 and W % 2 == 0, f"H and W must be even, got {H},{W}"

        x = x.view(B, H, W, C)
        x00 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x10 = x[:, 1::2, 0::2, :]
        x01 = x[:, 0::2, 1::2, :]
        x11 = x[:, 1::2, 1::2, :]
        x = (x00 + x10 + x01 + x11) * 0.25
        x = x.reshape(B, (H // 2) * (W // 2), C)
        return x

class TokenPool1DAvg(nn.Module):
    """
    Fixed 1-D token pooling (non-spatial-aware).
    Pools consecutive tokens in the sequence by averaging groups of `pool`.
    
    Input:  x (B, L, C)
    Output: (B, L/pool, C)
    """
    def __init__(self, pool: int = 4, handle_tail: str = "trim"):
        """
        Args:
            pool: group size for averaging. Use 4 to mimic 2x2 spatial merge.
            handle_tail:
                - "trim": drop leftover tokens if L % pool != 0
                - "pad":  right-pad with zeros to the next multiple of `pool`
        """
        super().__init__()
        assert handle_tail in ("trim", "pad")
        self.pool = pool
        self.handle_tail = handle_tail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        p = self.pool

        if L % p != 0:
            if self.handle_tail == "trim":
                L_new = (L // p) * p
                x = x[:, :L_new, :]
            else:  # pad
                L_new = ((L + p - 1) // p) * p
                pad_len = L_new - L
                pad = x.new_zeros(B, pad_len, C)
                x = torch.cat([x, pad], dim=1)

        # now length is divisible by p
        B, L, C = x.shape
        x = x.view(B, L // p, p, C).mean(dim=2)
        return x




class SingleAdaptedFixed(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,            # constant across pre/post
                 depth=(2, 8),             # (pre_depth, post_depth)
                 num_heads=(12, 12),       # (pre_heads, post_heads) or single int
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 num_classes=512,
                 activation_function='gelu',
                 flop_measure: bool = False,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.flop_measure = flop_measure
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        C = embed_dim

        # ----- grids -----
        Hp = image_size // patch_size
        Wp = image_size // patch_size
        assert Hp * patch_size == image_size and Wp * patch_size == image_size
        assert (Hp % 2 == 0) and (Wp % 2 == 0), "Need even H/ W (one 2x merge)."
        self.grid0 = (Hp, Wp)
        self.grid1 = (Hp // 2, Wp // 2)

        L0 = Hp * Wp
        L1 = (Hp // 2) * (Wp // 2)

        # ----- heads -----
        if isinstance(num_heads, int):
            heads = (num_heads, num_heads)
        else:
            assert len(num_heads) == 2
            heads = num_heads

        # ----- patch embed -----
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, C)
        self.dropout = nn.Dropout(drop_rate)

        # ----- pos embeddings (both C, constant width) -----
        self.pos_pre  = nn.Parameter(torch.zeros(1, 1 + L0, C))
        # self.pos_post = nn.Parameter(torch.zeros(1, 1 + L1, C))

        # FIXME: ablation - pool=1        
        self.pos_post = nn.Parameter(torch.zeros(1, 1 + L0, C))


        nn.init.trunc_normal_(self.pos_pre,  std=0.02)
        nn.init.trunc_normal_(self.pos_post, std=0.02)

        # ----- transformer stacks -----
        def make_layers(n_layers, d_model, nhead):
            return nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=int(d_model * mlp_ratio),
                    dropout=drop_rate,
                    activation=activation_function,
                    batch_first=False,
                    norm_first=True
                )
                # ResidualAttentionBlock(
                #     d_model=embed_dim,
                #     n_head=num_heads,
                #     mlp_ratio=mlp_ratio
                # ) # follow CLIP ViT design (GFLOP drops from 2.19 to 1.5)
                for _ in range(max(0, n_layers))
            ])

        self.pre_blocks  = make_layers(depth[0], C, heads[0])
        self.post_blocks = make_layers(depth[1], C, heads[1])

        # ----- fixed pooling (avg) -----
        # self.merge = TokenPool1DAvg(pool=4, handle_tail="trim")  # L0 -> L1, keep C

        # FIXME: ablation - pool=1        
        self.merge = TokenPool1DAvg(pool=1, handle_tail="trim")  # L0 -> L1, keep C


        # ----- norm + head -----
        self.post_ln = norm_layer(C)
        # self.head    = nn.Linear(C, num_classes)
        self.head = nn.Parameter(torch.empty(C, num_classes))  # matches CLIP's proj shape
        self.head_bias = None   # optional, CLIP has no bias


    # utils
    def _add_pos(self, x, pos_param):
        B, L, C = x.shape
        pos = pos_param[:, 1:1+L, :].to(device=x.device, dtype=x.dtype)
        return x + pos

    def _run_stack(self, x, layers):
        for blk in layers:
            x = blk(x)
        return x

    def encode(self, x: torch.Tensor, return_loss: bool = False):
        # pre
        x = self.patch_embed(x)                 # (B, L0, C)
        x = self.dropout(x)
        x = self._add_pos(x, self.pos_pre)
        x = x.transpose(0, 1)                   # (L0, B, C)
        x = self._run_stack(x, self.pre_blocks) # (L0, B, C)

        # merge (no channel change)
        x = x.transpose(0, 1)                   # (B, L0, C)
        x = self.merge(x)                       # (B, L1, C)
        x = self._add_pos(x, self.pos_post)
        x = x.transpose(0, 1)                   # (L1, B, C)

        # post
        x = self._run_stack(x, self.post_blocks)  # (L1, B, C)

        if return_loss:
            # keep DTPViT API
            dummy_loss = torch.zeros([], device=x.device)
            # single fixed merge → tokens /4
            avg_boundaries = 0.0
            boundary_ratio = 1.0 / 4.0
            return x, dummy_loss, avg_boundaries, boundary_ratio
        else:
            return x

    def forward(self, x, return_loss: bool = False):
        out = self.encode(x, return_loss=return_loss)
        if return_loss:
            feats, boundary_loss, avg_boundaries, boundary_ratio = out
        else:
            feats = out

        # mean pool over sequence (dense)
        x = feats.mean(dim=0)   # (B, C)
        x = self.post_ln(x)
        
        # FIXME: not a linear layer, but a projection matrix like CLIP
        # logits = self.head(x)
        logits = x @ self.head  # (B, num_classes == output_dim)


        if return_loss:
            return logits, boundary_loss, avg_boundaries, boundary_ratio
        else:
            return logits


class SingleAdaptedSwin(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,            # constant across pre/post
                 depth=(2, 8),             # (pre_depth, post_depth)
                 num_heads=(12, 12),       # (pre_heads, post_heads) or single int
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 num_classes=1000,
                 activation_function='gelu',
                 flop_measure: bool = False,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.flop_measure = flop_measure
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        C = embed_dim

        # ----- grids -----
        Hp = image_size // patch_size
        Wp = image_size // patch_size
        assert Hp * patch_size == image_size and Wp * patch_size == image_size
        assert (Hp % 2 == 0) and (Wp % 2 == 0), "Need even H/ W (one 2x merge)."
        self.grid0 = (Hp, Wp)
        self.grid1 = (Hp // 2, Wp // 2)

        L0 = Hp * Wp
        L1 = (Hp // 2) * (Wp // 2)

        # ----- heads -----
        if isinstance(num_heads, int):
            heads = (num_heads, num_heads)
        else:
            assert len(num_heads) == 2
            heads = num_heads

        # ----- patch embed -----
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, C)
        self.dropout = nn.Dropout(drop_rate)

        # ----- pos embeddings (both C, constant width) -----
        self.pos_pre  = nn.Parameter(torch.zeros(1, 1 + L0, C))
        self.pos_post = nn.Parameter(torch.zeros(1, 1 + L1, C))
        nn.init.trunc_normal_(self.pos_pre,  std=0.02)
        nn.init.trunc_normal_(self.pos_post, std=0.02)

        # ----- transformer stacks -----
        def make_layers(n_layers, d_model, nhead):
            return nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=int(d_model * mlp_ratio),
                    dropout=drop_rate,
                    activation=activation_function,
                    batch_first=False,
                    norm_first=True
                )
                for _ in range(max(0, n_layers))
            ])

        self.pre_blocks  = make_layers(depth[0], C, heads[0])
        self.post_blocks = make_layers(depth[1], C, heads[1])

        # ----- fixed pooling (avg) -----
        self.merge = PatchMergeAvg(self.grid0, dim=C)  # L0 -> L1, keep C

        # ----- norm + head -----
        self.post_ln = norm_layer(C)
        self.head    = nn.Linear(C, num_classes)

    # utils
    def _add_pos(self, x, pos_param):
        B, L, C = x.shape
        pos = pos_param[:, 1:1+L, :].to(device=x.device, dtype=x.dtype)
        return x + pos

    def _run_stack(self, x, layers):
        for blk in layers:
            x = blk(x)
        return x

    def encode(self, x: torch.Tensor, return_loss: bool = False):
        # pre
        x = self.patch_embed(x)                 # (B, L0, C)
        x = self.dropout(x)
        x = self._add_pos(x, self.pos_pre)
        x = x.transpose(0, 1)                   # (L0, B, C)
        x = self._run_stack(x, self.pre_blocks) # (L0, B, C)

        # merge (no channel change)
        x = x.transpose(0, 1)                   # (B, L0, C)
        x = self.merge(x)                       # (B, L1, C)
        x = self._add_pos(x, self.pos_post)
        x = x.transpose(0, 1)                   # (L1, B, C)

        # post
        x = self._run_stack(x, self.post_blocks)  # (L1, B, C)

        if return_loss:
            # keep DTPViT API
            dummy_loss = torch.zeros([], device=x.device)
            # single fixed merge → tokens /4
            avg_boundaries = 0.0
            boundary_ratio = 1.0 / 4.0
            return x, dummy_loss, avg_boundaries, boundary_ratio
        else:
            return x

    def forward(self, x, return_loss: bool = False):
        out = self.encode(x, return_loss=return_loss)
        if return_loss:
            feats, boundary_loss, avg_boundaries, boundary_ratio = out
        else:
            feats = out

        # mean pool over sequence (dense)
        x = feats.mean(dim=0)   # (B, C)
        x = self.post_ln(x)
        logits = self.head(x)

        if return_loss:
            return logits, boundary_loss, avg_boundaries, boundary_ratio
        else:
            return logits
