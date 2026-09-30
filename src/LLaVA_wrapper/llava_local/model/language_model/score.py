import torch
import os

VERSION = "1_0"
V2_0 = False

RETAIN_TOKN = 58

layer_dict = {2: 0}
sparse_token_list_4x = [144]
sparse_token_list_8x = [72]
sparse_token_list_10x = [58]

sparse_token_dict = {
    144: sparse_token_list_4x,
    72: sparse_token_list_8x,
    58: sparse_token_list_10x,
}

def attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx , v_token_start: v_token_start+v_token_num] # B, L2, L1

    relation_vis_text = relation_vis_text.mean(1) # B, L1

    relation_vis = relation_vis_text
    s_flag = False # NOTE: disable token supplement for fair comparison

    sparse_token_list = sparse_token_dict[RETAIN_TOKN]

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(sparse_token_list[layer_dict[layer_idx]], v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text

def select_attn_head_by_sum(self_attn_weights, t_token_idx, v_token_start, text_token_start):
    # [1,28,token_num,token_num] -> [28,text_token_num,visual_token_num]
    each_head_text_to_visual_attn = self_attn_weights[0][:, t_token_idx , v_token_start: text_token_start]
    # [28,text_token_num,visual_token_num] -> [28]
    sum_attn_per_head = each_head_text_to_visual_attn.sum((1,2))
    select_attn_head_idx = sum_attn_per_head.topk(14)[1]

    return self_attn_weights[:,select_attn_head_idx,:,:][:,:,:]

if __name__ == "__main__":

    self_attn_weights, v_token_start, v_token_num, text_token_start = torch.rand(4, 16, 1084, 1084), 36, 576, 700
    mask = attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start)
    print(mask.shape)