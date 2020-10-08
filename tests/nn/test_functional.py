import pytest
import torch
import torch.nn as nn
import proteinsolver  # noqa
from torch_geometric.utils import dense_to_sparse, scatter_

from proteinsolver.nn import functional as F2


def test_sparse_multi_head_attention_forward_0():
    # Parameters
    num_nodes = 100
    embed_dim = 162
    kdim = None
    vdim = None
    num_heads = 9
    bias = True
    add_bias_kv = False
    dropout = 0.0
    add_zero_attn = False
    training = True

    # Forward arguments
    query = torch.randn(num_nodes, embed_dim, dtype=torch.float32)
    mat = (
        #
        (torch.rand(num_nodes, num_nodes) > 0.8).to(torch.float32)
        + torch.eye(num_nodes, dtype=torch.float32)
    )
    index, _value = dense_to_sparse(mat)
    value = torch.randn(_value.size(0), embed_dim)
    key = value
    key_padding_mask = None
    need_weights = True
    attn_mask = None

    # Derived parameters
    kdim = kdim if kdim is not None else embed_dim
    vdim = vdim if vdim is not None else embed_dim
    _qkv_same_embed_dim = kdim == embed_dim and vdim == embed_dim
    assert _qkv_same_embed_dim is True

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    in_proj_weight = torch.empty(3 * embed_dim, embed_dim)
    nn.init.xavier_uniform_(in_proj_weight)

    out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    if bias:
        in_proj_bias = torch.empty(3 * embed_dim)
        nn.init.constant_(in_proj_bias, 0.0)
        nn.init.constant_(out_proj.bias, 0.0)

    if add_bias_kv:
        bias_k = torch.empty(1, 1, embed_dim)
        bias_v = torch.empty(1, 1, embed_dim)
        nn.init.xavier_normal_(bias_k)
        nn.init.xavier_normal_(bias_v)
    else:
        bias_k = bias_v = None

    attn_output, attn_output_weights = F2.sparse_multi_head_attention_forward(
        query,
        key,
        value,
        index,
        embed_dim,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn,
        dropout,
        out_proj.weight,
        out_proj.bias,
        training=training,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
    )

    assert list(query.shape) == list(attn_output.shape)

    total_weight_per_node = scatter_("add", attn_output_weights, index[0], dim=0)
    assert (total_weight_per_node > 0.99).all().item()
    assert (total_weight_per_node < 1.01).all().item()

    # Check that there are no nulls
    assert (attn_output == attn_output).all().item()
    assert (attn_output_weights == attn_output_weights).all().item()


@pytest.mark.skip(reason="Not implemented.")
def sparse_multi_head_attention_forward_1():
    # Forward arguments
    M = 100
    query = torch.randn(M, 12, dtype=torch.float32)
    mat = ((torch.rand(M, M) > 0.8) + torch.eye(M)).to(torch.float32)
    index, _value = dense_to_sparse(mat)
    value = torch.randn(_value.size(0), M)
    key = value
    key_padding_mask = None
    need_weights = True
    attn_mask = None

    # Parameters
    embed_dim = 162
    kdim = None
    vdim = None
    num_heads = 9
    bias = True
    add_bias_kv = False
    dropout = 0.0
    add_zero_attn = False
    training = True

    # Derived parameters
    kdim = kdim if kdim is not None else embed_dim
    vdim = vdim if vdim is not None else embed_dim
    _qkv_same_embed_dim = kdim == embed_dim and vdim == embed_dim
    assert _qkv_same_embed_dim is False

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    q_proj_weight = torch.Tensor(embed_dim, embed_dim)
    k_proj_weight = torch.Tensor(embed_dim, kdim)
    v_proj_weight = torch.Tensor(embed_dim, vdim)

    if bias:
        in_proj_bias = torch.empty(3 * embed_dim)

    out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    if add_bias_kv:
        bias_k = torch.empty(1, 1, embed_dim)
        bias_v = torch.empty(1, 1, embed_dim)
    else:
        bias_k = bias_v = None

    _ = F2.sparse_multi_head_attention_forward(
        query,
        key,
        value,
        index,
        embed_dim,
        num_heads,
        None,  # set in_proj_weight = None
        in_proj_bias,
        bias_k,
        bias_v,
        add_zero_attn,
        dropout,
        out_proj.weight,
        out_proj.bias,
        training=training,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        use_separate_proj_weight=True,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
    )
