import copy
from typing import Optional, Any, Union, Callable, List
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
# from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
import torch.nn as nn
import sys
import math
import numpy as np




# From https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]



# From https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/mha.py
class PrepareForMultiHeadAttention(nn.Module):
    """
    <a id="PrepareMHA"></a>
    ## Prepare for multi-head attention
    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, d_model]`
        return x


# Adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/mha.py
class MultiHeadAttention(nn.Module):
    r"""
    <a id="MHA"></a>
    ## Multi-Head Attention Module
    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.
    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.
    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.
    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, nhead: int, d_model: int, dropout: float = 0.3, bias: bool = True):
        """
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // nhead
        # Number of heads
        self.nhead = nhead

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, nhead, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, nhead, self.d_k, bias=bias)
        self.key_P = PrepareForMultiHeadAttention(d_model, nhead, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, nhead, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None
        self.batch_first = False



    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """

        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)

        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                r_w_bias = None, r_r_bias = None,
                mask: Optional[torch.Tensor] = None,
                need_weights = True):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.
        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        # query = self.query(query)
        # key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.

        if r_w_bias == None:
            query = self.query(query)
            key = self.key(key)
            scores = self.get_scores(query, key)
        else:
            scores = self.get_scores(query, key, r_w_bias, r_r_bias)
        

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)


        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Save attentions for any other calculations 
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        out = self.output(x)

        if need_weights:
            return (out, (attn.cpu().detach().numpy()))
        else:
            return out





# Skeleton adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/xl/relative_mha.py
# Computation of attention adapted from https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
class RelativePartialMultiHeadAttention(MultiHeadAttention):
    """
    ## Relative Multi-Head Attention Module
    We override [Multi-Head Attention](mha.html) module so we only need to 
    write the `get_scores` method.
    """

    def __init__(self, nhead: int, d_model: int, dropout: float = 0.3):
        super().__init__(nhead, d_model, dropout, bias=False)
        # self.P = 2 ** 12
        self.P = 210
        self.batch_first = False
        self.pos_emb = PositionalEmbedding(d_model)


    def get_scores(self, query: torch.Tensor, key: torch.Tensor, r_w_bias, r_r_bias):
        k_len = query.shape[0]

        pos_seq = torch.arange(k_len - 1, -1, -1.0, device=query.device, dtype=query.dtype)
        r = self.pos_emb(pos_seq)
        r_head_k = self.key_P(r)
        r_head_k = r_head_k.view(k_len, self.nhead, self.d_k)

        w_head_q = self.query(query)
        w_head_k = self.key(query)
        #### compute attention score
        # print(w_head_q.shape, self.r_w_bias.shape)
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        # print('rw_head_q', rw_head_q.shape, w_head_k.shape)

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        # print('AC', AC.shape)

        # print(w_head_q.shape, self.r_r_bias.shape)

        rr_head_q = w_head_q + r_r_bias
        # print('rr_head_q', rr_head_q.shape, r_head_k.shape)
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k)) # qlen x klen x bsz x n_head
        # print('BD', BD.shape)              
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)
        return attn_score

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x


# Adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``False`` (disabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        # self.layers[0].set_attn()
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.r_w_bias = nn.Parameter(torch.zeros(encoder_layer.nhead, encoder_layer.d_k), requires_grad=True)
        self.r_r_bias = nn.Parameter(torch.zeros(encoder_layer.nhead, encoder_layer.d_k), requires_grad=True)


    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        attn_list = []
        if isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor) :
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not())

        for mod in self.layers:
            if convert_to_nested:
                output, attn = mod(output, self.r_w_bias, self.r_r_bias, src_mask=mask)
            else:
                output, attn = mod(output, self.r_w_bias, self.r_r_bias, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            attn_list.append(attn)


        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_list



# Adapted from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - norm_first is ``False`` (this restriction may be loosened in the future)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.dropout_rate = dropout
        # self.self_attn = RelativeMultiHeadAttention(nhead, d_model, dropout=dropout)
        # self.self_attn = MultiHeadAttention(d_model = d_model, nhead = nhead, dropout=dropout)
        self.self_attn = RelativePartialMultiHeadAttention(nhead, d_model, dropout=dropout)
        # config = deberta.ModelConfig()
        # config.num_attention_heads = nhead
        # config.hidden_size = 256
        # config.hidden_dropout_prob = dropout
        # config.attention_probs_dropout_prob = dropout
        # config.max_position_embeddings = 230
        # config.intermediate_size = 256

        # self.self_attn = deberta.DisentangledSelfAttention(config)
        # self.self_attn = DisentangledSelfAttention(config)


        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation


    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu



    def forward(self, src: Tensor,  r_w_bias, r_r_bias,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        if (src.dim() == 3 and not self.norm_first and not self.training and
            self.self_attn.batch_first and
            self.self_attn._qkv_same_embed_dim and self.activation_relu_or_gelu and
            self.norm1.eps == self.norm2.eps and
            src_mask is None and
                not (src.is_nested and src_key_padding_mask is not None)):
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            if (not torch.overrides.has_torch_function(tensor_args) and
                    # We have to use a list comprehension here because TorchScript
                    # doesn't support generator expressions.
                    all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]) and
                    (not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]))):
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    False,  # norm_first, currently not supported
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    src_mask if src_mask is not None else src_key_padding_mask,  # TODO: split into two args
                )
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), r_w_bias, r_r_bias, src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            # print('Checkpoint 3')
            sa, attn = self._sa_block(x, r_w_bias, r_r_bias, src_mask, src_key_padding_mask)
            x = self.norm1(x + sa)
            x = self.norm2(x + self._ff_block(x))

        return x, attn

    # self-attention block
    def _sa_block(self, x: Tensor, r_w_bias, r_r_bias,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        # x, attn = self.self_attn(x, x, x, need_weights = True)
        # x = torch.transpose(x, 0, 1)
        # attn_mask = torch.ones((x.shape[0], 1, x.shape[1],x.shape[1]), device = x.device)
        # # print(attn_mask.shape, x.shape)
        # result = self.self_attn(x, attn_mask, return_att=True)
        # x = result['hidden_states']
        # attn = result['attention_probs']
        # # print(x.shape, attn.shape)
        # x = torch.transpose(x, 0, 1)

        x, attn = self.self_attn(x, x, x, r_w_bias, r_r_bias, need_weights = True)


        return self.dropout1(x), attn

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
