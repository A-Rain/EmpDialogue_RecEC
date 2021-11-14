import warnings
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union, List

import torch
from torch import nn
import texar.torch as tx

from texar.torch.core import layers
from texar.torch.modules.decoders.decoder_base import (
    DecoderBase, TokenEmbedder, TokenPosEmbedder, _make_output_layer)
from texar.torch.modules.decoders.decoder_helpers import (
    EmbeddingHelper, Helper)
from texar.torch.modules.encoders.multihead_attention import (
    Cache, MultiheadAttentionEncoder)
from texar.torch.modules.encoders.transformer_encoder import (
    default_transformer_poswise_net_hparams)
from texar.torch.modules.networks.networks import FeedForwardNetwork
from texar.torch.utils import transformer_attentions as attn
from texar.torch.utils.beam_search import beam_search
from texar.torch.utils.shapes import mask_sequences
from texar.torch.utils.utils import sequence_mask

from texar.torch.core import layers
from texar.torch.modules.encoders.encoder_base import EncoderBase
from texar.torch.utils.types import MaybeList
from mypy_extensions import TypedDict
import torch.nn.functional as F


EmbeddingFn = Callable[[torch.LongTensor, torch.LongTensor], torch.Tensor]


class TransformerDecoderOutput(NamedTuple):
    r"""The output of :class:`TransformerDecoder`.
    """
    logits: torch.Tensor
    r"""A :tensor:`Tensor` of shape ``[batch_size, max_time, vocab_size]``
    containing the logits."""
    sample_id: torch.LongTensor
    r"""A :tensor:`LongTensor` of shape ``[batch_size, max_time]``
    (or ``[batch_size, max_time, vocab_size]``) containing the sampled
    token indices. Note that the shape of ``sample_id`` is different for
    different decoding strategy or helper. Please refer to
    :class:`~texar.torch.modules.Helper` for the detailed information."""


class LayerCache(TypedDict):
    r"""Cache (state) for a single self-attention layer in
    :class:`MultiheadAttentionEncoder`.
    """
    keys: MaybeList[torch.Tensor]
    values: MaybeList[torch.Tensor]


class Cache(TypedDict):
    r"""Cache (state) for the entire :class:`MultiheadAttentionEncoder`.
    """
    memory: Optional[torch.Tensor]
    memory_attention_bias: Optional[torch.Tensor]
    layers: List[LayerCache]

class CustomMultiheadAttentionEncoder(MultiheadAttentionEncoder):
    def forward(self,  # type: ignore
                queries: torch.Tensor,
                memory: torch.Tensor,
                memory_attention_bias: torch.Tensor,
                cache: Optional[LayerCache] = None) \
            -> torch.Tensor:
        r"""Encodes the inputs.

        Args:
            queries: A 3D tensor with shape of
                ``[batch, length_query, depth_query]``.
            memory: A 3D tensor with shape of
                ``[batch, length_key, depth_key]``.
            memory_attention_bias: A 3D tensor with shape of
                ``[batch, length_key, num_units]``.
            cache: Memory cache only when inferring the sentence from scratch.

        Returns:
            A tensor of shape ``[batch_size, max_time, dim]`` containing the
            encoded vectors.
        """

        num_heads = self._hparams.num_heads
        num_units = self._hparams.num_units
        if num_units % num_heads != 0:
            raise ValueError(
                f"Value depth ({num_units}) must be divisible by "
                f"the number of attention heads ({num_heads}).")

        def _update_and_return(layer: nn.Module, key: str):
            if memory is None:
                # Self Attention
                out = layer(queries)

                if cache is not None:
                    # decoder self attention when dynamic decoding
                    res: MaybeList[torch.Tensor] = cache[key]
                    if isinstance(res, list):
                        # inference-like decoding
                        res.append(out.squeeze(1))
                        out = torch.stack(res, dim=1)
                    else:
                        # normal decoding
                        res = torch.cat([res, out], dim=1)
                        out = res
                    cache[key] = res

            else:
                # encoder decoder attention
                if cache is not None:
                    res: MaybeList[torch.Tensor] = cache[key]  # type: ignore
                    if isinstance(res, list):
                        # inference-like decoding
                        if len(res) == 0:
                            out = layer(memory)
                        else:
                            out = torch.stack(res, dim=1)
                    else:
                        # normal decoding
                        if res.size(1) == 0:
                            out = layer(memory)
                        else:
                            out = res
                else:
                    out = layer(memory)

            return out

        Q = self.Q_dense(queries)
        K = _update_and_return(self.K_dense, 'keys')
        V = _update_and_return(self.V_dense, 'values')

        Q_ = self._split_heads(Q)
        K_ = self._split_heads(K)
        V_ = self._split_heads(V)
        # [batch_size, num_heads, seq_length, memory_depth]
        key_depth_per_head = num_units // num_heads
        Q_ *= key_depth_per_head ** -0.5

        logits = torch.matmul(Q_, K_.transpose(-2, -1))
        if memory_attention_bias is not None:
            memory_attention_bias = memory_attention_bias.to(
                device=logits.device)
            logits += memory_attention_bias
        weights = torch.softmax(logits, dim=-1)
        weights = F.dropout(weights, self._hparams.dropout_rate, self.training)
        outputs = torch.matmul(weights, V_)

        outputs = self._combine_heads(outputs)
        outputs = self.O_dense(outputs)
        # (batch_size, length_query, output_dim)

        return outputs, logits

class TransformerEncoder(tx.modules.TransformerEncoder):
    def initialize_blocks(self):
        r"""Helper function which initializes blocks for encoder.

        Should be overridden by any classes where block initialization varies.
        """
        for _ in range(self._hparams.num_blocks):
            mh_attn = CustomMultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            mh_attn2 = CustomMultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)

            self.self_attns.append(mh_attn)
            self.self_attns.append(mh_attn2)

            if not self._hparams.use_bert_config:
                self.self_attn_layer_norm.append(
                    nn.LayerNorm(self._input_size, eps=self._hparams.eps))
            if self._hparams.dim != mh_attn.hparams.output_dim:
                raise ValueError(
                    'The "dim" in the hparams of '
                    '"multihead_attention" should be equal to the '
                    '"dim" of TransformerEncoder')

            pw_net = FeedForwardNetwork(
                hparams=self._hparams['poswise_feedforward'])

            final_dim = pw_net.hparams.layers[-1]['kwargs']['out_features']
            if self._hparams.dim != final_dim:
                raise ValueError(
                    'The output dimenstion of '
                    '"poswise_feedforward" should be equal '
                    'to the "dim" of TransformerEncoder.')

            self.poswise_networks.append(pw_net)
            self.poswise_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=self._hparams.eps))
            if self._hparams.use_bert_config:
                self.output_layer_norm.append(
                    nn.LayerNorm(self._input_size, eps=self._hparams.eps))

    def forward(self,  # type: ignore
                inputs: torch.Tensor,
                sequence_length: torch.LongTensor) -> torch.Tensor:
        r"""Encodes the inputs.

        Args:
            inputs: A 3D Tensor of shape ``[batch_size, max_time, dim]``,
                containing the embedding of input sequences. Note that
                the embedding dimension `dim` must equal "dim" in
                :attr:`hparams`. The input embedding is typically an
                aggregation of word embedding and position embedding.
            sequence_length: A 1D :tensor:`LongTensor` of shape
                ``[batch_size]``. Input tokens beyond respective sequence
                lengths are masked out automatically.

        Returns:
            A Tensor of shape ``[batch_size, max_time, dim]`` containing the
            encoded vectors.
        """
        # Multiply input embedding with the sqrt of its dimension for
        # normalization

        inputs_padding = 1 - sequence_mask(
            sequence_length, inputs.size()[1]).float()

        if self._hparams.use_bert_config:
            ignore_padding = attn.attention_bias_ignore_padding(
                inputs_padding, bias_value=-1e4)
        else:
            ignore_padding = attn.attention_bias_ignore_padding(
                inputs_padding)
        encoder_self_attention_bias = ignore_padding
        input_embedding = inputs
        if self._hparams.use_bert_config:
            x = self.input_normalizer(input_embedding)
            x = self.embed_dropout(x)
        else:
            x = self.embed_dropout(input_embedding)
        attn_logits  = []
        for i in range(self._hparams.num_blocks):
            # trivial difference between BERT and original Transformer
            if self._hparams.use_bert_config:
                _queries_input = x
            else:
                # _queries_input = self.self_attn_layer_norm[i](x)
                _queries_input = x

            attention_output, logits  = self.self_attns[i](
                queries=_queries_input,
                memory=_queries_input,
                memory_attention_bias=encoder_self_attention_bias,
            )

            

            attention_output = self.residual_dropout(attention_output)

            # x = x + attention_output
            x = attention_output
                
            attn_logits.append(logits)
            # x = self.poswise_layer_norm[i](x)

            # cause_attn_output, logits = self.self_attns[i+1](
            #     queries=x,
            #     memory=x,
            #     memory_attention_bias=coause_self_attention_bias,
            # )

            # x = cause_attn_output
            # x = attention_output
            

            # attention_output, logits  = self.self_attns[i+1](
            #     queries=x,
            #     memory=x,
            #     memory_attention_bias=encoder_self_attention_bias,
            # )

            # x = attention_output

            

            poswise_network = self.poswise_networks[i]
            poswise_normalizer = self.poswise_layer_norm[i]

            if self._hparams.use_bert_config:
                x = poswise_normalizer(x)
                y = x
            else:
                y = poswise_normalizer(x)

            original_shape = y.size()

            y = y.view(-1, self._hparams.dim)

            layer_output = poswise_network(y)
            sub_output = self.residual_dropout(layer_output)
            sub_output = layer_output
            sub_output = sub_output.view(original_shape)

            # x = x + sub_output
            if self._hparams.use_bert_config:
                x = self.output_layer_norm[i](x)

        if not self._hparams.use_bert_config:
            x = self.final_layer_norm(x)
        # x = x + inputs
        return x, attn_logits[-1]


class TransformerDecoder(tx.modules.TransformerDecoder):
    # State variables used during `dynamic_decode`. Assigned in `forward`.
    _state_max_decoding_length: int
    _state_context: Optional[torch.LongTensor]
    _state_context_sequence_length: Optional[torch.LongTensor]
    _state_cache: Cache

    def embed_tokens(self, tokens: torch.LongTensor,
                     positions: torch.LongTensor) -> torch.Tensor:
        r"""Convert tokens along with positions to embeddings.

        Args:
            tokens: A :tensor:`LongTensor` denoting the token indices to convert
                to embeddings.
            positions: A :tensor:`LongTensor` with the same size as
                :attr:`tokens`, denoting the positions of the tokens. This is
                useful if the decoder uses positional embeddings.

        Returns:
            A :tensor:`Tensor` of size ``tokens.size() + (embed_dim,)``,
            denoting the converted embeddings.
        """
        if self._token_embedder is not None:
            return self._token_embedder(tokens)
        assert self._token_pos_embedder is not None
        return self._token_pos_embedder(tokens, positions)
    def initialize_blocks(self):
        r"""Helper function which initializes blocks for decoder.

        Should be overridden by any classes where block initialization varies.
        """
        self.cause_attns = nn.ModuleList()
        self.cause_attn_layer_norm = nn.ModuleList()
        for _ in range(self._hparams.num_blocks):
            attn_module = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadEncoder should be equal "
                                 "to the dim of TransformerDecoder")
            self.self_attns.append(attn_module)
            self.self_attn_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=self._hparams.eps))

            attn_module = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadEncoder should be equal "
                                 "to the dim of TransformerDecoder")
            self.enc_dec_attns.append(attn_module)
            self.end_dec_attn_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=self._hparams.eps))

            attn_module = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadEncoder should be equal "
                                 "to the dim of TransformerDecoder")
            self.cause_attns.append(attn_module)
            self.cause_attn_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=self._hparams.eps))

            poswise_network = FeedForwardNetwork(
                hparams=self._hparams.poswise_feedforward)
            if (poswise_network.hparams.layers[-1]['kwargs']['out_features']
                    != self._hparams.dim):
                raise ValueError("The output dimension of "
                                 "FeedForwardNetwork should be equal "
                                 "to the dim of TransformerDecoder")
            self.poswise_networks.append(poswise_network)
            self.poswise_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=self._hparams.eps))

    def step(self, helper: Helper, time: int, inputs: torch.Tensor,
             state: Optional[Cache]) -> \
            Tuple[TransformerDecoderOutput, Cache]:
        assert state is not None
        outputs, state = self._inputs_to_outputs(inputs, state)
        sample_ids = helper.sample(time=time, outputs=outputs)
        if self._state_context is not None:
            assert self._state_context_sequence_length is not None
            sample_ids = torch.where(
                self._state_context_sequence_length > time,
                self._state_context[:, time],
                sample_ids)

        next_state = state
        outputs = TransformerDecoderOutput(
            logits=outputs,
            sample_id=sample_ids)
        return outputs, next_state

    def _init_cache(self, memory: Optional[torch.Tensor],
                    memory_attention_bias: Optional[torch.Tensor],
                    cause_attention_bias: Optional[torch.Tensor],
                    emotion_embed: Optional[torch.Tensor],
                    beam_search_decoding: bool,
                    batch_size: int) -> Cache:
        r"""Returns an initialized cache.

        In order to support both inference-like decoding and beam-search
        decoding, the elements of each layer must be initialized and extended
        as different structure respectively. Specifically, for inference-like
        decoding, a simple list is used; for beam-search decoding, a
        :tensor:`Tensor` of shape ``[batch_size, current_steps, num_units]``
        is maintained, where ``current_steps`` is the number of steps currently
        decoded.
        """

        device = next(self.parameters()).device

        def _create_ta():
            return []

        def _create_empty_tensor():
            ret = torch.zeros(
                batch_size, 0, self._hparams.multihead_attention.num_units,
                dtype=torch.float, device=device)
            return ret

        _create_fn = (_create_empty_tensor if beam_search_decoding
                      else _create_ta)

        cache: Cache = {
            'memory': memory,
            'memory_attention_bias': memory_attention_bias,
            'cause_attention_bias': cause_attention_bias,
            'emotion_embed': emotion_embed,
            'layers': [{
                'keys': _create_fn(),
                'values': _create_fn(),
            } for _ in range(self._hparams.num_blocks)],
        }

        return cache
    def _inputs_to_outputs(self, inputs: torch.Tensor,
                           cache: Cache) -> Tuple[torch.Tensor, Cache]:
        r"""Returns the outputs of one decoding step (for example,
        the predicted logits of the next token).

        :attr:`inputs` should be of shape ``[batch_size, dim]``.

        Returns:
            A tuple of logits and updated cache. Logits are of shape
            ``[batch_size, vocab_size]``.
        """

        outputs = self._self_attention_stack(
            inputs.unsqueeze(1)+cache['emotion_embed'], memory=cache['memory'], cache=cache)
        outputs = self._output_layer(outputs)
        outputs = outputs.squeeze(1)
        return outputs, cache
    def forward(self,  # type: ignore
                inputs: Optional[torch.Tensor] = None,
                emotion_embed: Optional[torch.Tensor] = None,
                sequence_length: Optional[torch.LongTensor] = None,
                memory: Optional[torch.Tensor] = None,
                memory_sequence_length: Optional[torch.LongTensor] = None,
                memory_attention_bias: Optional[torch.Tensor] = None,
                cause_labels: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                context_sequence_length: Optional[torch.LongTensor] = None,
                helper: Optional[Helper] = None,
                decoding_strategy: str = 'train_greedy',
                max_decoding_length: Optional[int] = None,
                impute_finished: bool = False,
                infer_mode: Optional[bool] = None,
                beam_width: Optional[int] = None,
                length_penalty: float = 0.,
                **kwargs) \
            -> Union[
                TransformerDecoderOutput,
                Tuple[TransformerDecoderOutput, torch.LongTensor],
                Dict[str, torch.Tensor]]:

        if memory is not None:
            if memory_attention_bias is None:
                if memory_sequence_length is None:
                    raise ValueError(
                        "`memory_sequence_length` is required if "
                        "`memory_attention_bias` is not given.")

                enc_padding = 1 - sequence_mask(
                    memory_sequence_length, memory.size(1),
                    dtype=torch.float32)
                memory_attention_bias = attn.attention_bias_ignore_padding(
                    enc_padding)
        # print(memory_attention_bias.size())
        # print(cause_labels.size())
        cause_attention_bias = cause_labels
        # cause_attention_bias = cause_labels
        # record the context, which will be used in step function
        # for dynamic_decode
        if context is not None:
            if context_sequence_length is None:
                raise ValueError("'context_sequence_length' must not be None"
                                 "when 'context' is specified.")
            self._state_context = context[:, 1:]
            self._state_context_sequence_length = context_sequence_length - 1
        else:
            self._state_context = None
            self._state_context_sequence_length = None

        # Faster code path for teacher-forcing training
        if (helper is None and beam_width is None and
                decoding_strategy == 'train_greedy'):
            if inputs is None:
                raise ValueError("'input' must not be none "
                                 "when using 'train_greedy' decoding strategy.")
            times = torch.arange(
                inputs.size(1), dtype=torch.long, device=inputs.device)
            times = times.unsqueeze(0).expand(inputs.size(0), -1)
            inputs = self.embed_tokens(inputs, times)
            inputs += emotion_embed
            if sequence_length is not None:
                inputs = mask_sequences(inputs, sequence_length)

            decoder_self_attention_bias = (
                attn.attention_bias_lower_triangle(inputs.size(1)))

            decoder_output = self._self_attention_stack(
                inputs, memory, decoder_self_attention_bias,
                memory_attention_bias, cause_attention_bias, cache=None)
            logits = self._output_layer(decoder_output)
            sample_id = torch.argmax(logits, dim=-1)

            return TransformerDecoderOutput(logits, sample_id)

        # Inference code path.
        if max_decoding_length is None:
            max_decoding_length = self._hparams.max_decoding_length

        self._state_max_decoding_length = max_decoding_length

        if beam_width is None or beam_width == 1:  # Inference-like decoding
            # Prepare helper
            if helper is None:
                kwargs.update(decoding_strategy=decoding_strategy)
                if context is not None:
                    kwargs.update(start_tokens=context[:, 0])
                helper = self._create_or_get_helper(infer_mode, **kwargs)
            assert isinstance(helper, EmbeddingHelper)

            

            self._state_cache = self._init_cache(
                memory, memory_attention_bias, cause_attention_bias, emotion_embed,
                beam_search_decoding=False, batch_size=helper.batch_size)
            if context is not None:
                assert self._state_context is not None
                pad_length = max_decoding_length - self._state_context.size(1)
                if pad_length > 0:
                    self._state_context = torch.cat((
                        self._state_context,
                        self._state_context.new_zeros(
                            self._state_context.size(0), pad_length)
                    ), dim=1)

            outputs, cache, sequence_lengths = self.dynamic_decode(
                helper, inputs=None, sequence_length=None,
                initial_state=None, max_decoding_length=max_decoding_length,
                impute_finished=impute_finished)
            del cache  # not used

            if context is not None:
                # Here the length of sample_id will be larger than that
                # of logit by 1, because there will be a additional
                # start_token in the returned sample_id.
                # the start_id should be the first token of the
                # given context
                start_tokens = context[:, 0]
                outputs = TransformerDecoderOutput(
                    logits=outputs.logits,
                    sample_id=torch.cat([
                        start_tokens.unsqueeze(1),
                        outputs.sample_id
                    ], dim=1))
                sequence_lengths = sequence_lengths + 1

            return outputs, sequence_lengths

        else:  # Beam-search decoding
            # Ignore `decoding_strategy` and # assume `helper` is not set.
            if helper is not None:
                raise ValueError("Must not set 'beam_width' and 'helper' "
                                 "simultaneously.")
            if context is not None:
                start_tokens = context[:, 0]
            else:
                if 'start_tokens' not in kwargs:
                    raise ValueError(
                        "'start_tokens' must be specified when using"
                        "beam search decoding.")
                start_tokens = kwargs['start_tokens']
            _batch_size = start_tokens.size(0)
            self._state_cache = self._init_cache(
                memory, memory_attention_bias,
                beam_search_decoding=True,
                batch_size=_batch_size)
            end_token: int = kwargs.get('end_token')  # type: ignore

            # The output format is different when running beam search.
            sample_id, log_prob = self.beam_decode(
                start_tokens,
                end_token,
                embedding_fn=self.embed_tokens,
                beam_width=beam_width,
                length_penalty=length_penalty,
                decode_length=max_decoding_length)

            return {
                'sample_id': sample_id,
                'log_prob': log_prob
            }

    def _self_attention_stack(
            self, inputs: torch.Tensor,
            memory: Optional[torch.Tensor],
            decoder_self_attention_bias: Optional[torch.Tensor] = None,
            memory_attention_bias: Optional[torch.Tensor] = None,
            cause_attention_bias: Optional[torch.Tensor] = None,
            cache: Optional[Cache] = None) -> torch.Tensor:
        r"""Forward through the stacked multi-head attentions.
        """
        inputs = self.embed_dropout(inputs)
        if cache is not None:
            if memory is not None:
                memory_attention_bias = cache['memory_attention_bias']
                cause_attention_bias = cache['cause_attention_bias']
        else:
            assert decoder_self_attention_bias is not None

        x = inputs
        for i in range(self._hparams.num_blocks):
            layer_cache = cache['layers'][i] if cache is not None else None

            
            selfatt_output = self.self_attns[i](
                queries=self.self_attn_layer_norm[i](x),
                memory=None,
                memory_attention_bias=decoder_self_attention_bias,
                cache=layer_cache)
            x = x + self.residual_dropout(selfatt_output)

            if memory is not None:
                encdec_output = self.enc_dec_attns[i](
                    queries=self.end_dec_attn_layer_norm[i](x),
                    memory=memory,
                    memory_attention_bias=memory_attention_bias)
                x = x + self.residual_dropout(encdec_output)

                gated_memory = memory * cause_attention_bias.unsqueeze(-1)

                cause_output = self.cause_attns[i](
                    queries=self.cause_attn_layer_norm[i](x),
                    memory=gated_memory,
                    memory_attention_bias=memory_attention_bias)
                x = x + self.residual_dropout(cause_output)


            sub_output = self.poswise_networks[i](self.poswise_layer_norm[i](x))
            x = x + self.residual_dropout(sub_output)

        return self.final_layer_norm(x)


class EmotionVocab(tx.data.Vocab):
    def load(self, filename: str) \
            -> Tuple[Dict[int, str], Dict[str, int]]:
        r"""Loads the vocabulary from the file.

        Args:
            filename (str): Path to the vocabulary file.

        Returns:
            A tuple of mapping tables between word string and
            index, (:attr:`id_to_token_map_py`, :attr:`token_to_id_map_py`),
            where and :attr:`token_to_id_map_py` are python `defaultdict`
            instances.
        """
        with open(filename, "r") as vocab_file:
            vocab = list(line.strip() for line in vocab_file)

        warnings.simplefilter("ignore", UnicodeWarning)

        if self._bos_token in vocab:
            raise ValueError("Special begin-of-seq token already exists in the "
                             "vocabulary: '%s'" % self._bos_token)
        if self._eos_token in vocab:
            raise ValueError("Special end-of-seq token already exists in the "
                             "vocabulary: '%s'" % self._eos_token)
        if self._unk_token in vocab:
            raise ValueError("Special UNK token already exists in the "
                             "vocabulary: '%s'" % self._unk_token)
        if self._pad_token in vocab:
            raise ValueError("Special padding token already exists in the "
                             "vocabulary: '%s'" % self._pad_token)

        warnings.simplefilter("default", UnicodeWarning)

        # Places _pad_token at the beginning to make sure it take index 0.
        # vocab = [self._pad_token, self._bos_token, self._eos_token,
        #          self._unk_token] + vocab
        # Must make sure this is consistent with the above line
        vocab_size = len(vocab)

        # Creates python maps to interface with python code
        id_to_token_map_py = dict(zip(range(vocab_size), vocab))
        token_to_id_map_py = dict(zip(vocab, range(vocab_size)))

        return id_to_token_map_py, token_to_id_map_py