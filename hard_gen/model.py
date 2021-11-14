from typing import Optional

import torch
from torch import nn
import texar.torch as tx
import torch.nn.functional as F

from modules import TransformerDecoder, TransformerEncoder
from texar.torch.data import embedding
import numpy as np

class EmotionNet(nn.Module):
    def __init__(self, model_config, data_config, vocab: tx.data.Vocab, emotion_vocab: tx.data.Vocab):
        super().__init__()
        self.config_model = model_config
        self.config_data = data_config
        self.vocab = vocab
        self.emotion_vocab = emotion_vocab
        self.vocab_size = vocab.size
        self.num_emotion_classes = self.emotion_vocab.size
        glove_embedding = np.random.rand(self.vocab_size, self.config_model.word_dim).astype('f')
        glove_embedding = embedding.load_glove(self.config_data.glove_file, self.vocab.token_to_id_map_py, glove_embedding)

        self.word_embedder = tx.modules.WordEmbedder(
            init_value=torch.from_numpy(glove_embedding),
            vocab_size=self.vocab_size,
            hparams=self.config_model.emb)

        self.emotion_encoder = TransformerEncoder(
            hparams=self.config_model.emotion_encoder)
        self.emotion_pred_layer = nn.Linear(self.config_model.hidden_dim, self.num_emotion_classes)
        self.cause_pred_layer = nn.Linear(self.config_model.hidden_dim, 2)
        self.emotion_loss_func = NvidiaLabelSmoothing(0.1)

    def forward(self,  # type: ignore
                encoder_input: torch.Tensor,
                emotion_label: Optional[torch.LongTensor] = None,
                cause_labels: Optional[torch.LongTensor] = None,
                ):

        batch_size = encoder_input.size(0)
        # Text sequence length excluding padding
        encoder_input_length = (encoder_input != 0).int().sum(dim=1)
        # positions = torch.arange(
        #     encoder_input_length.max(), dtype=torch.long,
        #     device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)


        emotion_input_embedding = self.word_embedder(encoder_input)
        emotion_encoder_output, emotion_attn_logits = self.emotion_encoder(
            inputs=emotion_input_embedding, sequence_length=encoder_input_length)

        
        emotion_cls = emotion_encoder_output[:,0,:]
        emotion_logits = self.emotion_pred_layer(emotion_cls)

        cause_outputs = emotion_encoder_output
        cause_logits = self.cause_pred_layer(cause_outputs)
        # cause_logits = emotion_attn_logits[:,0,0,:]

        if emotion_label is not None:

            emotion_preds = torch.argmax(emotion_logits, dim=-1)
            emotion_accu = tx.evals.accuracy(emotion_label, torch.flatten(emotion_preds))
            emotion_loss = self.emotion_loss_func(emotion_logits, emotion_label.view(-1))


            label_lengths = encoder_input_length - 1
            is_target = (encoder_input != 0).float()

            cause_label_lengths = label_lengths
            is_cause_target = is_target[:,1:]

            # print(cause_labels.size())
            cause_loss = tx.losses.sequence_softmax_cross_entropy(
                cause_labels[:,1:].unsqueeze(-1), cause_logits[:,1:], cause_label_lengths,
                average_across_batch=False, sum_over_timesteps=False,)
            cause_loss = (cause_loss * is_cause_target).sum() / (is_cause_target.sum())
            cause_logits = F.gumbel_softmax(cause_logits, hard=True)
            cause_preds = cause_logits[:,:,1]
            return {'emotion_loss':emotion_loss, 'cause_loss': cause_loss,'emotion_accu':emotion_accu}, emotion_preds, cause_preds

        else:
            emotion_preds = torch.argmax(emotion_logits, dim=-1)
            cause_logits = F.gumbel_softmax(cause_logits, hard=True)
            cause_preds = cause_logits[:,:,1]

            return emotion_preds, cause_preds

    



class Transformer(nn.Module):


    def __init__(self, model_config, data_config, vocab: tx.data.Vocab, emotion_vocab: tx.data.Vocab, use_mmoe=False):
        super().__init__()

        self.config_model = model_config
        self.config_data = data_config
        self.vocab = vocab
        self.emotion_vocab = emotion_vocab
        self.vocab_size = vocab.size
        self.num_emotion_classes = self.emotion_vocab.size
        glove_embedding = np.random.rand(self.vocab_size, self.config_model.word_dim).astype('f')
        glove_embedding = embedding.load_glove(self.config_data.glove_file, self.vocab.token_to_id_map_py, glove_embedding)
        self.word_embedder = tx.modules.WordEmbedder(
            init_value=torch.from_numpy(glove_embedding),
            vocab_size=self.vocab_size,
            hparams=self.config_model.emb)
        
        self.pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.config_data.max_decoding_length,
            hparams=self.config_model.position_embedder_hparams)
        self.emotion_embedder = tx.modules.WordEmbedder(
            vocab_size=self.emotion_vocab.size,
            hparams=self.config_model.emb)
        self.user_embedder = tx.modules.WordEmbedder(
            vocab_size=3,
            hparams=self.config_model.emb)
        self.encoder = tx.modules.TransformerEncoder(
            hparams=self.config_model.encoder)
        self.decoder = TransformerDecoder(
            token_pos_embedder=self._embedding_fn,
            vocab_size=self.vocab_size,
            output_layer=self.word_embedder.embedding,
            hparams=self.config_model.decoder)


        self.smoothed_loss_func = LabelSmoothingLoss(
            label_confidence=self.config_model.loss_label_confidence,
            tgt_vocab_size=self.vocab_size,
            ignore_index=0)

        self.emotion_loss_func = NvidiaLabelSmoothing(0.1)

        self.cause_loss_func = LabelSmoothingLoss(
            label_confidence=self.config_model.loss_label_confidence,
            tgt_vocab_size=3,
            ignore_index=0)

    def _embedding_fn(self, tokens: torch.LongTensor,
                      positions: torch.LongTensor) -> torch.Tensor:
        word_embed = self.word_embedder(tokens)
        #word_embed = self.input_layer(word_embed)
        # scale = self.config_model.hidden_dim ** 0.5
        pos_embed = self.pos_embedder(positions)
        
        return word_embed + pos_embed

    def forward(self,  # type: ignore
                encoder_input: torch.Tensor,
                emotion_preds: torch.Tensor,
                decoder_input: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                emotion_label: Optional[torch.LongTensor] = None,
                cause_labels: Optional[torch.LongTensor] = None,
                user_ids: Optional[torch.LongTensor] = None,
                beam_width: Optional[int] = None,
                ):

        batch_size = encoder_input.size(0)
        # Text sequence length excluding padding
        encoder_input_length = (encoder_input != 0).int().sum(dim=1)
        positions = torch.arange(
            encoder_input_length.max(), dtype=torch.long,
            device=encoder_input.device).unsqueeze(0).expand(batch_size, -1)

        # Source word embedding
        src_input_embedding = self._embedding_fn(encoder_input, positions)

        emotion_embed = self.emotion_embedder(emotion_preds.unsqueeze(-1))
        src_input_embedding = src_input_embedding + emotion_embed.expand_as(src_input_embedding)

        encoder_output = self.encoder(
            inputs=src_input_embedding, sequence_length=encoder_input_length)

        if decoder_input is not None and labels is not None:
            # enter the training logic

            # For training
            outputs = self.decoder(
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                inputs=decoder_input,
                emotion_embed=emotion_embed,
                cause_labels=cause_labels,
                decoding_strategy="train_greedy",
            )
            label_lengths = (labels != 0).long().sum(dim=1)
            is_target = (labels != 0).float()
            mle_loss = self.smoothed_loss_func(
                outputs.logits, labels, label_lengths)
            mle_loss = (mle_loss * is_target).sum() / is_target.sum()
        
            return {'mle_loss': mle_loss}

        else:
            start_tokens = encoder_input.new_full(
                (batch_size,), self.vocab.bos_token_id)

            helper = tx.modules.TopKSampleEmbeddingHelper(
            start_tokens=start_tokens, end_token=self.vocab.eos_token_id,
            top_k=3, softmax_temperature=0.6)

            predictions = self.decoder(
                memory=encoder_output, 
                memory_sequence_length=encoder_input_length,
                emotion_embed=emotion_embed,
                cause_labels=cause_labels,
                start_tokens=start_tokens,
                end_token=self.vocab.eos_token_id,
                max_decoding_length=self.config_data.max_decoding_length,
                helper=helper,
                # decoding_strategy="infer_greedy",
            )
            # Uses the best sample by beam search
            return predictions


class LabelSmoothingLoss(nn.Module):
    r"""With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    Args:
        label_confidence: the confidence weight on the ground truth label.
        tgt_vocab_size: the size of the final classification.
        ignore_index: The index in the vocabulary to ignore weight.
    """
    one_hot: torch.Tensor

    def __init__(self, label_confidence, tgt_vocab_size, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.tgt_vocab_size = tgt_vocab_size

        label_smoothing = 1 - label_confidence
        assert 0.0 < label_smoothing <= 1.0
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))
        self.confidence = label_confidence

    def forward(self,  # type: ignore
                output: torch.Tensor,
                target: torch.Tensor,
                label_lengths: torch.LongTensor) -> torch.Tensor:
        r"""Compute the label smoothing loss.
        Args:
            output (FloatTensor): batch_size x seq_length * n_classes
            target (LongTensor): batch_size * seq_length, specify the label
                target
            label_lengths(torch.LongTensor): specify the length of the labels
        """
        orig_shapes = (output.size(), target.size())
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob = model_prob.to(device=target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        output = output.view(orig_shapes[0])
        model_prob = model_prob.view(orig_shapes[0])

        return tx.losses.sequence_softmax_cross_entropy(
            labels=model_prob,
            logits=output,
            sequence_length=label_lengths,
            average_across_batch=False,
            sum_over_timesteps=False,
        )

class NvidiaLabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(NvidiaLabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

