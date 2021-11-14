import argparse
import functools
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
import texar.torch as tx
from texar.torch.run import *

from model import Transformer, EmotionNet
import data_utils
from modules import EmotionVocab
import sklearn
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import metric_utils
from misc_utils import init_logger, logger
from bert_score import BERTScorer
import numpy as np
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config-model", type=str, default="config_model",
    help="The model config.")
parser.add_argument(
    "--config-data", type=str, default="config_data",
    help="The dataset config.")
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do-test", action="store_true",
    help="Whether to run test on the test set.")
parser.add_argument(
    "--output-dir", type=str, default="./outputs/",
    help="Path to save the trained model and logs.")
parser.add_argument(
    "--log-file", type=str, default="exp.log",
    help="Path to save the trained model and logs.")
parser.add_argument(
    '--checkpoint', type=str, default="./",
    help="Model checkpoint to load model weights from.")
parser.add_argument(
    '--emotion-model', type=str, default="/",
    help="Model checkpoint to load model weights from.")
parser.add_argument(
    '--bert-score-model', type=str, default="/",
    help="model for bert-score")
parser.add_argument(
    '--bert-score-baseline', type=str, default="/",
    help="rescale_baseline path for bert-score")
parser.add_argument(
    '--glove', type=str, default="/",
    help="glove embedding")


args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)
config_data.glove_file = args.glove

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

make_deterministic(config_model.random_seed)


scorer = BERTScorer(lang="en",num_layers=17, rescale_with_baseline=True, model_type=args.bert_score_model, 
                    baseline_path=args.bert_score_baseline)
output_dir = Path(args.output_dir)
tx.utils.maybe_create_dir(output_dir)
# tx.utils.maybe_create_dir(output_dir/"emotion")
tx.utils.maybe_create_dir(output_dir/"generation")

init_logger(output_dir/args.log_file)

class ModelWrapper(nn.Module):
    def __init__(self, generate_net: Transformer, emotion_net: EmotionNet, beam_width: int):
        super().__init__()
        self.generate_net = generate_net
        self.emotion_net = emotion_net
        self.beam_width = beam_width
        

    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        emotion_return_dict, emotion_preds, cause_preds = self.emotion_net(encoder_input=batch.src_text_ids,
                                                                           emotion_label=batch.emotion_id,
                                                                           cause_labels=batch.cause_ids)

        return_dict = self.generate_net(encoder_input=batch.src_text_ids,
                          emotion_preds=emotion_preds,
                          decoder_input=batch.tgt_text_ids[:,:-1].contiguous(),
                          labels=batch.tgt_text_ids[:,1:].contiguous(),
                          emotion_label=batch.emotion_id,
                          cause_labels=cause_preds,
                          user_ids=batch.user_ids)


        return_dict.update(emotion_return_dict)
        ppl = torch.exp(return_dict['mle_loss'])
        return_dict["ppl"] = ppl
        # import ipdb
        # ipdb.set_trace()
        return_dict['loss'] =  return_dict['mle_loss']+return_dict['emotion_loss']+return_dict['cause_loss']
        
   
        return return_dict, emotion_preds

    def predict(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        emotion_preds, cause_preds = self.emotion_net(encoder_input=batch.src_text_ids)

        predictions = self.generate_net(encoder_input=batch.src_text_ids,
                          emotion_preds=emotion_preds,
                          emotion_label=batch.emotion_id,
                          cause_labels=cause_preds,
                          user_ids=batch.user_ids)

        decoded_ids = predictions[0].sample_id
        hypos = self.generate_net.vocab.map_ids_to_tokens_py(decoded_ids.cpu()).tolist()
        final_hypos = []
        for h in hypos:
            if '<EOS>' in h:
                h = ' '.join(h[:h.index('<EOS>')])
            else:
                h = ' '.join(h[:20])
            final_hypos.append(h)
        
        # hypos = [' '.join(h[:h.index('<EOS>')]) for h in hypos]
        
        return {"preds": decoded_ids, "hypos":final_hypos}


def main() -> None:
    """Entry point.
    """
    # Load data
    vocab = tx.data.Vocab(config_data.vocab_file)
    emotion_vocab = EmotionVocab(config_data.emotion_file)
    logger.info(f"Vocab size: {vocab.size}")
    logger.info(f"EmotionVocab Size: {emotion_vocab.size}")
    train_data = data_utils.TrainData(config_data.train_hparams,device=device)
    valid_data = data_utils.TrainData(config_data.valid_hparams,device=device)
    test_data = data_utils.TrainData(config_data.test_hparams,device=device)
    logger.info(f"Training data size: {len(train_data)}")
    logger.info(f"Valid data size: {len(valid_data)}")
    logger.info(f"Test data size: {len(test_data)}")
    batching_strategy = data_utils.CustomBatchingStrategy(
        config_data.max_batch_tokens)
    train_data_iterator = tx.data.DataIterator(train_data, batching_strategy)
    valid_data_iterator = tx.data.DataIterator(valid_data)
    test_data_iterator = tx.data.DataIterator(test_data)

    # Create model and optimizer
    generate_net = Transformer(config_model, config_data, vocab, emotion_vocab)
    emotion_net = EmotionNet(config_model, config_data, vocab, emotion_vocab)
    if args.emotion_model is not None:
        emotion_net.load_state_dict(torch.load(args.emotion_model))
        logger.info("loading emotion model...")
    emotion_net.eval()
    model = ModelWrapper(generate_net, emotion_net, config_model.beam_width)
    model.to(device)
    # For training
    lr_config = config_model.lr_config
    if lr_config["learning_rate_schedule"] == "static":
        init_lr = lr_config["static_lr"]
        scheduler_lambda = lambda x: 1.0
    else:
        init_lr = lr_config["lr_constant"]
        scheduler_lambda = functools.partial(
            data_utils.get_lr_multiplier, warmup_steps=lr_config["warmup_steps"])
    optim = torch.optim.Adam(
        generate_net.parameters(), lr=init_lr, betas=(0.9, 0.997), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)


    
    def _save_epoch(epoch):
        logger.info("saving model...")
        torch.save(model.state_dict(), output_dir/f"checkpoint{epoch}.pt")
        logger.info("done!")

    def _save_emotion(ckpt_str):
        logger.info("saving emotion model...")
        torch.save(emotion_net.state_dict(), output_dir/f"emotion/best_{ckpt_str}.pt")
        logger.info("done!")

    def _save_generation(epoch):
        logger.info("saving generation  model...")
        torch.save(generate_net.state_dict(), output_dir/f"generation/checkpoint{epoch}.pt")
        logger.info("done!")

    def _train_epoch(epoch):
        model.train()

        step = 0
        avg_rec = tx.utils.AverageRecorder()
        all_emotion_preds = []
        all_emotion_targets = []
        best_test_acc = 0.0
        for batch in train_data_iterator:
            optim.zero_grad()
            return_dict, emotion_preds = model(batch)
            all_emotion_preds += torch.flatten(emotion_preds).tolist()
            all_emotion_targets += torch.flatten(batch.emotion_id).tolist()
            loss = return_dict['loss']
            loss.backward()
            optim.step()
            scheduler.step()
            avg_rec.add(return_dict)
            if step % config_data.display_steps == 0:
                logger.info(f"epoch: {epoch} | step: {step} | {avg_rec.to_str(precision=4, delimiter=' | ')}")
                avg_rec.reset()
            # if step % config_data.eval_steps == 0:
            #     val_acc, test_acc = _eval_epoch()
            #     if test_acc > best_test_acc and test_acc > 0.395:
            #         _save_emotion(f"val{val_acc:.4f}_test_{test_acc:.4f}")
            #         best_test_acc = test_acc
            step += 1
        
        # logger.info(sklearn.metrics.classification_report(all_emotion_targets,all_emotion_preds, labels=list(range(32)) ,target_names=[emotion_vocab.id_to_token_map_py[i] for i in range(32)]))

        
    @torch.no_grad()
    def _eval_epoch():
        model.eval()
        avg_rec = tx.utils.AverageRecorder()
        for batch in valid_data_iterator:
            return_dict, emotion_preds = model(batch)
            avg_rec.add(return_dict)
        logger.info(f"Eval | {avg_rec.to_str(precision=4, delimiter=' | ')}")
        # val_acc = avg_rec.avg('emotion_accu')
        avg_rec = tx.utils.AverageRecorder()
        all_emotion_preds = []
        all_emotion_targets = []
        for batch in test_data_iterator:
            return_dict, emotion_preds = model(batch)
            all_emotion_preds += torch.flatten(emotion_preds).tolist()
            all_emotion_targets += torch.flatten(batch.emotion_id).tolist()
            
            avg_rec.add(return_dict)
        
        logger.info(f"Test | {avg_rec.to_str(precision=4, delimiter=' | ')}")
        # test_acc = avg_rec.avg('emotion_accu')
        
        #return val_acc, test_acc

    def calc_metrics(all_hyps, all_refs):
        bleu1 = corpus_bleu(all_refs, all_hyps, weights=(1.0,0.0,0.0,0.0))
        bleu2 = corpus_bleu(all_refs, all_hyps, weights=(0.5,0.5,0.0,0.0))
        bleu3 = corpus_bleu(all_refs, all_hyps, weights=(0.333,0.333,0.333,0.0))
        bleu4 = corpus_bleu(all_refs, all_hyps)

        dist1,dist2 = metric_utils.calc_diversity(all_hyps)

        P, R, F1 = scorer.score([' '.join(h) for h in all_hyps], [[' '.join(r[0])] for r in all_refs])
        P = P.mean()
        R = R.mean()
        F1 = F1.mean()
        length = [len(h) for h in all_hyps]
        length = np.array(length)
        length = length.mean()
        logger.info(f"Test | bleu1={bleu1:.4f} | bleu2={bleu2:.4f} | bleu3={bleu3:.4f} | bleu4={bleu4:.4f}")

        logger.info(f"dist1={dist1:.4f} | dist2={dist2:.4f} | length={length:.4f}")
        logger.info(f"P={P.item():.4f} | R={R.item():.4f} | F1={F1.item():.4f}")
        return P, R, F1, dist2, bleu4

    @torch.no_grad()
    def _test_epoch(epoch):
        model.eval()
        sample_id = 0
        all_hyps = []
        all_refs = []
        for batch in tqdm(test_data_iterator):
            return_dict = model.predict(batch)
            for idx,h in enumerate(return_dict['hypos']):
                # print(f"S-{sample_id}\t{batch.src_text[idx]}")
                # print(f"T-{sample_id}\t{batch.tgt_text[idx]}")
                # print(f"H-{sample_id}\t{h}")
                all_hyps.append(h.split(' '))
                all_refs.append([batch.tgt_text[idx].split(' ')[1:-1]])
                sample_id+=1
        _, _, F1, dist2, bleu4 = calc_metrics(all_hyps, all_refs)

        return F1, dist2, bleu4

    @torch.no_grad()
    def _significance_test(epoch):
        
        # Create model and optimizer
        generate_net2 = Transformer(config_model, config_data, vocab, emotion_vocab)
        emotion_net2 = EmotionNet(config_model, config_data, vocab, emotion_vocab)
        if args.emotion_model is not None:
            emotion_net2.load_state_dict(torch.load(args.emotion_model))
            logger.info("loading emotion model...")
        emotion_net2.eval()
        model2 = ModelWrapper(generate_net2, emotion_net2, config_model.beam_width)
        model2.to(device)

        model.load_state_dict(torch.load(args.checkpoint))
        model2.load_state_dict(torch.load("./"))
        model.eval()
        model2.eval()
        sample_id = 0
        all_hyps = []
        all_refs = []
        all_hyps2 = []

        all_bleu4_1 = []
        all_bleu4_2 = []

        for batch in tqdm(test_data_iterator):
            return_dict = model.predict(batch)
            return_dict2 = model2.predict(batch)
            for idx,h in enumerate(return_dict['hypos']):
                # logger.info(f"S-{sample_id}\t{batch.src_text[idx]}")
                # logger.info(f"T-{sample_id}\t{batch.tgt_text[idx]}")
                # logger.info(f"H-{sample_id}\t{h}")
                all_hyps.append(h.split(' '))
                all_refs.append([batch.tgt_text[idx].split(' ')[1:-1]])
                all_bleu4_1.append(sentence_bleu([batch.tgt_text[idx].split(' ')[1:-1]],h.split(' ')))
                sample_id+=1
            for idx,h in enumerate(return_dict2['hypos']):
                # logger.info(f"S-{sample_id}\t{batch.src_text[idx]}")
                # logger.info(f"T-{sample_id}\t{batch.tgt_text[idx]}")
                # logger.info(f"H-{sample_id}\t{h}")
                all_hyps2.append(h.split(' '))
                all_bleu4_2.append(sentence_bleu([batch.tgt_text[idx].split(' ')[1:-1]],h.split(' ')))
        # bleu1 = corpus_bleu(all_refs, all_hyps, weights=(1.0,0.0,0.0,0.0))
        # bleu2 = corpus_bleu(all_refs, all_hyps, weights=(0.5,0.5,0.0,0.0))
        # bleu3 = corpus_bleu(all_refs, all_hyps, weights=(0.333,0.333,0.333,0.0))
        # bleu4 = corpus_bleu(all_refs, all_hyps)
        
        # dist1,dist2 = metric_utils.calc_diversity(all_hyps)

        P, R, F1 = scorer.score([' '.join(h) for h in all_hyps], [[' '.join(r[0])] for r in all_refs])

        P2, R2, F12 = scorer.score([' '.join(h) for h in all_hyps2], [[' '.join(r[0])] for r in all_refs])

        res = ttest_ind(P, P2)
        print(res)
        res = ttest_ind(R, R2)
        print(res)
        res = ttest_ind(F1, F12)
        print(res)
        res = ttest_ind(all_bleu4_1, all_bleu4_2)
        print(res)


    @torch.no_grad()
    def _test_model():
        model.load_state_dict(torch.load(args.checkpoint))
        model.to(device)
        model.eval()

        all_hyps, all_refs = [], []
        sample_id = 0
        for batch in test_data_iterator:
            return_dict = model.predict(batch)
            for idx,h in enumerate(return_dict['hypos']):
                logger.info(f"S-{sample_id}\t{batch.src_text[idx]}")
                logger.info(f"T-{sample_id}\t{batch.tgt_text[idx]}")
                logger.info(f"H-{sample_id}\t{h}")
                all_hyps.append(h.split(' '))
                all_refs.append([batch.tgt_text[idx].split(' ')[1:-1]])
                sample_id+=1
        
        _, _, _, _, _ = calc_metrics(all_hyps, all_refs)


    if args.do_train:
        best_F1 = 0
        for epoch in range(config_data.max_train_epoch):
            _train_epoch(epoch)
            _eval_epoch()
            F1, dist2, bleu4 = _test_epoch(epoch)
            
            if F1 > 0.13 and dist2 > 0.08 and bleu4 > 0.016 and F1 > best_F1:
                best_F1 = F1
                _save_generation("_best")
                _save_epoch("_best")
            # _test_epoch(epoch)

    if args.do_test:
        _test_model()
        # _significance_test(0)


if __name__ == '__main__':
    main()
