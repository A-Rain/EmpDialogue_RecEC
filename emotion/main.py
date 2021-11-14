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
from misc_utils import init_logger, logger

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
    '--checkpoint', type=str, default=None,
    help="Model checkpoint to load model weights from.")
parser.add_argument(
    '--glove', type=str, default="/",
    help="glove embedding")
args = parser.parse_args()

config_model: Any = importlib.import_module(args.config_model)
config_data: Any = importlib.import_module(args.config_data)
config_data.glove_file = args.glove

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

make_deterministic(config_model.random_seed)

output_dir = Path(args.output_dir)
tx.utils.maybe_create_dir(output_dir)
tx.utils.maybe_create_dir(output_dir/"emotion")
# tx.utils.maybe_create_dir(output_dir/"generation")

init_logger(output_dir/args.log_file)

class ModelWrapper(nn.Module):
    def __init__(self, generate_net: Transformer, emotion_net: EmotionNet, beam_width: int):
        super().__init__()
        self.generate_net = generate_net
        self.emotion_net = emotion_net
        self.beam_width = beam_width
        

    def forward(self,  # type: ignore
                batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        return_dict, emotion_preds = self.emotion_net(encoder_input=batch.src_text_ids,
                                                    emotion_label=batch.emotion_id,
                                                    cause_labels=batch.cause_ids,)


        return_dict['loss'] =  return_dict['emotion_loss'] + return_dict['cause_loss']
        return return_dict, emotion_preds

    def predict(self, batch: tx.data.Batch) -> Dict[str, torch.Tensor]:
        predictions = self.model(encoder_input=batch.src_text_ids,
                                 beam_width=self.beam_width)
        decoded_ids = predictions[0].sample_id
        hypos = self.model.vocab.map_ids_to_tokens_py(decoded_ids.cpu()).tolist()
        hypos = [' '.join(h[:h.index('<EOS>')]) for h in hypos]
        
        return {"preds": decoded_ids, "hypos":hypos}


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
        emotion_net.parameters(), lr=init_lr, betas=(0.9, 0.997), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, scheduler_lambda)


    encoding = getattr(config_data, 'encoding', None)
    def _save_epoch(epoch):
        logger.info("saving model...")
        torch.save(model.state_dict(), output_dir/f"checkpoint{epoch}.pt")
        logger.info("done!")

    def _save_emotion(ckpt_str):
        logger.info("saving emotion model...")
        torch.save(emotion_net.state_dict(), output_dir/f"emotion/best_{ckpt_str}.pt")
        logger.info("done!")

    def _save_generation(ckpt_str):
        logger.info("saving generation  model...")
        torch.save(model.state_dict(), output_dir/f"generation/checkpoint{epoch}.pt")
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
                curr_acc = return_dict["emotion_accu"].item()
                if curr_acc > best_test_acc:
                    _save_emotion("emotion")
                logger.info(f"epoch: {epoch} | step: {step} | {avg_rec.to_str(precision=4, delimiter=' | ')}")
                avg_rec.reset()
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
        val_acc = avg_rec.avg('emotion_accu')
        avg_rec = tx.utils.AverageRecorder()
        all_emotion_preds = []
        all_emotion_targets = []
        for batch in test_data_iterator:
            return_dict, emotion_preds = model(batch)
            all_emotion_preds += torch.flatten(emotion_preds).tolist()
            all_emotion_targets += torch.flatten(batch.emotion_id).tolist()
            
            avg_rec.add(return_dict)
        
        logger.info(f"Test | {avg_rec.to_str(precision=4, delimiter=' | ')}")
        test_acc = avg_rec.avg('emotion_accu')
        
        # logger.info(sklearn.metrics.classification_report(all_emotion_targets,all_emotion_preds, labels=list(range(32)) ,target_names=[emotion_vocab.id_to_token_map_py[i] for i in range(32)]))
        return val_acc, test_acc

    @torch.no_grad()
    def _test_epoch(epoch):
        model.eval()
        for batch in test_data_iterator:
            return_dict = model.predict(batch)

            logger.info(return_dict['preds'])
    @torch.no_grad()
    def _test_model():
        model.load_state_dict(torch.load(args.checkpoint))
        model.to(device)
        model.eval()
        sample_id = 0
        for batch in test_data_iterator:
            return_dict = model.predict(batch)
            for idx,h in enumerate(return_dict['hypos']):
                logger.info(f"S-{sample_id}\t{batch.src_text[idx]}")
                logger.info(f"T-{sample_id}\t{batch.tgt_text[idx]}")
                logger.info(f"H-{sample_id}\t{h}")
                sample_id+=1

    if args.do_train:
        for epoch in range(config_data.max_train_epoch):
            _train_epoch(epoch)
            _eval_epoch()


    if args.do_test:
        _test_model()


if __name__ == '__main__':
    main()
