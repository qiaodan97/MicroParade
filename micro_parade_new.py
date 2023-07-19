# ===================
# Contains the model architecture and training code
# Author: @liamhebert
# ===================


import argparse
import torch
import transformers
from fairseq.modules import PositionalEmbedding
import pytorch_lightning as pl
from torchmetrics import AveragePrecision
from data_module import RecsysDataset
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import math
import dask.dataframe as dd
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger


class OutputHead(torch.nn.Module):
    """
    Used to transform meta-embedding [CLS] to a class prediction.
    Consists of Meta FF -> Meta + NVTabular FF -> Logit
    """

    def __init__(self, dropout, hidden_dim):
        super().__init__()

        self.feat_ff = torch.nn.ModuleList()
        # 192 + 55 (243) -> 330 -> 217 -> 143 -> 69 -> 32
        sizes = [55, 64, 128, 256, 512, 256, 64, 32, 16, 8]
        for start, end in zip(sizes[:-1], sizes[1:]):
            self.feat_ff.append(torch.nn.Linear(start, end))
            self.feat_ff.append(torch.nn.LayerNorm(end))
            self.feat_ff.append(torch.nn.ReLU())
            # self.feat_ff.append(torch.nn.Dropout(p=dropout))

        # 32 -> 1
        self.ff_output = torch.nn.Linear(sizes[-1], 1)  # sigmoid
        # sigmoid()

    def forward(self, text_embedding, nv_tabular):
        # x = text_embedding
        # # Meta FF
        # for layer in self.language_ff:
        #     x = layer(x)

        # x = torch.cat((x, nv_tabular), dim=1)
        # NVTabular + Meta FF
        x = nv_tabular
        for layer in self.feat_ff:
            x = layer(x)

        # Output predictions
        x = self.ff_output(x)
        x = x.view(x.size(0), -1)

        return x


class MicroParade(pl.LightningModule):
    """
    Where the magic happens! Class describes optimizer, model architecture and train/valid/test loops
    Model architecture consists of:
        - Tweet Text Cross Embeddings ([CLS]<query>[SEP]<history>[SEP]) with Transformer
        - Time Encodings (As "position" encodings)
        - Meta Tweet Transformer (To aggregate history into a [CLS] representation)
        - Output Heads (To make predictions, in this case, "like", "comment", "retweet", "retweet-with-comment", )
    This is trained end-to-end, but can be reduced by "freezing" language model
    Trained using BCE Loss and evaluated using AP
    """

    def __init__(self,
                 transformer_nheads,
                 num_transformer_layers,
                 output_classes,
                 adam_epsilon,
                 dropout,
                 hidden_dim,
                 lr,
                 weight_list):
        self.transformer_nheads = transformer_nheads
        self.num_transformer_layers = num_transformer_layers
        self.output_classes = output_classes
        self.adam_epsilon = adam_epsilon
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.lr = lr

        super(MicroParade, self).__init__()

        self.output_heads = torch.nn.ModuleList(
            [OutputHead(self.dropout, self.hidden_dim) for _ in range(self.output_classes)])

        # Evaluation metrics
        self.metrics = AveragePrecision(task='binary', num_labels=self.output_classes, average=None)

        # Loss function
        # self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([weight_list[0],
        #                                                                    weight_list[1],
        #                                                                    weight_list[2],
        #                                                                    weight_list[3]]))  # This is weighted to account for class imbalance
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([weight_list[3]]))  # This is weighted to account for class imbalance
        print("Training weight: ", weight_list)
        # first increase the lr
        # gradient vanishing
        # gradient changing weight randomly
        # graph training error vs validation error

        # self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([35.436611946475374,
        #                                                                    9.04134807583367,
        #                                                                    82.02527887007463,
        #                                                                    1.2184770791472297]))  # This is weighted to account for class imbalance

    def forward(self, input, nv_tabular):
        """
        Parameters:
        - input: [Batch x queries x length], Query Target input tokens assumed already tokenized
        - history_times: Times when a given historic tweet was engaged
        - nv_tabular: extra features from nv_tabular data

        Returns:
        - Relevance Score for each prediction type
        """
        x = input
        pre_head = []
        for head in self.output_heads:
            pre_head.append(head(x, nv_tabular))

        pre_head_tuple = tuple(pre_head)
        x = torch.cat(pre_head_tuple, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        nv_tabular, y = batch
        token_type_ids = None
        y_hat = self.forward(input,
                             nv_tabular)

        # y_hat = y_hat.sigmoid()
        loss = self.loss_fn(y_hat, y)  # SELF.ENCODER
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        nv_tabular, y = batch
        y_hat = self.forward(input,
                             nv_tabular)

        y_hat = y_hat.sigmoid()
        self.metrics.update(y_hat, y.to(torch.int))

    def validation_epoch_end(self, outputs):
        temp_result = self.metrics.compute()
        # record the tensor, not caring whether it's non or not
        self.log('test_like_precision', temp_result, on_epoch=True)
        self.log('valid_ap', temp_result, on_epoch=True)

    def test_step(self, batch, batch_idx):
        nv_tabular, y = batch

        y_hat = self.forward(input,
                             nv_tabular)
        y_hat = y_hat.sigmoid()
        self.metrics.update(y_hat, y.to(torch.int))

    def test_epoch_end(self, outputs):
        temp_result = self.metrics.compute()
        # record the tensor, not caring whether it's non or not
        self.log('test_like_precision', temp_result, on_epoch=True)
        self.log('valid_ap', temp_result, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, eps=self.adam_epsilon)


class TimeEncoding(torch.nn.Module):
    """
    Encodes the time of day that a tweet was interacted with and adds it to the input
    """

    def __init__(self, hidden):
        super().__init__()
        # learnable embeddings for each time
        self.position_embeddings = PositionalEmbedding(25, hidden, None, learned=True)

    def forward(self, inputs, times):
        embeddings = self.position_embeddings(inputs, positions=times.long())
        inputs[:, 1:, :] = inputs[:, 1:, :] + embeddings[:, 1:, :]
        return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--dataset_file", default='final_recsys_dataset.pt', type=str)
    parser.add_argument("--output_folder", default="final_models/", type=str)
    parser.add_argument("--target_tokenizer",
                        default="/lustre04/scratch/qiaodan/Multilingual-MiniLM-L12-H384/",
                        # default="/home/qiaodan/projects/def-emilios/qiaodan/MicroParade/code/bert-base-multilingual-cased/",
                        type=str)
    parser.add_argument("--root_folder", default="/lustre04/scratch/qiaodan/",
                        type=str)
    parser.add_argument("--data_folder", default="/preprocess_result_new1104/", type=str)
    parser.add_argument("--hidden_dim", default=768, type=int)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--valid_batch_size", default=128, type=int)
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument("--output_classes", default=1, type=int)
    parser.add_argument("--checkpoint_epoch", default=1, type=int)
    parser.add_argument("--transformer_nheads", default=4, type=int)
    parser.add_argument("--num_transformer_layers", default=3, type=int)
    parser.add_argument("--num_feed_forward_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.35, type=float)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--checkpoint", default='', action="store_true")
    parser.add_argument("--seed", default=44, type=int)
    parser.add_argument("--split_percent_train", default=0.8)  # train:test:valid = 8:1:1
    parser.add_argument("--split_percent_test", default=0.1)  # Here we are using it for subset
    parser.add_argument("--comment", default='', action="store_true")
    parser.add_argument('--tovalidate', action='store_true')
    parser.add_argument('--totest', action='store_true')
    parser.add_argument('--persistent_workers', default=True)
    parser.add_argument('--pin_memory', default=True)

    # args = parser.parse_args()

    log_dir = '/lustre04/scratch/qiaodan/tensorboardlog'
    logger = TensorBoardLogger(save_dir=log_dir)

    # from pytorch_lightning import Trainer
    # from argparse import ArgumentParser

    # Create an ArgumentParser instance
    # parser = ArgumentParser()

    # parser = pl.Trainer.add_argparse_args_to_parser(parser)

    args = parser.parse_args()
    # Create the Trainer and add the argparse arguments
    # trainer = Trainer()
    # trainer.add_argparse_args_to_parser(parser)

    # Parse the command-line arguments
    # args = parser.parse_args()

    # Use the parsed arguments
    # trainer.fit(model, datamodule)

    print(args)
    # for reproduce-ability
    pl.seed_everything(args.seed)

    # load data
    dm = RecsysDataset.from_argparse_args(args)
    weight_list = [35.436611946475374, 9.04134807583367, 82.02527887007463,
                   1.2184770791472297]  # TODO: change to last class, check weight initializing
    # load model
    if args.checkpoint != '':
        model = MicroParade.load_from_checkpoint(
            checkpoint=args.checkpoint,
            transformer_nheads=args.transformer_nheads,
            num_transformer_layers=args.num_transformer_layers,
            output_classes=args.output_classes,
            adam_epsilon=args.adam_epsilon,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_list=weight_list
        )
    else:
        model = MicroParade(
            transformer_nheads=args.transformer_nheads,
            num_transformer_layers=args.num_transformer_layers,
            output_classes=args.output_classes,
            adam_epsilon=args.adam_epsilon,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            weight_list=weight_list
        )

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # automatic checkpointing
    metrics = 'valid_ap'
    dirpath = args.comment + f'lightning_logs/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor=metrics,
        dirpath=dirpath,
        filename=args.comment + '-{epoch:03d}-{' + metrics + ':.4f}',
        save_top_k=2,
        mode='max',
        save_last=True,
    )
    if os.path.exists(dirpath + '/last.ckpt'):
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)

    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    if args.totest:
        result = trainer.test(model, datamodule=dm)
        pprint(result)
    elif args.tovalidate:
        result = trainer.validate(model, datamodule=dm)
        pprint(result)
    else:
        trainer.fit(model, datamodule=dm)