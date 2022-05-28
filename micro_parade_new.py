# ===================
# Contains the model architecture and training code
# Author: @liamhebert
# ===================


import argparse
import torch
import transformers
import time
from torch.utils.data import DataLoader
from recsys_dataset import RecSysMasterDataset
from fairseq.modules import PositionalEmbedding
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score, log_loss
from torchmetrics import AveragePrecision
from datasets_new import RecsysDataset
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os


class OutputHead(torch.nn.Module):
    """
    Used to transform meta-embedding [CLS] to a class prediction.
    Consists of Meta FF -> Meta + NVTabular FF -> Logit
    """

    def __init__(self, dropout, hidden_dim):
        super().__init__()

        self.language_ff = torch.nn.ModuleList()

        # 768 -> 682 -> 450 -> 297 -> 192
        sizes = [hidden_dim, 682, 450, 297, 192]
        for start, end in zip(sizes[:-1], sizes[1:]):
            self.language_ff.append(torch.nn.Linear(start, end))
            self.language_ff.append(torch.nn.LayerNorm(end))
            self.language_ff.append(torch.nn.ReLU())
            self.language_ff.append(torch.nn.Dropout(p=dropout))

        self.feat_ff = torch.nn.ModuleList()
        # 192 + 55 (243) -> 330 -> 217 -> 143 -> 69 -> 32
        sizes = [243, 330, 217, 143, 69, 32]
        for start, end in zip(sizes[:-1], sizes[1:]):
            self.feat_ff.append(torch.nn.Linear(start, end))
            self.feat_ff.append(torch.nn.LayerNorm(end))
            self.feat_ff.append(torch.nn.ReLU())
            self.feat_ff.append(torch.nn.Dropout(p=dropout))

        # 32 -> 1
        self.ff_output = torch.nn.Linear(sizes[-1], 1)

    def forward(self, text_embedding, nv_tabular):
        x = text_embedding
        # Meta FF
        for layer in self.language_ff:
            x = layer(x)

        x = torch.cat((x, nv_tabular), dim=1)
        # NVTabular + Meta FF
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
                 # encoder_model,
                 transformer_nheads,
                 num_transformer_layers,
                 output_classes,
                 adam_epsilon,
                 dropout,
                 hidden_dim,
                 lr,
                 target_tokenizer):
        # self.encoder_model = encoder_model TODO:???
        self.transformer_nheads = transformer_nheads
        self.num_transformer_layers = num_transformer_layers
        self.output_classes = output_classes
        self.adam_epsilon = adam_epsilon
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.target_tokenizer = target_tokenizer

        super(MicroParade, self).__init__()
        # self.save_hyperparameters()  # moves all model args to "self"
        # Tweet Encoder
        self.encoder = transformers.AutoModel.from_pretrained(self.target_tokenizer)
        # Time Encoder 
        self.time_encoding = TimeEncoding(self.hidden_dim)
        # Meta Transformer Aggregator 
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.transformer_nheads)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, self.num_transformer_layers)
        # Output prediction heads (one per class)
        self.output_heads = torch.nn.ModuleList(
            [OutputHead(self.dropout, self.hidden_dim) for _ in range(self.output_classes)])

        # Evaluation metric
        self.metric = AveragePrecision(num_classes=self.output_classes, average=None)

        # Loss function
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([35.436611946475374,
                                                                           9.04134807583367,
                                                                           82.02527887007463,
                                                                           1.2184770791472297]))  # This is weighted to account for class imbalance

    def forward(self, input, token_type_ids, attn_mask, history_times, history_mask, nv_tabular):
        """
        Parameters:
        - input: [Batch x queries x length], Query Target input tokens assumed already tokenized
        - history_times: Times when a given historic tweet was engaged 
        - nv_tabular: extra features from nv_tabular data

        Returns:
        - Relevance Score for each prediction type 
        """

        input_size = input.size()

        # reshape data to fit in one pass [batch * queries, length]
        def compress(x: torch.Tensor):
            size = x.size()
            return torch.squeeze(x.reshape(1, size[0] * size[1], size[2]), 0)

        # create tweet embeddings
        x = self.encoder(input_ids=compress(input), attention_mask=compress(attn_mask),
                         token_type_ids=compress(token_type_ids))

        # extract cls of each tweet pair
        x = x.last_hidden_state[:, 0, :]
        # return back to normal shape
        x = x.view(input_size[0], int(x.shape[0] / input_size[0]), x.shape[1])
        # add time embeddings for position
        x = self.time_encoding(x, history_times)
        # reshape to fit transformer
        x = x.permute(1, 0, 2)
        # create meta embeddings
        x = self.transformer(x, src_key_padding_mask=history_mask)
        # get [cls] representation of meta embedding
        x = x[0, :, :]  # final cls token
        # get predictions per class
        x = torch.cat((head(x, nv_tabular) for head in self.output_heads), dim=1)

        return x

    def training_step(self, batch, batch_idx):
        input, mask, token_type_ids, history_time, history_mask, nv_tabular, y = batch

        y_hat = self.forward(input,
                             token_type_ids,
                             mask,
                             history_time,
                             history_mask,
                             nv_tabular)

        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def valid_step(self, batch, batch_idx):
        input, mask, token_type_ids, history_time, history_mask, nv_tabular, y = batch

        y_hat = self.forward(input,
                             token_type_ids,
                             mask,
                             history_time,
                             history_mask,
                             nv_tabular)

        y_hat = y_hat.sigmoid()
        self.metric.update(y_hat, y)

    def valid_epoch_end(self, outputs):
        self.log('valid_ap', self.metric, on_epoch=True)

    def test_step(self, batch, batch_idx):
        input, mask, token_type_ids, history_time, history_mask, nv_tabular, y = batch

        y_hat = self.forward(input,
                             token_type_ids,
                             mask,
                             history_time,
                             history_mask,
                             nv_tabular)

        y_hat = y_hat.sigmoid()
        self.metric.update(y_hat, y)

    def test_epoch_end(self, outputs):
        self.log('test_ap', self.metric, on_epoch=True)

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
    # parser.add_argument("--output_folder", default='.\\final_models\\')
    parser.add_argument("--output_folder", default='/final_models/')
    parser.add_argument("--target_tokenizer", default='bert-base-multilingual-cased1')
    # parser.add_argument("--root_folder", default='gs://micro-parade-data/recsys2021/')
    parser.add_argument("--root_folder", default='/mnt/d/summer2022/MicroParade1/')
    # parser.add_argument("--root_folder", default='/lustre03/project/6001735/qiaodan/MicroParade1/')
    # parser.add_argument("--data_folder", default='val/')
    parser.add_argument("--data_folder", default='datasets/recsys2021/val-4/')
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--hidden_dim", default=768, type=int)
    parser.add_argument("--valid_batch_size", default=32, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--output_classes", default=2, type=int)
    parser.add_argument("--checkpoint_epoch", default=1, type=int)
    parser.add_argument("--transformer_nheads", default=4, type=int)
    parser.add_argument("--num_transformer_layers", default=3, type=int)
    parser.add_argument("--num_feed_forward_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.35, type=float)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--checkpoint", default='')
    parser.add_argument("--seed", default=44, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--split_percent", default=0.7)
    parser.add_argument("--comment", default='') # What should this  be?
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)

    args = parser.parse_args()

    print(args)
    # for reproduce-ability
    pl.seed_everything(args.seed)

    # load data 
    dm = RecsysDataset.from_argparse_args(args)

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
            target_tokenizer=args.target_tokenizer
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
            target_tokenizer=args.target_tokenizer
        )

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # automatic checkpointing
    metric = 'valid_ap'
    dirpath = args.comment + f'/lightning_logs/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=dirpath,
        filename=args.comment + '-{epoch:03d}-{' + metric + ':.4f}',
        save_top_k=2,
        mode='max',
        save_last=True,
    )
    if os.path.exists(dirpath + '/last.ckpt'):
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)

    # trainer initialization
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    if args.test:
        result = trainer.test(model, datamodule=dm)
        pprint(result)
    elif args.validate:
        result = trainer.validate(model, datamodule=dm)
        pprint(result)
    else:
        trainer.fit(model, datamodule=dm)
