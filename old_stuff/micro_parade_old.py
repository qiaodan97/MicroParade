import argparse

import torch
import transformers
import time
from torch.utils.data import DataLoader
from recsys_dataset import RecSysMasterDataset
from fairseq.modules import PositionalEmbedding


class OutputHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_ff = torch.nn.ModuleList()


        sizes = [768, 1024, 682, 450, 297, 192]  # 768 -> 384 -> 192
        for start, end in zip(sizes[:-1], sizes[1:]):
            self.language_ff.append(torch.nn.Linear(start, end))
            self.language_ff.append(torch.nn.LayerNorm(end))
            self.language_ff.append(torch.nn.ReLU())
            self.language_ff.append(torch.nn.Dropout(p=0.35))
 
        self.feat_ff = torch.nn.ModuleList()
        sizes = [243, 330, 217, 143, 69, 32]  # 192 + 55 (243) -> 185 -> 138 -> 69 -> 32 -> 1
        for start, end in zip(sizes[:-1], sizes[1:]):
            self.feat_ff.append(torch.nn.Linear(start, end))
            self.feat_ff.append(torch.nn.LayerNorm(end))
            self.feat_ff.append(torch.nn.ReLU())
            self.feat_ff.append(torch.nn.Dropout(p=0.35))
        
        self.ff_output = torch.nn.Linear(sizes[-1], 1)
    
    def forward(self, text_embedding, nv_tabular):
        x = text_embedding
        for layer in self.language_ff:
            x = layer(x)
        # print('Transformer Head', x, flush=True)
        x = torch.cat((x, nv_tabular), dim=1)

        for layer in self.feat_ff:
            x = layer(x)
        # print('Combined Head', x, flush=True)
        x = self.ff_output(x)
        x = x.view(x.size(0), -1)
    
        return x



class MicroParade(torch.nn.Module):
    def __init__(self, args):
        super(MicroParade, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained(args.target_tokenizer)
        self.encoder.config.xla_device = True
        self.time_encoding = TimeEncoding()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=args.transformer_nheads)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, args.num_transformer_layers)
        
        self.output_heads = torch.nn.ModuleList([OutputHead() for _ in range(4)])
        

    def forward(self, input, token_type_ids, attn_mask, history_times, history_mask, nv_tabular):
        """
        :param training:
        :param history_mask:
        :param history_times:
        :param attn_mask:
        :param token_type_ids:
        :param input: [Batch x queries x length], Query Target input tokens assumed already tokenized
        :return: relevance_score, single score denoting relevance
        """
        input_size = input.size()

        def compress(x: torch.Tensor):
            size = x.size()
            return torch.squeeze(x.reshape(1, size[0] * size[1], size[2]), 0)

        x = self.encoder(input_ids=compress(input), attention_mask=compress(attn_mask),
                         token_type_ids=compress(token_type_ids))

        x = x.last_hidden_state[:, 0, :]  # cls token for encoding
        x = x.view(input_size[0], int(x.shape[0] / input_size[0]), x.shape[1])
        x = self.time_encoding(x, history_times)
        x = x.permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=history_mask)  # might need attention mask here
        x = x[0, :, :]  # final cls token
        # print('Transformer', x, flush=True)
        
        x = torch.cat((self.output_heads[0](x, nv_tabular), 
                       self.output_heads[1](x, nv_tabular),
                       self.output_heads[2](x, nv_tabular),
                       self.output_heads[3](x, nv_tabular)), dim=1)
        
        return x


class TimeEncoding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embeddings = PositionalEmbedding(30, 768, None, learned=True)

    def forward(self, inputs, times):
        embeddings = self.position_embeddings(inputs, positions=times.long())
        inputs[:, 1:, :] = inputs[:, 1:, :] + embeddings[:, 1:, :]
        return inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_nheads", default=4, type=int)
    parser.add_argument("--num_transformer_layers", default=3, type=int)
    parser.add_argument("--num_feed_forward_layers", default=2, type=int)
    parser.add_argument("--target_tokenizer", default='bert-base-multilingual-cased')
    parser.add_argument("--root_folder", default='E:\\liamx\\Documents\\recsys2021-val\\')
    parser.add_argument("--layer_dropout", default=0.35, type=float)
    # parser.add_argument("--root_folder", default='gs:\\micro-parade-data')
    parser.add_argument("--data_folder", default='result-6\\')

    args = parser.parse_args()
    model = MicroParade(args).to('cuda:0')
    loader = DataLoader(RecSysMasterDataset(args), batch_size=8, num_workers=2)
    device = torch.device('cuda:0')
    start = time.time()
    i = 0
    with torch.no_grad():
        for input, mask, token_type_ids, history_time, history_mask, nv_tabular, labels in loader:
            input = input.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            history_time = history_time.to(device)
            history_mask = history_mask.to(device)
            nv_tabular = nv_tabular.to(device)

            labels = labels.to(device)
            output = model.forward(input,
                                   token_type_ids,
                                   mask,
                                   history_time,
                                   history_mask,
                                   nv_tabular)

            i += 1
            if i == 10:
                break

    print(time.time() - start)


