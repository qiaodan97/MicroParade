import sys
import torch
from torch.utils.data import DataLoader
from recsys_dataset import RecSysMasterDataset, worker_init_fn, generate_split
from tqdm import tqdm, trange
import argparse
from datetime import datetime
from micro_parade_old import MicroParade
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from fairseq.modules import PositionalEmbedding
import torch_xla.debug.metrics as met
import numpy as np
from custom_logging_old import get_summary_writer, write_to_summary, _train_update, now, close_summary_writer
import torch_xla.utils.serialization as xser
import gc
from sklearn.metrics import average_precision_score, log_loss

SERIAL_EXEC = xmp.MpSerialExecutor()


def train(args, train_dataset, test_dataset):
    torch.manual_seed(1)
    device = xm.xla_device()

    def prepare_datasets():
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.valid_batch_size,
            shuffle=False,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            num_workers=args.num_workers)

        model = MicroParade(args).to(device)
        if args.load_checkpoint:
            xm.master_print('LOADING CHECKPOINT')
            model.load_state_dict(torch.load(args.checkpoint))

        return train_loader, test_loader, model

    train_loader, test_loader, model = SERIAL_EXEC.run(prepare_datasets)
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    test_device_loader = pl.MpDeviceLoader(test_loader, device)
    lr = args.learning_rate * xm.xrt_world_size()

    writer = None
    if xm.is_master_ordinal():
        writer = get_summary_writer(args.logdir, args.comment)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=args.adam_epsilon)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([35.436611946475374,
                                                                  9.04134807583367,
                                                                  82.02527887007463,
                                                                  1.2184770791472297])).to(device)
                                                                  
    # loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    # loss_fn = torch.nn.BCELoss(weight=torch.Tensor([35.436611946475374,
    #                                                 9.04134807583367,
    #                                                 82.02527887007463,
    #                                                 1.2184770791472297]))
    def train_loop_fn(loader, epoch):
        tracker = xm.RateTracker()
        model.train()

        for step, (input, mask, token_type_ids, history_time, history_mask, nv_tabular, labels) in enumerate(loader):
            input = input.to(device)
            # xm.master_print(f'INPUT {input}')
            mask = mask.to(device)
            # xm.master_print(f'MASK {mask}')
            token_type_ids = token_type_ids.to(device)
            # xm.master_print(f'IDS {token_type_ids}')
            history_time = history_time.to(device)
            # xm.master_print(f'TIME {history_time}')
            history_mask = history_mask.to(device)
            # xm.master_print(f'HISTORY {history_mask}')
            nv_tabular = nv_tabular.to(device)
            # xm.master_print(f'NV {nv_tabular}')
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model.forward(input,
                                   token_type_ids,
                                   mask,
                                   history_time,
                                   history_mask,
                                   nv_tabular)
         
            loss = loss_fn(output, labels)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(args.train_batch_size)
            if step % args.log_steps == 0:
                xm.add_step_closure(
                    _train_update, args=(device, step, loss, tracker, writer, epoch)
                )

        return loss.item()

    def test_loop_fn(loader, epoch):

        model.eval()
        tracker = xm.RateTracker()

        with torch.no_grad():
            pred_total = None
            labels_total = None
            for step, (input, mask, token_type_ids, history_time, history_mask, nv_tabular, labels) in enumerate(
                    loader):
                input = input.to(device)
                mask = mask.to(device)
                token_type_ids = token_type_ids.to(device)
                history_time = history_time.to(device)
                history_mask = history_mask.to(device)
                labels = labels.to(device)
                nv_tabular = nv_tabular.to(device)

                pred = model.forward(input,
                                     token_type_ids,
                                     mask,
                                     history_time,
                                     history_mask,
                                     nv_tabular)
                pred = pred.sigmoid()
                tracker.add(args.valid_batch_size)
                if pred_total is None:
                    pred_total = pred
                    labels_total = labels
                else:
                    pred_total = torch.cat((pred, pred_total))
                    labels_total = torch.cat((labels, labels_total))
            pred_total = xm.all_gather(pred_total).cpu()
            labels_total = xm.all_gather(labels_total).cpu()
            ap = average_precision_score(labels_total, pred_total, average=None)
            avg_ap = ap.sum() / 4

        # accuracy = 100.0 * correct / total_samples
        # mean = accuracy.mean()
        # avg_accuracy = xm.mesh_reduce('test_accuracy', mean, np.mean)
        # avg_accuracy_0 = xm.mesh_reduce('test_accuracy', accuracy[0], np.mean)
        # avg_accuracy_1 = xm.mesh_reduce('test_accuracy', accuracy[1], np.mean)
        # avg_accuracy_2 = xm.mesh_reduce('test_accuracy', accuracy[2], np.mean)
        # avg_accuracy_3 = xm.mesh_reduce('test_accuracy', accuracy[3], np.mean)
        return avg_ap, ap

        # test loop

    avg_ap, max_ap = 0.0, 0.0
    start = 1
    if args.load_checkpoint:
        start = args.start_epoch
    for epoch in range(start, args.epoch + 1):
        xm.master_print('Epoch {} train begin {}'.format(epoch, now()))
        gc.collect()

        loss = train_loop_fn(train_device_loader, epoch)

        xm.master_print('Epoch {} train end {}'.format(epoch, now()))

        avg_ap, ap = test_loop_fn(test_device_loader, epoch)

        xm.master_print('Epoch {} test end {}, avg_ap={:.2f}'.format(epoch, now(), avg_ap))
        max_ap = max(avg_ap, max_ap)
        if xm.is_master_ordinal():
            write_to_summary(
                writer,
                epoch,
                dict_to_write={'Loss/train/epoch': loss, 'Average Precision/test': avg_ap}
            )
            write_to_summary(
                writer,
                epoch,
                dict_to_write={'Average Precision/test/' + str(i): val.item() for i, val in enumerate(ap)},
            )

        xm.save(model.state_dict(), args.output_folder + args.comment + "-" + 'model.pt', master_only=True,
                global_master=True)

        if args.metrics_debug:
            xm.master_print(met.metrics_report())

    close_summary_writer(writer)
    xm.master_print('Max Accuracy: {:.2f}%'.format(max_ap))
    xm.save(model.state_dict(), args.output_folder + args.comment + "-" + 'model.pt', master_only=True,
            global_master=True)
    xm.rendezvous('init')
    return max_ap, model


def _mp_fn(index, args):
    torch.set_default_tensor_type('torch.FloatTensor')
    train_dataset, test_dataset = generate_split(args, 0.9)

    train_dataset.add_tpu_data(xm.xrt_world_size(), xm.get_ordinal())
    test_dataset.add_tpu_data(xm.xrt_world_size(), xm.get_ordinal())
    # print(args)
    accuracy, model = train(args, train_dataset, test_dataset)
    print(accuracy)
    print("done!")
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # this will need to be duplicated for train and test and val
    parser.add_argument("--dataset_file", default='final_recsys_dataset.pt', type=str)
    parser.add_argument("--checkpoint_folder", default='.\\checkpoints\\', type=str)
    parser.add_argument("--output_folder", default='.\\final_models\\')
    parser.add_argument("--target_tokenizer", default='bert-base-multilingual-cased')
    parser.add_argument("--root_folder", default='gs://micro-parade-data/recsys2021/')
    parser.add_argument("--data_folder", default='val/')
    parser.add_argument("--metrics_debug", default=False, type=bool)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--valid_batch_size", default=32, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--output_labels", default=2, type=int)
    parser.add_argument("--checkpoint_epoch", default=1, type=int)
    parser.add_argument("--transformer_nheads", default=4, type=int)
    parser.add_argument("--num_transformer_layers", default=3, type=int)
    parser.add_argument("--num_feed_forward_layers", default=2, type=int)
    parser.add_argument("--train_dropout", default=0.1, type=float)
    parser.add_argument("--layer_dropout", default=0.35, type=float)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--comment", default="default", type=str)
    parser.add_argument("--num_cores", default=1, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logdir", default="runs", type=str)
    parser.add_argument("--log_steps", default=10, type=int)
    parser.add_argument("--load_checkpoint", default=False, type=bool)
    parser.add_argument("--checkpoint", default='final_model_old/first run, recsys2020 dev only, multi-head-model.pt')
    parser.add_argument("--start_epoch", default=1, type=int)
    args = parser.parse_args()
    print('loading dataset')

    print('loaded_dataset')

    print('spawning....')
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores)
