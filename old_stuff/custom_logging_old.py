import torch_xla.core.xla_model as xm
from datetime import datetime
from tensorboardX import SummaryWriter
import time


# borrowed from torch-xla
def _get_device_spec(device):
    ordinal = xm.get_ordinal(defval=-1)
    return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)


def now(format='%H:%M:%S'):
    return datetime.now().strftime(format)


def get_summary_writer(logdir, run_name):
    """Initialize a Tensorboard SummaryWriter.
  Args:
    logdir (str): File location where logs will be written or None. If None, no
      writer is created.
  Returns:
    Instance of Tensorboard SummaryWriter.
  """
    if logdir:
        writer = SummaryWriter(log_dir=logdir, comment=run_name)
        write_to_summary(
            writer, 0, dict_to_write={'TensorboardStartTimestamp': time.time()})
        return writer


def write_to_summary(writer, global_step, dict_to_write):
    if writer is not None:
        for k, v in dict_to_write.items():
            writer.add_scalar(k, v, global_step)

def close_summary_writer(summary_writer):
  """Flush and close a SummaryWriter.
  Args:
    summary_writer (SummaryWriter, optional): The Tensorboard SummaryWriter to
      close and flush. If None, no action is taken.
  """
  if summary_writer is not None:
    summary_writer.flush()
    summary_writer.close()

def _train_update(device, step, loss, tracker, writer, epoch):
    update_data = [
        'Training', 'Device={}'.format(_get_device_spec(device)),
        'Epoch={}'.format(epoch) if epoch is not None else None,
        'Step={}'.format(step), 'Loss={:.5f}'.format(loss),
        'Rate={:.2f}'.format(tracker.rate()), 'GlobalRate={:.2f}'.format(tracker.global_rate()),
        'Time={}'.format(now())
    ]
    print('|', ' '.join(item for item in update_data if item), flush=True)
    write_to_summary(writer, global_step=step, dict_to_write={'train/loss/' + str(epoch) + '/batch': loss,
                                                              'examples/sec': tracker.rate(),
                                                              'average_examples/sec': tracker.global_rate()})
