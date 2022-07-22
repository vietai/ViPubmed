from unittest.util import _MAX_LENGTH
import tensorflow
import functools
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import gin
from t5 import models
import t5
import gin
import argparse
from random import shuffle

print(tensorflow.__version__)

parser = argparse.ArgumentParser(description='Pretraining ViPubmedT5')
parser.add_argument('-tpu', dest='tpu', type=str, help='tpu address', default='0.0.0.0')
parser.add_argument('-length', dest='length', type=int, help='sequence length', default=512)
args = parser.parse_args()


TPU_TOPOLOGY = 'v3-8'
TPU_ADDRESS = args.tpu
TPU_ADDRESS = f'grpc://{TPU_ADDRESS}:8470'
MAX_LENGTH = args.length

ON_CLOUD = True
tf.config.experimental_connect_to_host(TPU_ADDRESS)

# Improve logging.
from contextlib import contextmanager

@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

gin.parse_config_file(
        '../configs/t5/base_operative_config.gin'
    )


def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    files_name = [x for x in tf.io.gfile.glob('gs://translationv2/data/vi_pubmed_512/*txt')]

    shuffle(files_name)

    print(files_name[0])

    ds = tf.data.TextLineDataset(
       files_name
    )
    ds = ds.map(lambda *ex: dict(zip(['title', 'text'], ['None',ex[0]])))
    ds = ds.shuffle(buffer_size=1000000)

    return ds

MODEL_SIZE = 'base'

vocab = f"gs://vietai_public/viT5/viT5_{MODEL_SIZE}_1024/spiece.model"
t5.data.TaskRegistry.remove('dumping_dataset')
t5.data.TaskRegistry.add(
    'dumping_dataset',
    dataset_fn = dumping_dataset,
    splits = ['train'],
    text_preprocessor =  functools.partial(
        t5.data.preprocessors.rekey,
        key_map = {'inputs': None, 'targets': 'text'},
    ),
    token_preprocessor = t5.data.preprocessors.unsupervised,
    output_features=t5.data.Feature(vocabulary=t5.data.SentencePieceVocabulary(vocab)),
    metric_fns = [],
)



t5.data.MixtureRegistry.remove('all')
t5.data.MixtureRegistry.add(
    'all',
    [
        'dumping_dataset',
    ],
    default_rate = 1.0,
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


model_parallelism, train_batch_size, keep_checkpoint_max = {
    'small': (1, 256, 16),
    'base': (4, 256, 8),
    'large': (8, 256, 4),
    '3B': (8, 16, 1),
    '11B': (8, 16, 1),
}[MODEL_SIZE]


model_dir = f'gs://translationv2/models/ViPubmedT5_{MAX_LENGTH}_{MODEL_SIZE}'
pretrained_model_dir = f'gs://vietai_public/viT5/viT5_{MODEL_SIZE}_1024'

model = models.MtfModel(
  model_dir = model_dir,
  tpu = TPU_ADDRESS,
  tpu_topology = TPU_TOPOLOGY,
  model_parallelism = model_parallelism,
  batch_size = train_batch_size,
  sequence_length = {'inputs': MAX_LENGTH, 'targets': MAX_LENGTH},
  learning_rate_schedule = 0.001,
  save_checkpoints_steps = 10000,
  keep_checkpoint_max = 5,
  iterations_per_loop = 100,
)

model.train(mixture_or_task_name = 'all', steps = 1500000, pretrained_model_dir=pretrained_model_dir)