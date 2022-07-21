from click import style
import tensorflow
import functools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import tensorflow.compat.v1 as tf
import gin
import t5
import os
from t5.models import MtfModel

print(tensorflow.__version__)


parser = argparse.ArgumentParser(description='Finetunning ViT5')
parser.add_argument('-tpu', dest='tpu', type=str, help='tpu address', default='0.0.0.0')
parser.add_argument('-model', dest='model', type=str, help='model dir', default='gs://vien-translation/checkpoints/enviT5_finetune/mtet_envi_1000000enviT5_total_lr0.001_pubmed_512')
parser.add_argument('-offset', dest='offset', type=int, help='Offset', default=0)

args = parser.parse_args()

TPU_TOPOLOGY = 'v2-8'
TPU_ADDRESS = args.tpu
TPU_ADDRESS = f'grpc://{TPU_ADDRESS}:8470'

print(f"TPU Address {TPU_ADDRESS}")
ON_CLOUD = True


tf.disable_v2_behavior()

# Improve logging.
from contextlib import contextmanager


@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

MODEL_SIZE = "base"

# Set parallelism and batch size to fit on v3-8 TPU (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (4, 256, 8),
    "large": (8, 256, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

MODEL_DIR = args.model
model = MtfModel(
    model_dir=MODEL_DIR,
    tpu=TPU_ADDRESS,
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": 512, "targets": 512},
    learning_rate_schedule=0.001,
    save_checkpoints_steps=0,
    keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
    # iterations_per_loop=100,
)

import os.path

BASE_DIR = '../data/mednli/sentences'
raw_file = f'{BASE_DIR}/sentences.txt'
input_file = 'tmp_input.txt'
output_file = 'tmp_output.txt'

with open(raw_file, 'r', encoding='utf-8') as file:
    with open(input_file, 'w', encoding='utf-8') as out_file:
        for line in file:
            out_file.write(f'en: {line}')

vocab = f"gs://translationv2/models/spm/cc100_envi_vocab.model"

print('Start to translate MedNLI')
import time
start_time = time.time()

with tf_verbosity_level('ERROR'):
    model.batch_size = 256  # Min size for small model on v2-8 with parallelism 1.
    model.predict(
        input_file=input_file,
        output_file=output_file,
        # Select the most probable output token at each step.
        vocabulary=t5.data.SentencePieceVocabulary(vocab),
        checkpoint_steps=-1,
        temperature=0,
    )
print('End ', output_file, )
print("--- %s seconds ---" % (time.time() - start_time))

with open([s for s in os.listdir('.') if output_file in s][0], 'r', encoding='utf-8') as file:
    with open(f'{BASE_DIR}/vi_sentences.txt', 'w', encoding='utf-8') as out_file:
        for line in file:
            line = line.replace('vi: ', '')
            out_file.write(line)

os.system(f'rm {input_file}')
os.system(f'rm {output_file}')

