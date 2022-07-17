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
import logging as py_logging


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

vocab = f"gs://translationv2/models/spm/cc100_envi_vocab.model"
input_files = [x for x in tf.io.gfile.glob('gs://vien-translation/raw/filtered_len_pubmed/*txt')]
cloud_files = [x for x in tf.io.gfile.glob('gs://vien-translation/data/vi_pubmed_with_vi_tag_512/*txt')]

os.system('mkdir vi')
existed_file = [x.split('-')[0] for x in os.listdir('vi')] + [x.replace('gs://vien-translation/data/vi_pubmed_with_vi_tag_512/', '').split('-')[0] for x in cloud_files]

file_length = 56
offset = args.offset

start = file_length*offset
input_files = input_files[start:start+file_length]

print('='*20, 'Input Files', '='*20)
print(input_files)

for input_file in input_files:
    # Ignore any logging so that we only see the model's answers to the questions.
    output_file = input_file.replace('gs://vien-translation/raw/filtered_len_pubmed/', '')
    import time
    start_time = time.time()
    print('Starting ', output_file)
    predict_inputs_path = input_file
    predict_outputs_path = f"vi/{output_file}"

    
    if output_file in existed_file:
        print('skipping file ', predict_outputs_path)
        continue
    with tf_verbosity_level('ERROR'):
        model.batch_size = 256  # Min size for small model on v2-8 with parallelism 1.
        model.predict(
            input_file=predict_inputs_path,
            output_file=predict_outputs_path,
            # Select the most probable output token at each step.
            vocabulary=t5.data.SentencePieceVocabulary(vocab),
            checkpoint_steps=-1,
            temperature=0,
        )
    print('End ', output_file, )
    print("--- %s seconds ---" % (time.time() - start_time))

