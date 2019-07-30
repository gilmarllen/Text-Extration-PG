from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_SHUFFLE_TRAIN_SMALL_DATA = [
    [
        "",
        ("shuffle_in_train.txt",
         "shuffle_out_train.txt")
    ],
]
_SHUFFLE_TEST_SMALL_DATA = [
    [
        "",
        ("shuffle_in_valid.txt",
         "shuffle_out_valid.txt")
    ],
]
_SHUFFLE_TRAIN_LARGE_DATA = []
_SHUFFLE_TEST_LARGE_DATA = []


@registry.register_problem
class ShuffleProblem(translate.TranslateProblem):

  @property
  def approx_vocab_size(self):
    return 2**15

  @property
  def use_small_dataset(self):
    return True

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    if self.use_small_dataset:
      datasets = _SHUFFLE_TRAIN_SMALL_DATA if train else _SHUFFLE_TEST_SMALL_DATA
    else:
      datasets = _SHUFFLE_TRAIN_LARGE_DATA if train else _SHUFFLE_TEST_LARGE_DATA
    return datasets

  def vocab_data_files(self):
    return (_SHUFFLE_TRAIN_SMALL_DATA if self.use_small_dataset
            else _SHUFFLE_TRAIN_LARGE_DATA)

