from collections import OrderedDict

output_classes = OrderedDict(
    consonant_diacritic=7,
    grapheme_root=168,
    vowel_diacritic=11,
)

SEED = 2411

HEIGHT = 137
WIDTH = 236
# SIZE = 224
# SIZE = 128
SIZE = 64

TRAIN = [
  'train_image_data_0.parquet',
  'train_image_data_1.parquet',
  'train_image_data_2.parquet',
  'train_image_data_3.parquet',
]

TEST = [
  'test_image_data_0.parquet',
  'test_image_data_1.parquet',
  'test_image_data_2.parquet',
  'test_image_data_3.parquet',
]

ZIP_TRAIN_FILE = f'train{SIZE}.zip'
ZIP_TEST_FILE = f'test{SIZE}.zip'

VALID_SIZE = 0.1

TRAIN_MEAN = 0.06922848809290576
TRAIN_STD = 0.20515700083327537

num_workers = 4
logdir = './logs'

