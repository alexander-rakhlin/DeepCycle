import cv2
import numpy as np
import threading
from keras import backend as K
from keras.utils import to_categorical
from albumentations import (
    Compose, VerticalFlip, HorizontalFlip, RandomRotate90
)


def preprocess(x, means, stds):
    x = np.moveaxis(x, -1, 0)
    x = np.array([(ch - m[:, None, None]) / s[:, None, None] for ch, m, s in zip(x, means, stds)])
    return np.moveaxis(x, 0, -1)


def null_transform():
    def transform_fun(img):
        return img
    return transform_fun


def train_transform(p=1.0):
    augmentation = Compose([
        VerticalFlip(p=p),
        HorizontalFlip(p=p),
        RandomRotate90(p=p),
    ], p=p)

    def transform_fun(img):
        data = {'image': img}
        augmented = augmentation(**data)
        return augmented['image']

    return transform_fun


class Iterator(object):
    def __init__(self, channel_roots, cell_df, crop_sz,
                 shuffle=True, seed=None, infinite_loop=True, batch_size=32,
                 classes=4, target_column='class_2x2', intensity_cols=None, output_intensities=False,
                 output_df_index=False, verbose=False, gen_id=""):
        self.n_channels = 2
        self.lock = threading.Lock()
        self.channel_roots = channel_roots
        self.frames = sorted(cell_df['FRAME'].unique())
        self.cell_df = cell_df
        self.crop_sz = crop_sz
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose
        self.gen_id = gen_id
        self.total_batches_seen = 0
        self.infinite_loop = infinite_loop
        self.batch_size = batch_size
        self.target_column = target_column
        self.classes = classes
        self.output_intensities = output_intensities
        self.output_df_index = output_df_index
        self.intensity_cols = list(intensity_cols)
        self.index_generator = self._flow_index()

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.frame_index = 0
        self.batch_index = 0
        while 1:
            if self.seed is None:
                random_seed = None
            else:
                random_seed = self.seed + self.total_batches_seen

            # next frame, read images
            if self.batch_index == 0:
                if self.frame_index >= len(self.frames):
                    if not self.infinite_loop:
                        break
                    self.frame_index = 0
                if self.frame_index == 0:
                    if self.verbose:
                        print(f'\n************** New epoch. Generator {self.gen_id} *******************')
                    if self.shuffle:
                        np.random.RandomState(random_seed).shuffle(self.frames)

                frame_num = self.frames[self.frame_index]
                if self.verbose:
                    print(f'************** Frame T{frame_num + 1:0>3}, index {self.frame_index}. '
                          f'Generator {self.gen_id} *******************')

                channels = [cv2.imread(str(root / f'T{frame_num + 1:0>3}'), cv2.CV_16U) for root in self.channel_roots]
                h, w = channels[0].shape
                df = self.cell_df.loc[
                    (self.cell_df['FRAME'] == frame_num) &
                    (self.cell_df['POSITION_Y'] - self.crop_sz // 2 >= 0) &
                    (self.cell_df['POSITION_X'] - self.crop_sz // 2 >= 0) &
                    (self.cell_df['POSITION_Y'] - self.crop_sz // 2 + self.crop_sz < h) &
                    (self.cell_df['POSITION_X'] - self.crop_sz // 2 + self.crop_sz < w)
                    ]
                frame_len = len(df)
                if self.shuffle:
                    df = df.sample(frame_len)

                self.frame_index += 1

            current_index = (self.batch_index * self.batch_size) % frame_len
            if frame_len > current_index + self.batch_size:
                current_batch_size = self.batch_size
                self.batch_index += 1
            else:
                current_batch_size = frame_len - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield df.iloc[current_index: current_index + current_batch_size],\
                  current_index, \
                  current_batch_size,\
                  channels

    def next(self):
        with self.lock:
            df_slice, current_index, current_batch_size, channels = next(self.index_generator)

        batch_x = np.zeros((current_batch_size, self.crop_sz, self.crop_sz, self.n_channels), dtype=K.floatx())
        batch_y = np.zeros((current_batch_size, self.classes), dtype=K.floatx())
        if self.output_intensities:
            batch_intensities = np.zeros((current_batch_size, len(self.intensity_cols)), dtype=K.floatx())
        if self.output_df_index:
            batch_df_index = np.zeros((current_batch_size, 1), dtype=int)

        for i, (df_index, row) in enumerate(df_slice.iterrows()):
            top, left = row[['POSITION_Y', 'POSITION_X']].astype(int).values - self.crop_sz // 2
            batch_x[i] = np.stack([channel[top: top + self.crop_sz, left: left + self.crop_sz]
                                   for channel in channels], axis=-1)
            batch_y[i] = to_categorical(row[self.target_column], num_classes=self.classes)
            if self.output_intensities:
                batch_intensities[i] = row[self.intensity_cols].values
            if self.output_df_index:
                batch_df_index[i] = df_index

        batch_means = [df_slice[root.name + '_average'].values for root in self.channel_roots]
        batch_stds = [df_slice[root.name + '_std'].values for root in self.channel_roots]
        batch_x = preprocess(batch_x, batch_means, batch_stds)

        if K.image_data_format() == 'channels_first':
            batch_x = np.moveaxis(batch_x, -1, 1)

        result = batch_x, batch_y
        if self.output_intensities:
            result += (batch_intensities,)
        if self.output_df_index:
            result += (batch_df_index,)
        return result

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
