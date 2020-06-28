from itertools import islice
import pandas as pd
import numpy as np
import cv2
from const import DATA_ROOT, GFP_ROOT, CY3_ROOT


def make_division_adjusted_tracks():
    """Align 1000 curated tracks based on division events"""

    curated_tracks = sorted(pd.read_csv(DATA_ROOT / 'curated_tracks.csv', header=None).astype(int).values.flatten())
    df = pd.read_csv(DATA_ROOT / 'Spots in tracks statistics nq.csv', na_values='None').dropna()
    df = df[df['TRACK_ID'].isin(curated_tracks)]

    div_frames = dict.fromkeys(curated_tracks)
    rows = []
    for frame_num in range(200):
        print('Frame', frame_num + 1)
        row = []
        dt = df.loc[df['FRAME'] == frame_num, ['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'GFP_cmdn', 'Cy3_cmdn']]
        gfp_frame_average = df.loc[df['FRAME'] == frame_num, 'GFP_cmdn'].median()
        cy3_frame_average = df.loc[df['FRAME'] == frame_num, 'Cy3_cmdn'].median()
        row.extend([frame_num, gfp_frame_average, cy3_frame_average])

        for track in curated_tracks:
            dxy = dt[dt['TRACK_ID'] == track]
            if (dxy.shape[0] > 1) and (div_frames[track] is None):  # div_frame is where 2 cells
                div_frames[track] = frame_num
            if dxy.shape[0] < 1:
                time = np.nan  # div_frame
                x, y = np.nan, np.nan
                green_median = np.nan
                red_median = np.nan
                green_mean = np.nan
                red_mean = np.nan
            else:
                time = frame_num
                x, y = dxy[['POSITION_X', 'POSITION_Y']].astype(int).values[0]
                green_median = dxy['GFP_cmdn'].values[0]
                red_median = dxy['Cy3_cmdn'].values[0]
                green_mean = dxy['GFP_cmdn'].values[0]
                red_mean = dxy['Cy3_cmdn'].values[0]
            row.extend([time, x, y, green_median, red_median, green_mean, red_mean])
        rows.append(row)

    div_frames = {k: 0 if v is None else v for k, v in div_frames.items()}
    columns = [('frame_num',), ('gfp_frame_average',), ('cy3_frame_average',)]
    columns_ = [[(track, 'time'), (track, 'x'), (track, 'y')] +
                [(track, color, fun)
                 for fun in ('median', 'mean')
                 for color in ('green', 'red')]
                for track in curated_tracks]
    columns.extend(tt for t in columns_ for tt in t)
    dfo = pd.DataFrame.from_records(rows, columns=pd.MultiIndex.from_tuples(columns))
    for t in curated_tracks:
        dfo[(t, 'time')] -= div_frames[t]
    dfo.to_csv(DATA_ROOT / 'intensities nq.csv', index=False)


def clean_df():
    """Clean and remove unnecessary columns"""

    df = pd.read_csv(DATA_ROOT / 'Spots in tracks statistics nq.csv', na_values="None", header=0,
                     usecols=['ID', 'TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME',
                              'GFP_cmdn', 'Cy3_cmdn', 'DAPI_cmdn', 'BF_cmdn']).dropna()
    df.to_csv(DATA_ROOT / 'statistics_clean nq.csv', index=False)


def add_mean_std(df, verbose=False):
    """Add frame average and standard deviation columns for each channel
    Using curated tracks averages"""

    channels = ['GFP', 'Cy3', 'DAPI', 'BF']

    print(f'Adding averages and standard deviations for {", ".join(channels)} channels')

    curated_tracks = sorted(pd.read_csv(DATA_ROOT / 'curated_tracks.csv', header=None).astype(int).values.flatten())
    df_curated_tracks = df[df['TRACK_ID'].isin(curated_tracks)]

    for channel in channels:
        if verbose:
            print(channel)
        df[channel + '_average'] = 0
        df[channel + '_std'] = 0

        for frame_num in range(200):
            if verbose:
                print('Frame', frame_num + 1)

            img_average = df_curated_tracks.loc[df['FRAME'] == frame_num, channel + '_cmdn'].median()
            img_std = df_curated_tracks.loc[df['FRAME'] == frame_num, channel + '_cmdn'].std()

            df.loc[df['FRAME'] == frame_num, channel + '_average'] = img_average
            df.loc[df['FRAME'] == frame_num, channel + '_std'] = img_std

        df[channel + '_std'] = df[channel + '_std'].mean()

    return df


def add_intensities(df, sz=20, n_frames=None, verbose=False):
    """Add cell intensities for GFP and Cy3 channels"""

    print('Adding cell intensities for GFP and Cy3 channels')
    df['GFP_nq'] = df['GFP_cmdn'] - df['GFP_average']
    df['Cy3_nq'] = df['Cy3_cmdn'] - df['Cy3_average']

    del df['GFP_cmdn']
    del df['Cy3_cmdn']
    del df['DAPI_cmdn']
    del df['BF_cmdn']

    df['POSITION_X'] = df['POSITION_X'].astype(int)
    df['POSITION_Y'] = df['POSITION_Y'].astype(int)

    return df


def add_classes(df, gfp_intensity_col, cy3_intensity_col, class_col_prefix='', n_green=2, n_red=2):
    """Add class column"""

    print('Adding cell classes')

    df['clsCy3'] = pd.qcut(df[cy3_intensity_col], n_red, labels=False)
    df['clsGFP'] = -1

    for cls in range(n_red):
        df.loc[df['clsCy3'] == cls, 'clsGFP'] = \
            pd.qcut(df.loc[df['clsCy3'] == cls, gfp_intensity_col], n_green, labels=False)

    df[class_col_prefix + f'_cls{n_red}x{n_green}'] = (df['clsCy3'] + df['clsGFP'] * n_red).astype(int)
    df.loc[(df['clsCy3'] == -1) | (df['clsGFP'] == -1), class_col_prefix + f'_cls{n_red}x{n_green}'] = np.nan
    del df['clsCy3'], df['clsGFP']
    return df


if __name__ == '__main__':
    # make_division_adjusted_tracks()
    # clean_df()
    df = pd.read_csv(DATA_ROOT / 'statistics_clean nq.csv')
    df = add_mean_std(df, verbose=True)

    df = add_intensities(df, sz=20, n_frames=None, verbose=True)

    df = add_classes(df, gfp_intensity_col='GFP_nq', cy3_intensity_col='Cy3_nq',
                     class_col_prefix='nq', n_green=2, n_red=2)

    df.to_csv(DATA_ROOT / 'statistics_mean_std nq.csv', index=False, float_format='%.3f')
