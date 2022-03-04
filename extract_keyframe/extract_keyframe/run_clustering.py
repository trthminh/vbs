import pandas as pd
from clustering_kf.keyframe_selection import KeyFrameSelector
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

sys.path.append('/mlcv/WorkingSpace/Personals/khangtd/VBS/source')
# from Utils.utils import mkdir


def run_cluster_kf_one_video(row: pd.Series):
    cluster = KeyFrameSelector(
        merge_threshold=0.3,
        min_frames_in_cluster=1,
        max_frames_in_cluster=100,
        interpolation_bound=10,
        visualize=False)

    mkdir(row['save_dir'])
    try:
        cluster.predict_and_save(
            cluster.read_video_to_frames(row['video_path']),
            cluster.load_result_transnetv2(row['TransNetV2_scene']),
            row['save_dir'])
    except:
        print(f'Error in video {row["video_path"]}')
    finally:
        del cluster


def check_processed(df: pd.DataFrame) -> pd.DataFrame:
    print('[+] Getting the list of unprocessed videos...')
    idxs = [index
            for index, row in df.iterrows()
            if not os.path.exists(os.path.join(row['save_dir'], 'keyframes.txt'))]
    df = df.loc[idxs]
    print(f'[+] There are {len(df)} unprocessed videos')
    return df.reset_index()


def run_cluster_kf(n_process=0):
    df = pd.read_csv('/mlcv/Databases/VBS/Processed_Data/Info/V3C_paths.csv')
    df['save_dir'] = '/mlcv/Databases/VBS/Processed_Data/Keyframes/Predict_txt/' + df['video_id'].astype(str).str.zfill(
        5)
    df['TransNetV2_scene'] += '.scenes.txt'
    df = check_processed(df)
    df.sort_values(by=['length'], inplace=True)
    print(df['length'])
    print('[+] Clustering...')
    if not n_process:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            run_cluster_kf_one_video(row)
    else:
        executor = ProcessPoolExecutor(max_workers=n_process)
        futures = []
        for index, row in df.iterrows():
            futures.append(executor.submit(run_cluster_kf_one_video, row))

        for _ in tqdm(as_completed(futures), total=len(df)):
            pass


if __name__ == '__main__':
    run_cluster_kf(n_process=11)
