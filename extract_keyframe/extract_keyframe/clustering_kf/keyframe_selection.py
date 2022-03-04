import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.metrics.pairwise import cosine_similarity

from .clustering import IterativeBestMerge1D, form_clusters
from .keyframe_selection_utils import compute_color_histograms, visualize_scenes_and_keyframes

logger = logging.getLogger(__file__)


class KeyFrameSelector:

    def __init__(self,
                 merge_threshold: float,
                 min_frames_in_cluster: int,
                 max_frames_in_cluster: int,
                 interpolation_bound: int,
                 visualize=False,
                 name="keyframes"):
        self.merge_threshold = merge_threshold
        self.min_frames_in_cluster = min_frames_in_cluster
        self.max_frames_in_cluster = max_frames_in_cluster
        self.interpolation_bound = interpolation_bound
        self.visualize = visualize
        self.__name__ = name

    @staticmethod
    def _visualize_linkage(linkage: np.ndarray, frames: np.ndarray, save_to):
        fig, ax = plt.subplots(1, 1, figsize=(int(len(frames) / 2.), 5))
        ddata = hierarchy.dendrogram(linkage, leaf_font_size=14)

        xl, _, xh, _ = np.array(ax.get_position()).ravel()
        box_w = xh - xl
        img_size = box_w / len(frames)

        for i, frame_no in enumerate(ddata["leaves"]):
            ax = fig.add_axes([xl + img_size * i, -.12, img_size * 0.92, .15])
            ax.axison = False
            ax.imshow(np.rot90(frames[frame_no]))

        fig.savefig(save_to, bbox_inches="tight")
        plt.close()

    def _select_keyframes(self, clusters: np.ndarray):
        cluster_bounds = np.concatenate([[0], np.where(clusters[:-1] != clusters[1:])[0] + 1, [len(clusters)]], 0)
        cluster_bounds = np.stack([cluster_bounds[:-1], cluster_bounds[1:]], -1)
        # selected middle frames of each cluster
        middle_frames = np.floor(np.mean(cluster_bounds, -1)).astype(np.int32)

        # too long clusters
        long_seqs = (cluster_bounds[:, 1] - cluster_bounds[:, 0]) > self.max_frames_in_cluster
        # remove selected middle frames of the too long clusters
        middle_frames = middle_frames[np.logical_not(long_seqs)]
        # cluster bounds of the too long clusters
        long_cluster_bounds = cluster_bounds[long_seqs]

        seq_lengths = long_cluster_bounds[:, 1] - long_cluster_bounds[:, 0]
        first_keyframe_of_cluster = long_cluster_bounds[:, 0] + np.floor(
            (seq_lengths % self.max_frames_in_cluster) / 2
        ).astype(np.int32)

        keyframes = [middle_frames]
        # iterate over all long clusters and
        for cr_start, cr_end in zip(first_keyframe_of_cluster, long_cluster_bounds[:, 1]):
            kfs = np.arange(cr_start, cr_end, self.max_frames_in_cluster)
            keyframes.append(kfs)

        return np.sort(np.concatenate(keyframes))

    def predict_and_save(self, frames: np.ndarray, scenes: np.ndarray, save_dir: str):

        def cluster_func(dist, no_frames_cluster1, no_frames_cluster2):
            no_frames = no_frames_cluster1 + no_frames_cluster2

            if no_frames >= self.interpolation_bound:
                return dist <= self.merge_threshold
            if no_frames < self.min_frames_in_cluster:
                return True

            threshold = self.merge_threshold + (1 - self.merge_threshold) * \
                        (self.interpolation_bound - no_frames) / (self.interpolation_bound - self.min_frames_in_cluster)
            return dist <= threshold


        if self.visualize:
            vis_path = os.path.join(save_dir, self.__name__ + ".visualization")
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)

        clusters_for_visualization, keyframes = [], []
        for scene_no, (start, end) in enumerate(scenes):  # one video
            scene_frames = frames[start:end + 1]

            histograms = compute_color_histograms(scene_frames).reshape([-1, 12 * 6 * 6 * 6])

            dist_matrix = 1 - cosine_similarity(histograms, histograms)
            dist_matrix = np.maximum(0, dist_matrix)  # fix for rounding errors

            linkage = IterativeBestMerge1D(dist_matrix, method="average").linkage
            scene_clusters = form_clusters(linkage, cluster_func)

            if self.visualize:
                self._visualize_linkage(linkage, scene_frames, os.path.join(
                    vis_path, "scene{:04d}.frames{:05d}-{:05d}.png".format(scene_no, start, end)))

            kfs = self._select_keyframes(scene_clusters)
            kfs += start  # keyframes are computed locally, add 'start' to create it global
            keyframes.append(kfs)

            scene_clusters = np.where(scene_clusters[:-1] != scene_clusters[1:])[0] + 1 + start
            clusters_for_visualization.append(scene_clusters)
            logger.debug(f"processed scenes {scene_no + 1} / {len(scenes)}")

        keyframes = np.concatenate(keyframes)

        # clusters_for_visualization = np.concatenate(clusters_for_visualization)
        # img = visualize_scenes_and_keyframes(frames, scenes, keyframes, clusters_for_visualization)
        # img.save(os.path.join(save_dir, self.__name__ + ".visualization.png"))

        np.savetxt(os.path.join(save_dir, self.__name__ + ".txt"), keyframes, "%d")
        return keyframes

    def load_result_from_disk(self, save_dir: str):
        return np.genfromtxt(os.path.join(save_dir, self.__name__ + ".txt"), dtype=np.int32).reshape([-1])

    def load_result_transnetv2(self, scene_path: str):
        return np.genfromtxt(scene_path, dtype=np.int32).reshape([-1, 2])


    def read_video_to_frames(self, video_fn: str):
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError("For `predict_video` function `ffmpeg` needs to be installed in order to extract "
                                      "individual frames from video file. Install `ffmpeg` command line tool and then "
                                      "install python wrapper by `pip install ffmpeg-python`.")

        # print("[+] Loading frames from video {}".format(video_fn))
        video_stream, err = ffmpeg.input(video_fn).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

        return np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

