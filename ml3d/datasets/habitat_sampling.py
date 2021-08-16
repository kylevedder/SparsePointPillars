import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
from pyntcloud import PyntCloud
import pandas as pd
import open3d as o3d
import logging
import yaml
import joblib
import time

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import Config, make_dir, DATASET
from .utils import DataProcessing, BEVBox3D

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class HabitatSampling(BaseDataset):
    """This class is used to create a dataset based on the KITTI dataset, and
    used in object detection, visualizer, training, or testing.
    """

    def __init__(self,
                 dataset_path,
                 name='HabitatSampling',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 val_split=3750,
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (HabitatSampling in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            val_split: The split value to get a set of images for training,
            validation, for testing.
            test_result_folder: Path to store test output.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         val_split=val_split,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 1
        self.label_to_names = self.get_label_to_names()

        self.all_files = glob(
            join(cfg.dataset_path, 'training', 'pc', '*.bin'))
        self.all_files.sort()
        self.train_files = []
        self.val_files = []

        for f in self.all_files:
            idx = int(Path(f).name.replace('.bin', ''))
            if idx < cfg.val_split:
                self.train_files.append(f)
            else:
                self.val_files.append(f)

        self.test_files = glob(
            join(cfg.dataset_path, 'testing', 'pc', '*.bin'))
        self.test_files.sort()

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.
        """
        label_to_names = {
            0: 'chair',
            1: 'DontCare'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()        
        return joblib.load(path)

    @staticmethod
    def read_label(path):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        assert Path(path).exists()
        boxes = joblib.load(path)
        objects = []
        for b in boxes:
            name, img_left, img_top, img_right, img_bottom, center_x, center_y, center_z, l, w, h, yaw = b
            yaw = -np.deg2rad(np.float32(yaw))
            # image_bb = np.array([img_left, img_top, img_right, img_bottom])
            center = np.array([center_x, center_y, center_z], np.float32)

            size = np.array([l, h, w], np.float32)  # Weird order is what the BEV box takes
            center = np.array([center_x, center_y, center_z], np.float32) # Actual center of the box
            objects.append(BEVBox3D(center, size, yaw, name, 1))
        return objects

    @staticmethod
    def _extend_matrix(mat):
        mat = np.concatenate(
            [mat, np.array([[0., 0., 1., 0.]], dtype=mat.dtype)], axis=0)
        return mat

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return HabitatSamplingSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['train', 'training']:
            return self.train_files
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then resturn the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attrs):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            attrs: The attributes that correspond to the outputs passed in
            results.
        """
        make_dir(self.cfg.test_result_folder)
        for attr, res in zip(attrs, results):
            name = attr['name']
            path = join(self.cfg.test_result_folder, name + '.txt')
            f = open(path, 'w')
            for box in res:
                f.write(box.to_kitti_format(box.confidence))
                f.write('\n')


def save_pc(pc, filename):
    PyntCloud(pd.DataFrame(data=pc[:,:3],
            columns=["x", "y", "z"])).to_file(filename)

def save_boxes(boxes, filename, log=False):
    mesh = o3d.geometry.TriangleMesh()
    for bb in boxes:
        if (np.array(bb.size) < 0).any():
            continue
        if log:
            print(bb.size)
            print(bb.yaw)
        box = o3d.geometry.TriangleMesh.create_box(bb.size[2], bb.size[0], bb.size[1])
        box.rotate(mesh.get_rotation_matrix_from_xyz((0, 0, -bb.yaw - np.pi / 2)), box.get_center())
        box.translate(bb.center, relative=False)
        mesh += box
    o3d.io.write_triangle_mesh(filename, mesh)

class HabitatSamplingSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        label_path = pc_path.replace('pc',
                                     'boxes').replace('.bin', '.txt')

        pc = self.dataset.read_lidar(pc_path)
        label = self.dataset.read_label(label_path)

        # save_pc(pc, f"pc_before{idx:06d}.ply")
        # save_boxes(label, f"boxes{idx:06d}.ply")

        data = {
            'point': pc,
            'full_point': pc,
            'feat': None,
            'calib': {},
            'bounding_boxes': label,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr


DATASET._register_module(HabitatSampling)
