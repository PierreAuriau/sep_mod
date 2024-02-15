# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import os
import pickle
import logging
import bisect
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from typing import Callable, List, Type, Sequence, Dict
from torch.utils.data.dataset import Dataset

logger = logging.getLogger()


class ClinicalBase(ABC, Dataset):
    """
        A generic clinical Dataset written in a torchvision-like manner. It parses a .pkl file defining the different
        splits based on a <unique_key>. All clinical datasets must have:
        - a training set
        - a validation set
        - a test set
        - (eventually) an other intra-test set

        This generic dataset is memory-efficient, taking advantage of memory-mapping implemented with NumPy.
        It always come with:
        ... 3 pre-processings:
            - Quasi-Raw
            - VBM
            - Skeleton
        ... And 4 differents tasks:
            - Diagnosis prediction (classification)
            - Site prediction (classification)
            - Age prediction (regression)
            - Sex prediction (classification)
        ... With meta-data:
            - user-defined unique identifier across pre-processing and split
            - TIV + ROI measures based on Neuromorphometrics atlas
    Attributes:
          * target, list[int]: labels to predict
          * all_labels, pd.DataFrame: all labels stored in a pandas DataFrame containing ["diagnosis", "site", "age", "sex]
          * shape, tuple: shape of the data
          * metadata: pd DataFrame: Age + Sex + TIV + ROI measures extracted for each image
          * id: pandas DataFrame, each row contains a unique identifier for an image

    """
    def __init__(self, root: str, preproc: [str, List[str]] = 'vbm', target: [str, List[str]] = 'diagnosis',
                 split: str = 'train', transforms: Callable[[np.ndarray], np.ndarray] = None,
                 load_data: bool = False, two_views: bool = False):
        """
        :param root: str, path to the root directory containing the different .npy and .csv files
        :param preproc: str, must be either VBM ('vbm'), Quasi-Raw ('quasi_raw') or Skeleton ('skeleton')
        :param target: str or [str], either 'dx' or 'site'.
        :param split: str, either 'train', 'val', 'test' (inter) or (eventually) 'test_intra'
        :param transforms (callable, optional): A function/transform that takes in
            a 3D MRI image and returns a transformed version.
        :param load_data (bool, optional): If True, loads all the data in memory
               --> WARNING: it can be time/memory-consuming
        """
        # 0) Check parameters and set attributes
        # preproc
        if isinstance(preproc, str):
            assert preproc in {'vbm', 'quasi_raw', 'skeleton'}, f"Unknown preproc: {preproc}"
            self.preproc = [preproc]
        elif isinstance(preproc, list):
            assert set(preproc) <= {'vbm', 'quasi_raw', 'skeleton'}, f"Unknown preproc: {preproc}"
            self.preproc = preproc
        # root
        self.root = root
        if not self._check_integrity():
            raise FileNotFoundError(f"Files not found. Check the the root directory {root}")
        # split
        assert split in ['train', 'val', 'test', 'test_intra', 'validation'], f"Unknown split: {split}"
        if split == "val":
            self.split = "validation"
        else:
            self.split = split
        self.scheme = self.load_pickle(os.path.join(
            self.root, self._train_val_test_scheme))[self.split]
        # target
        if isinstance(target, str):
            assert target in {"age", "sex", "diagnosis", "site"}, f"Unknown target: {target}"
            self.target_name = [target]
        elif isinstance(target, list):
            assert set(target) <= {"age", "sex", "diagnosis", "site"}, f"Unknown target: {target}"
            self.target_name = target
        self.transforms = transforms
        # transforms
        self.transforms = dict()
        for pr in self.preproc:
            if transforms is None:
                self.transforms[pr] = None
            elif pr not in transforms.keys():
                self.transforms[pr] = None
            else:
                self.transforms[pr] = transforms[pr]
        # two views
        self.two_views = two_views
                
        # 1) Loads globally all the data for a given pre-processing
        df = pd.concat([pd.read_csv(os.path.join(root, self._get_pd_files % db), dtype=self._id_types) 
                        for db in self._studies],
                       ignore_index=True, sort=False)
        data = {pr: [np.load(os.path.join(root, self._get_npy_files[pr] % db), mmap_mode='r')
                     for db in self._studies]
                for pr in self.preproc}
                 
        cumulative_sizes = {pr: np.cumsum([len(db) for db in data[pr]]) for pr in self.preproc}
        # FIXME: check if all cumulative sizes are equals
        assert all([np.array_equal(cumulative_sizes[self.preproc[0]], arr) for arr in cumulative_sizes.values()]), \
        f"All npy files of the different preprocessings do not have same number of subjects."

        # 2) Selects the data to load in memory according to selected scheme
        mask = self._extract_mask(df, unique_keys=self._unique_keys, check_uniqueness=self._check_uniqueness)

        # Get TIV and tissue volumes according to the Neuromorphometrics atlas
        self.metadata = self._extract_metadata(df[mask]).reset_index(drop=True)
        self.id = df[mask][self._unique_keys].reset_index(drop=True)

        # Get the labels to predict
        assert set(self.target_name) <= set(df.keys()), \
            f"Inconsistent files: missing {self.target_name} in participants DataFrame"
        self.target = df[mask][self.target_name]
        assert self.target.isna().sum().sum() == 0, f"Missing values for {self.target_name} label"
        self.target = self.target.apply(self.target_transform_fn, axis=1, raw=True).values.astype(np.float32)
        all_keys = ["age", "sex", "diagnosis", "site"]
        self.all_labels = df[mask][all_keys].reset_index(drop=True)
        # Transforms (dx, site) according to _target_mappings
        self.all_labels = self.all_labels.apply(lambda row: [row[0], 
                                                             self._target_mappings["sex"][row[1]],
                                                             self._target_mappings["diagnosis"][row[2]],
                                                             self._target_mappings["site"][row[3]]],
                                                axis=1, raw=True, result_type="broadcast")
        # Prepares private variables to build mapping target_idx -> img_idx
        self.shape = {pr: (mask.sum(), *data[pr][0][0].shape) for pr in self.preproc}
        self._mask_indices = np.arange(len(df))[mask]
        self._cumulative_sizes = cumulative_sizes[self.preproc[0]]
        self._data = data
        self._data_loaded = None

        # Loads all in memory to retrieve it rapidly when needed
        if load_data:
            self._data_loaded = self.get_data()[0]

    @property
    @abstractmethod
    def _studies(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def _train_val_test_scheme(self) -> str:
        ...

    @property
    @abstractmethod
    def _unique_keys(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def _target_mappings(self) -> Dict[str, Dict[str, int]]:
        ...

    @property
    def _check_uniqueness(self) -> bool:
        return True

    @property
    def _get_npy_files(self) -> Dict[str, str]:
        return {"vbm": "%s_t1mri_mwp1_gs-raw_data64.npy",
                "quasi_raw": "%s_t1mri_quasi_raw_data32_1.5mm_skimage.npy",
                "skeleton": "%s_t1mri_skeleton_data32.npy"}

    @property
    def _get_pd_files(self) -> Dict[str, str]:
        return "%s_t1mri_participants.csv"
    
    @property
    def _id_types(self):
        return {"participant_id": str,
                "session": int,
                "acq": int,
                "run": int}

    def _check_integrity(self):
        """
        Check the integrity of root dir (including the directories/files required). It does NOT check their content.
        Should be formatted as:
        /root
            <train_val_test_split.pkl>
            [cohort]_t1mri_mwp1_gs-raw_data64.npy
            [cohort]_t1mri_skeleton_data32.npy
            [cohort]_t1mri_participants.csv
        """
        is_complete = os.path.isdir(self.root)
        is_complete &= os.path.isfile(os.path.join(self.root, self._train_val_test_scheme))

        for pr in self.preproc:
            files = [self._get_pd_files, self._get_npy_files[pr]]
            for file in files:
                for db in self._studies:
                    is_complete &= os.path.isfile(os.path.join(self.root, file % db))
        return is_complete

    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool = True):
        """
        :param df: pandas DataFrame
        :param unique_keys: list of str
        :param check_uniqueness: if True, check the unique_keys identified uniquely an image in the dataset
        :return: a binary mask indicating, for each row, if the participant belongs to the current scheme or not.
        """
        _source_keys = df[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        if check_uniqueness:
            assert len(set(_source_keys)) == len(_source_keys), f"Multiple identique identifiers found"
        _target_keys = self.scheme[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        mask = _source_keys.isin(_target_keys).values.astype(bool)
        return mask

    def _extract_metadata(self, df: pd.DataFrame):
        """
        :param df: pandas DataFrame
        :return: TIV and tissue volumes defined by the Neuromorphometrics atlas
        """
        metadata = ["age", "sex", "tiv"] + [k for k in df.keys() if "GM_Vol" in k or "WM_Vol" in k or "CSF_Vol" in k]
        if len(metadata) != 290:
            logger.warning(f"Missing meta-data values ({len(metadata)} != 290)")
        assert set(metadata) <= set(df.keys()), "Missing meta-data columns: {}".format(set(metadata) - set(df.keys))
        if df[metadata].isna().sum().sum() > 0:
            logger.warning("NaN values found in meta-data")
        return df[metadata]
    
    def target_transform_fn(self, target):
        """ Transforms the target according to mapping site <-> int and dx <-> int """
        target = target.copy()
        for i, name in enumerate(self.target_name):
            if name in self._target_mappings.keys():
                target[i] = self._target_mappings[name][target[i]]
        return target

    def load_pickle(self, path: str):
        with open(path, 'rb') as f:
            pkl = pickle.load(f)
        return pkl

    def get_data(self, indices: Sequence[int] = None, dtype: Type = np.float32):
        """
        Loads all (or selected ones) data in memory and returns a big numpy array X_data with y_data
        The input/target transforms are ignored.
        Warning: this can be memory-consuming (~10GB if all data are loaded)
        :param indices : (Optional) list of indices to load
        :param mask : (Optional) binary mask to apply to the data. Each 3D volume is transformed into a
        vector. Can be 3D mask or 4D (channel + img)
        :param dtype : (Optional) the final type of data returned (e.g np.float32)
        :return (np.ndarray, np.ndarray), a tuple (X, y)
        """
        tf = self.transforms
        self.transforms = None
        data = dict()
        for pr in self.preproc:
            if indices is None:
                ngb = np.product(self.shape[pr])*np.dtype(dtype).itemsize(1024*1024*1024)
                logger.infof(f"Dataset size to load (shape {self.shape[pr]}): {ngb:.2f} GB")
                if self._data_loaded is not None:
                    data[pr] = self._data_loaded[pr]
                else:
                    data[pr] = np.zeros(self.shape[pr], dtype=dtype)
                    for i in range(len(self)):
                        data[pr][i] = self[pr][i][0]
                self.transforms = tf
                return data, np.copy(self.target)
            else:
                ngb = np.product(self.shape[pr][1:])*len(indices)*np.dtype(dtype).itemsize(1024*1024*1024)
                logger.info(f"Dataset size to load (shape {(len(indices),) + self.shape[pr][1:]}): {ngb:.2f} GB")
                if self._data_loaded is not None:
                    data[pr] = self._data_loaded[pr][indices].astype(dtype)
                else:
                    data[pr] = np.zeros((len(indices), *self.shape[pr][1:]), dtype=dtype)
                    for i, idx in enumerate(indices):
                        data[pr][i] = self[pr][idx][0]
                    
                self.transforms = tf
                return data, self.target[indices]

    def _mapping_idx(self, idx: int):
        """
        :param idx: int ranging from 0 to len(dataset)-1
        :return: integer that corresponds to the original image index to load
        """
        idx = self._mask_indices[idx]
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        sample_idx = idx - self._cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else idx
        return dataset_idx, sample_idx
    
    def __getitem__(self, idx: int):
        sample = dict()
        if self._data_loaded is not None:
            for pr in self.preproc:
                sample[pr] = self._data_loaded[pr][idx]
            for t, tgt in enumerate(self.target_name):
                sample[tgt] = self.target[idx][t]
        else:
            (dataset_idx, sample_idx) = self._mapping_idx(idx)
            for pr in self.preproc:
                sample[pr] = self._data[pr][dataset_idx][sample_idx]
            for t, tgt in enumerate(self.target_name):
                sample[tgt] = self.target[idx][t]
        for pr in self.preproc:
            if self.two_views:
                view_1 = self.transforms[pr](sample[pr])
                view_2 = self.transforms[pr](sample[pr])
                sample[pr] = (view_1, view_2)
            else:
                if self.transforms[pr] is not None:
                    sample[pr] = self.transforms[pr](sample[pr])
        return sample
    
    def copy(self):
        this = self.__class__(*self.args, N_train_max=self.N_train_max, stratify=self.stratify,
                              nb_folds=self.nb_folds, **self.kwargs)
        return this
    def __len__(self):
        return len(self.target)

    def __str__(self):
        return f"{type(self).__name__}-{tuple(self.preproc)}-{self.split}"


class SCZDataset(ClinicalBase):

    @property
    def _studies(self):
        return ["schizconnect-vip-prague", "bsnip1", "cnp", "candi"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_scz_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study"]

    @property
    def _target_mappings(self):
        return dict(diagnosis={"control": 0, "schizophrenia": 1, "scz": 1},
                    sex={"M": 0, "F": 1},
                    site=self._site_mapping)
    
    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_scz.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_scz.pkl"))
        super().__init__(root, *args, **kwargs)


class BDDataset(ClinicalBase):

    @property
    def _studies(self):
        return ["biobd", "bsnip1", "cnp", "candi"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_bip_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study"]

    @property
    def _target_mappings(self):
        return dict(diagnosis={"control": 0, "bipolar": 1, "bipolar disorder": 1, "psychotic bipolar disorder": 1, "bd": 1},
                    sex={"M": 0, "F": 1},
                    site=self._site_mapping)

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_bip.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_bip.pkl"))
        super().__init__(root, *args, **kwargs)


class ASDDataset(ClinicalBase):

    @property
    def _studies(self):
        return ["abide1", "abide2"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_asd_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study", "run"]

    @property
    def _target_mappings(self):
        return dict(diagnosis={"control": 0, "autism": 1, "asd": 1},
                    sex={"M": 0, "F": 1},
                    site=self._site_mapping)

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_asd.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_asd.pkl"))
        super().__init__(root, *args, **kwargs)

if __name__ == "__main__":
    dataset = SCZDataset(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", preproc=["vbm", "skeleton"], split="test")
    for sample in dataset:
        print(sample)
        break