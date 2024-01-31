# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
import numpy as np
import pandas as pd
import logging  
from collections import namedtuple
from typing import List
import torch
from torchvision.transforms.transforms import Compose
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# project imports
from datasets import SCZDataset, BDDataset, ASDDataset
from data_augmentation import Normalize
from da_module import DAModule

logger = logging.getLogger()
SetItem = namedtuple("SetItem", ["test", "train", "validation"], defaults=(None,) * 3)
DataItem = namedtuple("DataItem", ["inputs", "labels"])
TwoViewItem = namedtuple("TwoViewItem", ["view_1", "view_2"])
TwoModalityItem = namedtuple("TwoModalityItem", ["weak", "strong", "labels"])

class ClinicalDataManager(object):

    def __init__(self, root: str, db: str, preproc: str, labels: List[str] = None, 
                 two_views: bool = False, batch_size: int = 1, **dataloader_kwargs):

        assert db in ["scz", "bd", "asd"], f"Unknown db: {db}"

        self.db = db
        self.labels = labels or []
        self.two_views = two_views
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs

        dataset_cls = None
        if db == "scz":
            dataset_cls = SCZDataset
        elif db == "bd":
            dataset_cls = BDDataset
        elif db == "asd":
            dataset_cls = ASDDataset
        logger.debug(f"Dataset CLS : {dataset_cls.__name__}")

        input_transforms = {preproc: self.get_input_transforms(preproc=preproc, two_views=two_views)}       
        logger.debug(f"input_transforms : {input_transforms}")        
        self.dataset = dict()
        self.dataset["train"] = dataset_cls(root, preproc=preproc, split="train", two_views=two_views,
                                             transforms=input_transforms, target=labels)
        input_transforms = {preproc: self.get_input_transforms(preproc=preproc, two_views=False)}        
        self.dataset["validation"] = dataset_cls(root, preproc=preproc, split="val", two_views=two_views,
                                                  transforms=input_transforms, target=labels)
        self.dataset["test"] = dataset_cls(root, preproc=preproc, split="test",
                                           transforms=input_transforms, target=labels)
        self.dataset["test_intra"] = dataset_cls(root, preproc=preproc, split="test_intra", two_views=two_views,
                                                 transforms=input_transforms, target=labels)
            
    @staticmethod
    def get_collate_fn(two_views=False):
        if two_views:
            def collate_fn(list_samples):
                view_1 = torch.stack([torch.tensor(sample[0][0]) for sample in list_samples], dim=0).float()
                view_2 = torch.stack([torch.tensor(sample[0][1]) for sample in list_samples], dim=0).float()
                inputs = TwoViewItem(view_1=view_1, view_2=view_2)
                labels = torch.stack([torch.tensor(sample[1]) for sample in list_samples], dim=0).squeeze().float()
                return DataItem(inputs=inputs, labels=labels)
        else:
            def collate_fn(list_samples):
                """ After fetching a list of samples using the indices from sampler,
                the function passed as the collate_fn argument is used to collate lists
                of samples into batches.

                A custom collate_fn is used here to apply the transformations.

                See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn.
                """
                data = dict()
                data["inputs"] = torch.stack([torch.tensor(sample[0]) for sample in list_samples], dim=0).float()
                data["labels"] = torch.stack([torch.tensor(sample[1]) for sample in list_samples], dim=0).squeeze().float()
                return DataItem(**data)
        return collate_fn

    def get_dataloader(self, train=False, validation=False,
                       test=False, test_intra=False):

        assert test + test_intra <= 1, "Only one test accepted"
        tests_to_return = []
        if validation:
            tests_to_return.append("validation")
        if test:
            tests_to_return.append("test")
        if test_intra:
            tests_to_return.append("test_intra")
        test_loaders = dict()
        for t in tests_to_return:
            dataset = self.dataset[t]
            collate_fn = self.get_collate_fn(two_views=False)
            drop_last = True if len(dataset) % self.batch_size == 1 else False
            if drop_last:
                logger.warning(f"The last subject of the {t} set will not be tested ! "
                               f"Change the batch size ({self.batch_size}) to test on all subject ({len(dataset)})")
            test_loaders[t] = DataLoader(dataset, batch_size=self.batch_size,
                                         collate_fn=collate_fn, drop_last=drop_last,
                                         **self.dataloader_kwargs)
        if "test_intra" in test_loaders:
            assert "test" not in test_loaders
            test_loaders["test"] = test_loaders.pop("test_intra")
        if train:
            dataset = self.dataset["train"]
            if "test" in test_loaders.keys():
                sampler = SequentialSampler(dataset)
            else:
                sampler = RandomSampler(dataset)
            logger.info(f"Set {sampler.__class__.__name__} for train dataloader")
            drop_last = True if len(dataset) % self.batch_size == 1 else False
            collate_fn = self.get_collate_fn(two_views=self.two_views)
            _train = DataLoader(
                dataset, batch_size=self.batch_size, sampler=sampler,
                collate_fn=collate_fn, drop_last=drop_last,
                **self.dataloader_kwargs)
            return SetItem(train=_train, **test_loaders)
        else:
            return SetItem(**test_loaders)

    @staticmethod
    def get_input_transforms(preproc, two_views):
        input_transforms = []
        if preproc == "vbm":
            input_transforms.append(Normalize())
        if two_views:
            input_transforms.append(DAModule(transforms=(f"tf_{preproc}", )))
        return Compose(input_transforms)
    
    def __str__(self):
        return f"ClinicalDataManager({self.preproc}, {self.db})"
    

class TwoModalityDataManager(object):

    def __init__(self, root: str, db: str, weak_modality: str, strong_modality: str, 
                 labels: [str, List[str]] = None, two_views: bool = False, batch_size: int = 1, 
                 **dataloader_kwargs):

        assert db in ["scz", "bd", "asd"], f"Unknown db: {db}"
        assert weak_modality in ["skeleton", "vbm", "quasi-raw"], f"Unknown modality {weak_modality}"
        assert strong_modality in ["skeleton", "vbm", "quasi-raw"], f"Unknown modality {strong_modality}"

        self.db = db
        self.weak_modality = weak_modality
        self.strong_modality = strong_modality
        if labels is None:
            self.labels = []
        elif isinstance(labels, str):
            self.labels = [labels]
        elif isinstance(labels, list):
            self.labels = labels
        self.two_views = two_views
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs

        dataset_cls = None
        if db == "scz":
            dataset_cls = SCZDataset
        elif db == "bd":
            dataset_cls = BDDataset
        elif db == "asd":
            dataset_cls = ASDDataset
        logger.debug(f"Dataset CLS : {dataset_cls.__name__}")

        input_transforms = {weak_modality: self.get_input_transforms(preproc=weak_modality, two_views=two_views),
                            strong_modality: self.get_input_transforms(preproc=strong_modality, two_views=two_views)}        
        logger.debug(f"input_transforms : {input_transforms}")        
        self.dataset = dict()
        self.dataset["train"] = dataset_cls(root, preproc=[weak_modality, strong_modality], split="train", 
                                            two_views=self.two_views,
                                             transforms=input_transforms, target=self.labels)
        input_transforms = {weak_modality: self.get_input_transforms(preproc=weak_modality, two_views=False),
                            strong_modality: self.get_input_transforms(preproc=strong_modality, two_views=False)}         
        self.dataset["validation"] = dataset_cls(root, preproc=[weak_modality, strong_modality], split="val", 
                                                 two_views=False, transforms=input_transforms, target=self.labels)
        self.dataset["test"] = dataset_cls(root, preproc=[weak_modality, strong_modality], split="test",
                                           two_views=False, transforms=input_transforms, target=self.labels)
        self.dataset["test_intra"] = dataset_cls(root, preproc=[weak_modality, strong_modality], split="test_intra", 
                                                 two_views=False, transforms=input_transforms, target=self.labels)
            
    @staticmethod
    def get_collate_fn(weak_modality, strong_modality, labels, two_views=False):
        if two_views:
            def collate_fn(list_samples):
                data = TwoModalityItem(
                    weak= TwoViewItem(view_1=torch.stack([torch.tensor(sample[weak_modality][0]) for sample in list_samples], dim=0).float(),
                                      view_2=torch.stack([torch.tensor(sample[weak_modality][1]) for sample in list_samples], dim=0).float()),
                    strong=TwoViewItem(view_1=torch.stack([torch.tensor(sample[strong_modality][0]) for sample in list_samples], dim=0).float(),
                                       view_2=torch.stack([torch.tensor(sample[strong_modality][1]) for sample in list_samples], dim=0).float()),
                    labels=torch.stack([torch.tensor(np.asarray([sample[l] for l in labels])) 
                                              for sample in list_samples], dim=0).squeeze().float())
                return data
        else:
            def collate_fn(list_samples):
                data = TwoModalityItem(
                    weak=torch.stack([torch.tensor(sample[weak_modality]) for sample in list_samples], dim=0).float(),
                    strong=torch.stack([torch.tensor(sample[strong_modality]) for sample in list_samples], dim=0).float(),
                    labels=torch.stack([torch.tensor(np.asarray([sample[l] for l in labels])) 
                                              for sample in list_samples], dim=0).squeeze().float())
                return data
        return collate_fn

    def get_dataloader(self, train=False, validation=False,
                       test=False, test_intra=False):

        assert test + test_intra <= 1, "Only one test accepted"
        tests_to_return = []
        if validation:
            tests_to_return.append("validation")
        if test:
            tests_to_return.append("test")
        if test_intra:
            tests_to_return.append("test_intra")
        test_loaders = dict()
        for t in tests_to_return:
            dataset = self.dataset[t]
            collate_fn = self.get_collate_fn(strong_modality=self.strong_modality,
                                             weak_modality=self.weak_modality,
                                             labels=self.labels,
                                             two_views=False)
            drop_last = True if len(dataset) % self.batch_size == 1 else False
            if drop_last:
                logger.warning(f"The last subject of the {t} set will not be tested ! "
                               f"Change the batch size ({self.batch_size}) to test on all subject ({len(dataset)})")
            test_loaders[t] = DataLoader(dataset, batch_size=self.batch_size,
                                         collate_fn=collate_fn, drop_last=drop_last,
                                         **self.dataloader_kwargs)
        if "test_intra" in test_loaders:
            assert "test" not in test_loaders
            test_loaders["test"] = test_loaders.pop("test_intra")
        if train:
            dataset = self.dataset["train"]
            if "test" in test_loaders.keys():
                sampler = SequentialSampler(dataset)
            else:
                sampler = RandomSampler(dataset)
            logger.info(f"Set {sampler.__class__.__name__} for train dataloader")
            drop_last = True if len(dataset) % self.batch_size == 1 else False
            collate_fn = self.get_collate_fn(strong_modality=self.strong_modality,
                                             weak_modality=self.weak_modality,
                                             labels=self.labels,two_views=self.two_views)
            _train = DataLoader(
                dataset, batch_size=self.batch_size, sampler=sampler,
                collate_fn=collate_fn, drop_last=drop_last,
                **self.dataloader_kwargs)
            return SetItem(train=_train, **test_loaders)
        else:
            return SetItem(**test_loaders)

    @staticmethod
    def get_input_transforms(preproc, two_views):
        input_transforms = []
        if preproc == "vbm":
            input_transforms.append(Normalize())
        if two_views:
            input_transforms.append(DAModule(transforms=(f"tf_{preproc}", )))
        return Compose(input_transforms)
    
    def __str__(self):
        return f"TwoModalityDataManager({self.preweak_modality}, {self.strong_modality}, {self.db})"


if __name__ == "__main__":
    dataset = "scz"
    manager = TwoModalityDataManager(root="/neurospin/psy_sbox/analyses/2023_pauriau_sepmod/data/root", 
                                    db=dataset, weak_modality="skeleton", strong_modality="vbm",
                                    labels=["diagnosis", "sex"], batch_size=8, two_views=True,
                                    num_workers=8, pin_memory=True)
    loader = manager.get_dataloader(train=True)
    
    for input in loader.train:
        print(input.strong.view_1.shape)
        print(input.weak.view_2.shape)
        print(input.labels.shape)
        break
    for copinput in loader.train:
        print((copinput.labels == input.labels).all())
        break
    loader = manager.get_dataloader(train=True,
                                    test=True)
    
    for input in loader.train:
        print(input.strong.view_1.shape)
        print(input.weak.view_2.shape)
        print(input.labels.shape)
        break
    for copinput in loader.train:
        print((copinput.labels == input.labels).all())
        break