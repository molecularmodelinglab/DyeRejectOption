import numpy as np
import pandas as pd
import random
from collections import defaultdict
from typing import List, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_split(data_col, use_scaffold, percent):
    if use_scaffold:
        mix_train_ids, mix_test_ids = generate_scaffold_split(data_col, sizes=(1 - percent, percent))
    else:
        mix_set = np.unique(data_col)
        mix_test_ids = random.sample(range(len(mix_set)), int(len(mix_set) * percent))
        mix_mask = np.ones(len(mix_set), dtype=bool)
        mix_mask[mix_test_ids] = False
        mix_train_ids = mix_set[mix_mask]
        mix_test_ids = mix_set[~mix_mask]

    return mix_train_ids, mix_test_ids


def generate_mixture_fold(data, mixture1: str, mixture2: str, percent: float = 0.30, use_scafold_split: bool = False,
                          scaffold_1: bool = True, scaffold_2: bool = False):
    """
    generate a mixture validation fold for a binary mixture

    :param data: either numpy array or pandas dataframe of the dataset
    :param mixture1: column name containing the class of mixture component 1
    :param mixture2: column name containing the class of mixture component 2
    :param percent: What percent should be held out from each list of components
    :param use_scafold_split: Do you want to use a scaffold clustering to make the component splits more strict
    :param scaffold_1: Use scaffold split for mixture component 1
    :param scaffold_2: Use scaffold split for mixture component 2
    :return: numpy array of data for each of the train, everything out, mixture 1 out and mixture 2 out sets
    """

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    mix1_train_ids, mix1_test_ids = get_split(data[mixture1], use_scafold_split and scaffold_1, percent)
    mix2_train_ids, mix2_test_ids = get_split(data[mixture2], use_scafold_split and scaffold_2, percent)

    mix1_train = np.isin(data[mixture1], mix1_train_ids)
    mix1_test = np.isin(data[mixture1], mix1_test_ids)

    mix2_train = np.isin(data[mixture2], mix2_train_ids)
    mix2_test = np.isin(data[mixture2], mix2_test_ids)

    train = np.logical_and(mix1_train, mix2_train)
    test = np.logical_and(mix1_test, mix2_test)
    out_1 = np.logical_and(mix2_train, mix1_test)
    out_2 = np.logical_and(mix1_train, mix2_test)

    return data[train], data[test], data[out_1], data[out_2]


####
# FOLLOWING ADAPTED FROM https://github.com/learningmatter-mit/uvvisml/blob/main/uvvisml/data/scaffold_splits.py
####


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False):
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]], use_indices: bool = False):
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def generate_scaffold_split(data, sizes=(0.8, 0.2), balanced=True, seed=0, return_set=False):
    assert sum(sizes) == 1

    # Split
    train_size, test_size = sizes[0] * len(data), sizes[1] * len(data)
    train, test = [], []
    train_scaffold_count, test_scaffold_count = 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data[:, 1], use_indices=True)

    # Seed randomness
    if seed == 0:
        seed = random.randint(0, int(1e6))
    r = random.Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        r.seed(seed)
        r.shuffle(big_index_sets)
        r.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()), key=lambda index_set: len(index_set), reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if return_set:
        return list(set(data[train, 1])), list(set(data[test, 1]))
    else:
        # Map from indices to data
        return data[train, :], data[test, :]
