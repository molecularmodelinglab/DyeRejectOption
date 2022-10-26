import pandas as pd
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as RDKitDescriptors
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from copy import deepcopy
from collections import Counter
import pickle
from Mixtures.mixtures import generate_mixture_fold, generate_scaffold_split
from MultiStageModel.GlobalGlobalEnsemble import GlobalGlobalEnsemble
from MultiStageModel.GlobalLocalEnsemble import GlobalLocalEnsemble
from utils import make_equal_sized_bins, ncs_descriptors

ALL_ENDPOINTS = ['Absorption max (nm)', 'Emission max (nm)', 'Quantum yield',
                 'log(e/mol-1 dm3 cm-1)', 'abs FWHM (nm)', 'emi FWHM (nm)']

c_dict = {
    'Absorption max (nm)': "max_abs",
    'Emission max (nm)': "max_emi",
    'Quantum yield': "log_qy",
    'log(e/mol-1 dm3 cm-1)': "log_ec",
    'abs FWHM (nm)': "fwhm_abs",
    'emi FWHM (nm)': "fwhm_emi"
}

# TODO add in support for custom metrics


def main(datafile: str,
         dyes_col: str = "SMILES",
         solvent_col: str = "Solvent",
         endpoints: str = "all",
         dye_desc: str = "morgan",
         dye_fp_args: dict = None,
         solvent_desc: str = "morgan",
         solvent_fp_args: dict = None,
         padel_dye_lookup: str = None,
         verbose: int = 1):

    if endpoints == "all":
        endpoints = ALL_ENDPOINTS
    else:
        if isinstance(endpoints, str):
            endpoints = [endpoints]

    if verbose:
        print("Loading dataset...   ", end="")
        t0 = time.time()
    df = pd.read_csv(datafile)

    if not all([x in df.columns for x in endpoints]):
        raise ValueError(f"endpoint(s) must be a valid column in dataframe")

    results = {}

    if dye_desc == "padel":
        if padel_dye_lookup is None:
            raise ValueError("Need to pass PaDEL lookup table to use PaDEL dye descriptors. Use padel.py script "
                             "to generate lookup table for your dataset")

        df_padel = pd.read_csv(padel_dye_lookup)
        df_padel.drop_duplicates(subset=[dyes_col], inplace=True)
        df_padel = df_padel[df_padel.columns[2:]].set_index(dyes_col)

    if verbose:
        print(f"done in {time.time() - t0} sec")

    for endpoint in endpoints:
        if verbose:
            print(f"Starting endpoint {endpoint}")
        endpoint_results = {}
        df_endpoint = deepcopy(df[~np.isnan(df[endpoint])])

        import pickle
        with open(f"{save_model}_{c_dict[endpoint]}.pkl", "rb") as f:
            model = pickle.load(f)

        # log scale quantum yield
        if endpoint == "Quantum yield":
            df_endpoint[endpoint] = np.log10(df_endpoint[endpoint].astype(float))
            df_endpoint[endpoint].replace(-np.inf, np.nan, inplace=True)  # log of 0 is -inf so drop those
            df_endpoint.dropna(subset=[endpoint], inplace=True)

        # generate rdkit mol objects and drop anything that rdkit cannot parse
        df_endpoint["ROMol_dye"] = df_endpoint.Solvent.apply(Chem.MolFromSmiles)
        df_endpoint["ROMol_solvent"] = df_endpoint.Solvent.apply(Chem.MolFromSmiles)
        df_endpoint.dropna(subset=[dyes_col, solvent_col], inplace=True)

        if len(df_endpoint) == 0:
            continue
        else:
            print("dataset size:", len(df_endpoint))
            print("num_uniq: ", len(set(df_endpoint["SMILES"])))

        # Process dye descriptors
        if verbose:
            print("Loading dye descriptors...   ", end="")
            t0 = time.time()

        if dye_desc == "padel":
            X1 = np.array(
                [df_padel.loc[x].to_list() if x in df_padel.index else [np.nan] * 2741 for x in df_endpoint[dyes_col]])

        elif dye_desc == "rdkit":
            desc_funcs = [x[1] for x in RDKitDescriptors.descList]
            X1 = np.array(df_endpoint["ROMol_dye"].apply(
                lambda x: [func(x) for func in desc_funcs] if x is not None else [np.nan for _ in range(
                    len(desc_funcs))]).to_list()).astype(float)

        elif dye_desc == "morgan":
            if dye_fp_args is None:
                dye_fp_args = {"radius": 3, "nBits": 2048}
            else:
                if "radius" not in dye_fp_args.keys():
                    raise ValueError(f"Morgan fingerprint requires a radius argument, found args {dye_fp_args}")
            df_endpoint["fps_dye"] = df_endpoint.ROMol_dye.apply(AllChem.GetMorganFingerprintAsBitVect, **dye_fp_args)
            X1 = np.array([list(x) for x in df_endpoint.fps_dye])

        elif dye_desc == "morgan_count":
            if dye_fp_args is None:
                dye_fp_args = {"radius": 3, "nBits": 2048}
            else:
                if "radius" not in dye_fp_args.keys():
                    raise ValueError(f"Morgan fingerprint requires a radius argument, found args {dye_fp_args}")
            df_endpoint["fps_dye"] = df_endpoint.ROMol_dye.apply(AllChem.GetHashedMorganFingerprint, **dye_fp_args)
            X1 = np.array([list(x) for x in df_endpoint.fps_dye])

        elif dye_desc == "ncs":
            X1 = ncs_descriptors(df_endpoint, dyes_col, 2048)

        else:
            raise ValueError(f"Cannot process dye_desc of {dye_desc}: must be in ['morgan', 'morgan_count', 'padel',"
                             f" 'rdkit', 'ncs']")

        if verbose:
            print(f"done in {time.time() - t0} sec")

        # process solvent descriptors
        if verbose:
            print("Loading solvent descriptors...   ", end="")
            t0 = time.time()

        if solvent_desc == "rdkit":
            desc_funcs = [x[1] for x in RDKitDescriptors.descList]
            X2 = np.array(df_endpoint["ROMol_solvent"].apply(
                lambda x: [func(x) for func in desc_funcs] if x is not None else [np.nan for _ in range(
                    len(desc_funcs))]).to_list()).astype(float)

        elif solvent_desc == "morgan":
            if solvent_fp_args is None:
                solvent_fp_args = {"radius": 3, "nBits": 256}
            else:
                if "radius" not in solvent_fp_args.keys():
                    raise ValueError(f"Morgan fingerprint requires a radius argument, found args {solvent_fp_args}")
            df_endpoint["fps_solvent"] = df_endpoint.ROMol_solvent.apply(AllChem.GetMorganFingerprintAsBitVect,
                                                                         **solvent_fp_args)
            X2 = np.array([list(x) for x in df_endpoint.fps_solvent])

        elif solvent_desc == "morgan_count":
            if solvent_fp_args is None:
                solvent_fp_args = {"radius": 3, "nBits": 256}
            else:
                if "radius" not in solvent_fp_args.keys():
                    raise ValueError(f"Morgan fingerprint requires a radius argument, found args {solvent_fp_args}")
            df_endpoint["fps_solvent"] = df_endpoint.ROMol_solvent.apply(AllChem.GetHashedMorganFingerprint,
                                                                         **solvent_fp_args)
            X2 = np.array([list(x) for x in df_endpoint.fps_solvent])

        elif solvent_desc == "ncs":
            X2 = ncs_descriptors(df_endpoint, solvent_col, 256)

        else:
            raise ValueError(
                f"Cannot process solvent_desc of {solvent_desc}: must be in ['morgan', 'morgan_count', 'rdkit', 'ncs']")

        if verbose:
            print(f"done in {time.time() - t0} sec")


        X = np.concatenate((X1, X2), axis=1)
        y = df_endpoint[[endpoint]].to_numpy().reshape(-1, 1)

        data = np.concatenate((y, df_endpoint[dyes_col].to_numpy().reshape(-1, 1),
                               df_endpoint[solvent_col].to_numpy().reshape(-1, 1), X), axis=1)

        data = data[~np.isnan(data[:, 3:].astype(float)).any(axis=1), :]

        X = data[:, 3:]
        y = data[:, 0]

        fold_results = []
        fold_preds = []
        fold_pred_proba = []
        folds_true = []

        y_pred_test = model.predict(X)

        try:
            y_test_proba = model.predict_proba(X)
        except Exception:
            y_test_proba = None

        fold_metrics = []

        folds_to_test = [(y, y_pred_test)]

        for y_true, y_pred in folds_to_test:
            fold_metrics += [
                r2_score(y_true, y_pred.reshape(-1)),
                mean_absolute_error(y_true, y_pred.reshape(-1)),
                mean_squared_error(y_true, y_pred.reshape(-1), squared=False)
            ]

        fold_results.append(fold_metrics)
        fold_preds.append(deepcopy(y_pred_test))
        fold_pred_proba.append(deepcopy(y_test_proba))
        folds_true.append(deepcopy(y))

        results[endpoint] = [fold_results, fold_preds, fold_pred_proba, folds_true]

    with open(f"{save_model}_preds.pkl", 'wb') as f:
        pickle.dump(results, f)

    return results



if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser("MultiStage reject option modeling for optical properties")
    #
    # parser.add_argument("--inpath", type=str, required=True,
    #                     help="file loc of data containing smiles and solvent endpoints")
    #
    # parser.add_argument("--models", type=str, nargs='+', default=["RidgeMultiStageEnsemble"],
    #                     help="names of models to use. Can be 'RidgeMultiStageEnsemble', 'GBTMultiStageEnsemble', 'GBT', 'RF' or 'Ridge'")
    #
    # parser.add_argument("--model_arg_file", type=str, default=None,
    #                     help="file loc for json file containing model parameters (see readme for info)")
    #
    # parser.add_argument("--dyes_col", type=str, default="SMILES",
    #                     help="name of column holding dye SMILES")
    #
    # parser.add_argument("--solvent_col", type=str, default="Solvent",
    #                     help="name of column holding solvent SMILES")
    #
    # parser.add_argument("--endpoints", type=str, nargs='+', default=["all"],
    #                     help="names of endpoint columns to use")
    #
    # parser.add_argument("--validation_method", type=str, default="mixture_scaffold",
    #                     help="name of validation approach to use: ['mixture', 'mixture_scaffold', 'random', 'scaffold']")
    #
    # parser.add_argument("--n_folds", type=int, default=5,
    #                     help="number of validation folds")
    #
    # parser.add_argument("--min_solvent_count", type=int, default=1,
    #                     help="minimum number of times a solvent must be present to keep in dataset")
    #
    # parser.add_argument("--dye_desc", type=str, default="morgan",
    #                     help="type of dye descriptor to be used must be in ['morgan', 'morgan_count', 'padel', 'rdkit']")
    #
    # parser.add_argument("--solvent_desc", type=str, default="morgan",
    #                     help="type of dye descriptor to be used must be in ['morgan', 'morgan_count', 'rdkit']")
    #
    # parser.add_argument("--dye_fp_args", type=str, nargs="+", default=["nBits:2048", 'radius:3'],
    #                     help="arguments for morgan fingerprint generation for dyes ei '--dye_fp_args nBits:2048 radius:3' sets nBits to 2048 and radius to 3")
    #
    # parser.add_argument("--solvent_fp_args", type=str, nargs="+", default=["nBits:256", 'radius:3'],
    #                     help="arguments for morgan fingerprint generation for solvents ei '--dye_fp_args nBits:2048 radius:3' sets nBits to 2048 and radius to 3")
    #
    # parser.add_argument("--model_arg_file", type=str, default=None,
    #                     help="file loc for json file containing model parameters (see readme for info)")
    #
    # parser.add_argument("--padel_dye_lookup", type=str, default=None,
    #                     help="file loc for csv containing padel descriptors for dye smiles")
    #
    # parser.add_argument("--save_filename", type=str, default=None,
    #                     help="file loc to save results. Defaults to not saving when unset")
    #
    # parser.add_argument("--drop_smiles", type=str, nargs="+", default=None,
    #                     help="dye smiles that you want you want to remove from the dataset")
    #
    # parser.add_argument("--drop_duplicates", action='store_true',
    #                     help="drop dye-solvent duplicates")
    #
    # parser.add_argument("--verbose", action='store_true',
    #                     help="drop dye-solvent duplicates")
    #
    # args = parser.parse_args()

    save_model = "s22_all_gbt"

    results = {}
    for e in ALL_ENDPOINTS:

        external = pd.read_csv(f"/home/james/Projects/MultiStageModelingColor/data/dye_datasets/d4c_22S_no_reicharts_f{c_dict[e]}_external.csv")
        train = pd.read_csv(f"/home/james/Projects/MultiStageModelingColor/data/dye_datasets/d4c_22S_no_reicharts_f{c_dict[e]}_train.csv")

        everything_out = external[~np.isin(external["SMILES"], train["SMILES"])]
        everything_out.to_csv(f"/home/james/Projects/MultiStageModelingColor/data/dye_datasets/d4c_22S_no_reicharts_f{c_dict[e]}_strict_external.csv")

        res = main(f"/home/james/Projects/MultiStageModelingColor/data/dye_datasets/d4c_22S_no_reicharts_f{c_dict[e]}_strict_external.csv",
                   endpoints=e, dye_desc="padel", solvent_desc="morgan_count",
                   padel_dye_lookup="/home/james/Projects/MultiStageModelingColor/data/dye_padel_table.csv")
        results[e] = res

    # TODO process fp settings and model setting dicts to make the models to pass to the main
