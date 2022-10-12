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

from Mixtures.mixtures import generate_mixture_fold, generate_scaffold_split
from MultiStageModel.GlobalGlobalEnsemble import GlobalGlobalEnsemble
from MultiStageModel.GlobalLocalEnsemble import GlobalLocalEnsemble
from utils import make_equal_sized_bins


ALL_ENDPOINTS = ['Absorption max (nm)', 'Emission max (nm)', 'Lifetime (ns)', 'Quantum yield',
                 'log(e/mol-1 dm3 cm-1)', 'abs FWHM (cm-1)', 'emi FWHM (cm-1)', 'abs FWHM (nm)', 'emi FWHM (nm)']

# TODO add in support for custom metrics


def main(datafile: str,
         models,
         dyes_col: str = "SMILES",
         solvent_col: str = "Solvent",
         endpoints: str = "all",
         validation_method: str = "mixture_scaffold",
         num_folds: int = 5,
         min_solv_count: int = 1,
         dye_desc: str = "morgan",
         dye_fp_args: dict = None,
         solvent_desc: str = "morgan",
         solvent_fp_args: dict = None,
         padel_dye_lookup: str = None,
         save_filename: str = None,
         drop_smiles=None,
         drop_duplicates=True,
         verbose: int = 1):

    if endpoints == "all":
        endpoints = ALL_ENDPOINTS
    else:
        if isinstance(endpoints, str):
            endpoints = [endpoints]

    if validation_method in ["mixture", "mixture_scaffold"]:
        mixture = True
    else:
        mixture = False

    if verbose:
        print("Loading dataset...   ", end="")
        t0 = time.time()
    df = pd.read_csv(datafile)

    if not all([x in df.columns for x in endpoints]):
        raise ValueError(f"endpoint(s) must be a valid column in dataframe")

    df = df[~(df[dyes_col] == "[H]")]  # remove protons they are bad

    if drop_smiles is not None:
        if isinstance(drop_smiles, str):
            drop_smiles = [drop_smiles]
        for ds in drop_smiles:
            df = df[~(df[dyes_col] == ds)].copy()  # remove protons they are bad

    if drop_duplicates:
        df.drop_duplicates([dyes_col, solvent_col], keep=False, inplace=True)  # drop duplicates (drop all conflicts)

    # find the good solvents and trim to datapoints with those
    picked_solvents = [key for key, val in Counter(df[solvent_col]).items() if val >= min_solv_count]
    df = df[np.isin(df[solvent_col].to_numpy(), picked_solvents)]

    results = {}

    if not isinstance(models, list):
        models = [models]

    if not all(["fit" in dir(model) and "predict" in dir(model) for model in models]):
        raise ValueError("cannot find fit and predict function for all passed models")

    if verbose:
        print(f"done in {time.time() - t0} sec")

    for endpoint in endpoints:
        endpoint_results = {}
        df_endpoint = deepcopy(df[~np.isnan(df[endpoint])])

        # log scale quantum yield
        if endpoint == "Quantum yield":
            df_endpoint[endpoint] = np.log10(df_endpoint[endpoint].astype(float))
            df_endpoint[endpoint].replace(-np.inf, np.nan, inplace=True)  # log of 0 is -inf so drop those
            df_endpoint.dropna(subset=[endpoint], inplace=True)

        # generate rdkit mol objects and drop anything that rdkit cannot parse
        df_endpoint["ROMol_dye"] = df_endpoint.Solvent.apply(Chem.MolFromSmiles)
        df_endpoint["ROMol_solvent"] = df_endpoint.Solvent.apply(Chem.MolFromSmiles)
        df.dropna(subset=[dyes_col, solvent_col])

        # Process dye descriptors
        if verbose:
            print("Loading dye descriptors...   ", end="")
            t0 = time.time()

        if dye_desc == "padel":
            if padel_dye_lookup is None:
                raise ValueError("Need to pass PaDEL lookup table to use PaDEL dye descriptors. Use padel.py script "
                                 "to generate lookup table for your dataset")

            df_padel = pd.read_csv("/home/james/Projects/DyeModeling/data/dye_padel_table.csv")
            df_padel.drop_duplicates(subset=[dyes_col], inplace=True)
            df_padel = df_padel[df_padel.columns[2:]].set_index(dyes_col)

            X1 = np.array([df_padel.loc[x].to_list() for x in df_endpoint[dyes_col]])

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

        else:
            raise ValueError(f"Cannot process dye_desc of {dye_desc}: must be in ['morgan', 'morgan_count', 'padel',"
                             f" 'rdkit']")

        if verbose:
            print(f"done in {time.time() - t0} sec")

        # process solvent descriptors
        if verbose:
            print("Loading solvent descriptors...   ", end="")
            t0 = time.time()

        if dye_desc == "rdkit":
            desc_funcs = [x[1] for x in RDKitDescriptors.descList]
            X2 = np.array(df_endpoint["ROMol_solvent"].apply(
                lambda x: [func(x) for func in desc_funcs] if x is not None else [np.nan for _ in range(
                    len(desc_funcs))]).to_list()).astype(float)

        elif dye_desc == "morgan":
            if dye_fp_args is None:
                dye_fp_args = {"radius": 3, "nBits": 2048}
            else:
                if "radius" not in dye_fp_args.keys():
                    raise ValueError(f"Morgan fingerprint requires a radius argument, found args {solvent_fp_args}")
            df_endpoint["fps_solvent"] = df_endpoint.ROMol_solvent.apply(AllChem.GetMorganFingerprintAsBitVect,
                                                                 **solvent_fp_args)
            X2 = np.array([list(x) for x in df_endpoint.fps_solvent])

        elif dye_desc == "morgan_count":
            if dye_fp_args is None:
                dye_fp_args = {"radius": 3, "nBits": 2048}
            else:
                if "radius" not in dye_fp_args.keys():
                    raise ValueError(f"Morgan fingerprint requires a radius argument, found args {solvent_fp_args}")
            df_endpoint["fps_solvent"] = df_endpoint.ROMol_solvent.apply(AllChem.GetHashedMorganFingerprint, **solvent_fp_args)
            X2 = np.array([list(x) for x in df_endpoint.fps_solvent])

        else:
            raise ValueError(
                f"Cannot process solvent_desc of {solvent_desc}: must be in ['morgan', 'morgan_count', 'rdkit']")

        if verbose:
            print(f"done in {time.time() - t0} sec")

        # run the models
        for i, model in enumerate(models):
            # assign an unique name to the model
            try:
                name = f"{i}_{model.__name__()}_{dye_desc}_{solvent_desc}"
            except Exception:
                name = f"{i}_{model.__class__.__name__}_dye_{dye_desc}_solvent_{solvent_desc}"

            if verbose:
                print(f"Running model: {name}")

            X = np.concatenate((X1, X2), axis=1)
            y = df_endpoint[[endpoint]].to_numpy().reshape(-1, 1)

            data = np.concatenate((y, df_endpoint[dyes_col].to_numpy().reshape(-1, 1),
                                   df_endpoint[solvent_col].to_numpy().reshape(-1, 1), X), axis=1)

            # drop any columns with NA values
            data = data[~np.isnan(data[:, 3:].astype(float)).any(axis=1), :]

            # TODO need to drop cols where val is too big too for RDKIT

            fold_results = []
            fold_preds = []
            fold_pred_proba = []
            folds_true = []

            fold_preds_dyeout = []
            fold_pred_proba_dyeout = []
            folds_true_dyeout = []

            fold_preds_solvout = []
            fold_pred_proba_solvout = []
            folds_true_solvout = []

            if verbose:
                print("doing cross validation")
                t0 = time.time()
            for fold in range(num_folds):
                print(f"generating crossval splits for fold {fold}...")
                t1 = time.time()
                if mixture:
                    train, test, dye_out, solvent_out = generate_mixture_fold(data, data[:, 1], data[:, 2], use_scafold_split=validation_method == "mixture_scaffold", scaffold_1=True, scaffold_2=False)

                    train_X = train[:, 3:].astype(float)
                    train_y = train[:, 0].astype(float)

                    test_X = test[:, 3:].astype(float)
                    test_y = test[:, 0].astype(float)

                    dye_out_X = dye_out[:, 3:].astype(float)
                    dye_out_y = dye_out[:, 0].astype(float)

                    solvent_out_X = solvent_out[:, 3:].astype(float)
                    solvent_out_y = solvent_out[:, 0].astype(float)
                elif validation_method == "random":
                    np.random.shuffle(data)

                    train_X = data[len(data) // 5:, 3:].astype(float)
                    train_y = data[len(data) // 5:, 0].astype(float)

                    test_X = data[:len(data) // 5, 3:].astype(float)
                    test_y = data[:len(data) // 5, 0].astype(float)
                elif validation_method == "scaffold":
                    train, test, = generate_scaffold_split(data)

                    train_X = train[:, 3:].astype(float)
                    train_y = train[:, 0].astype(float)

                    test_X = test[:, 3:].astype(float)
                    test_y = test[:, 0].astype(float)
                else:
                    raise ValueError(f"validation method {validation_method} not recognized must be in "
                                     f"['random', 'mixture', 'scaffold', 'mixture_scaffold']")

                if verbose:
                    print(f"done in {t1 - time.time()}")

                if verbose:
                    print(f"fitting model on fold {fold}...   ", end="")
                    t1 = time.time()

                # fit the model
                model_c = deepcopy(model)
                model_c.fit(train_X, train_y)

                if verbose:
                    print(f"done in {t1 - time.time()}")

                # eval the fold
                if verbose:
                    print(f"evaluating fold {fold}...   ", end="")
                    t1 = time.time()

                y_pred_train = model_c.predict(train_X)
                y_pred_test = model_c.predict(test_X)

                try:
                    y_test_proba = model_c.predict_proba(test_X)
                except Exception:
                    y_test_proba = None

                if mixture:
                    y_pred_out2 = model_c.predict(dye_out_X)
                    y_pred_out3 = model_c.predict(solvent_out_X)

                    try:
                        y_test_proba_dyeout = model_c.predict_proba(dye_out_X)
                        y_test_proba_solvout = model_c.predict_proba(solvent_out_X)
                    except Exception:
                        y_test_proba_dyeout = None
                        y_test_proba_solvout = None

                fold_metrics = []

                if mixture:
                    folds_to_test = [(train_y, y_pred_train), (test_y, y_pred_test),
                                     (dye_out_y, y_pred_out2), (solvent_out_y, y_pred_out3)]
                else:
                    folds_to_test = [(train_y, y_pred_train), (test_y, y_pred_test)]

                for y_true, y_pred in folds_to_test:
                    fold_metrics += [
                        r2_score(y_true, y_pred.reshape(-1)),
                        mean_absolute_error(y_true, y_pred.reshape(-1)),
                        mean_squared_error(y_true, y_pred.reshape(-1), squared=False)
                    ]

                fold_results.append(fold_metrics)
                fold_preds.append(deepcopy(y_pred_test))
                fold_pred_proba.append(deepcopy(y_test_proba))
                folds_true.append(deepcopy(test_y))

                if mixture:
                    fold_preds_dyeout.append(deepcopy(y_pred_out2))
                    fold_pred_proba_dyeout.append(deepcopy(y_test_proba_dyeout))
                    folds_true_dyeout.append(deepcopy(dye_out_y))

                    fold_preds_solvout.append(deepcopy(y_pred_out3))
                    fold_pred_proba_solvout.append(deepcopy(y_test_proba_solvout))
                    folds_true_solvout.append(deepcopy(solvent_out_y))

                if verbose:
                    print(f"done in {t1 - time.time()}")

            # get average metrics
            mean_metrics = np.array(fold_results).mean(axis=0)
            std_metrics = np.array(fold_results).std(axis=0)

            # save results
            if mixture:
                endpoint_results[name] = {
                    "average_metrics": {
                        "train": {
                            "r2": mean_metrics[0],
                            "mae": mean_metrics[1],
                            "rmsd": mean_metrics[2]
                        },
                        "everything_out": {
                            "r2": mean_metrics[3],
                            "mae": mean_metrics[4],
                            "rmsd": mean_metrics[5]
                        },
                        "dye_out": {
                            "r2": mean_metrics[6],
                            "mae": mean_metrics[7],
                            "rmsd": mean_metrics[8]
                        },
                        "solvent_out": {
                            "r2": mean_metrics[9],
                            "mae": mean_metrics[10],
                            "rmsd": mean_metrics[11]
                        }
                    },
                    "std_metrics": {
                        "train": {
                            "r2": std_metrics[0],
                            "mae": std_metrics[1],
                            "rmsd": std_metrics[2]
                        },
                        "everything_out": {
                            "r2": std_metrics[3],
                            "mae": std_metrics[4],
                            "rmsd": std_metrics[5]
                        },
                        "dye_out": {
                            "r2": std_metrics[6],
                            "mae": std_metrics[7],
                            "rmsd": std_metrics[8]
                        },
                        "solvent_out": {
                            "r2": std_metrics[9],
                            "mae": std_metrics[10],
                            "rmsd": std_metrics[11]
                        }
                    },
                    "results_everything_out": {
                        "preds": fold_preds,
                        "pred_prob": fold_pred_proba,
                        "y_true": folds_true
                    },
                    "results_dye_out": {
                        "preds": fold_preds_dyeout,
                        "pred_prob": fold_pred_proba_dyeout,
                        "y_true": folds_true_dyeout
                    },
                    "results_solvent_out": {
                        "preds": fold_preds_solvout,
                        "pred_prob": fold_pred_proba_solvout,
                        "y_true": folds_true_solvout
                    }
                }
            else:
                endpoint_results[name] = {
                    "average_metrics": {
                        "train": {
                            "r2": mean_metrics[0],
                            "mae": mean_metrics[1],
                            "rmsd": mean_metrics[2]
                        },
                        "test": {
                            "r2": mean_metrics[3],
                            "mae": mean_metrics[4],
                            "rmsd": mean_metrics[5]
                        }
                    },
                    "std_metrics": {
                        "train": {
                            "r2": std_metrics[0],
                            "mae": std_metrics[1],
                            "rmsd": std_metrics[2]
                        },
                        "test": {
                            "r2": std_metrics[3],
                            "mae": std_metrics[4],
                            "rmsd": std_metrics[5]
                        }
                    },
                    "results_test": {
                        "preds": fold_preds,
                        "pred_prob": fold_pred_proba,
                        "y_true": folds_true
                    }
                }

            print(f"completed {name}\n")
        results[endpoint] = deepcopy(results)

    if save_filename is not None:
        import pickle
        with open(save_filename, "wb") as f:
            pickle.dump(results, f)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MultiStage reject option modeling for optical properties")

    parser.add_argument("--inpath", type=str, required=True,
                        help="file loc of data containing smiles and solvent endpoints")

    parser.add_argument("--models", type=str, nargs='+', default=["RidgeMultiStageEnsemble"],
                        help="names of models to use. Can be 'RidgeMultiStageEnsemble', 'GBTMultiStageEnsemble', 'GBT', 'RF' or 'Ridge'")

    parser.add_argument("--model_arg_file", type=str, default=None,
                        help="file loc for json file containing model parameters (see readme for info)")

    parser.add_argument("--dyes_col", type=str, default="SMILES",
                        help="name of column holding dye SMILES")

    parser.add_argument("--solvent_col", type=str, default="Solvent",
                        help="name of column holding solvent SMILES")

    parser.add_argument("--endpoints", type=str, nargs='+', default=["all"],
                        help="names of endpoint columns to use")

    parser.add_argument("--validation_method", type=str, default="mixture_scaffold",
                        help="name of validation approach to use: ['mixture', 'mixture_scaffold', 'random', 'scaffold']")

    parser.add_argument("--n_folds", type=int, default=5,
                        help="number of validation folds")

    parser.add_argument("--min_solvent_count", type=int, default=1,
                        help="minimum number of times a solvent must be present to keep in dataset")

    parser.add_argument("--dye_desc", type=str, default="morgan",
                        help="type of dye descriptor to be used must be in ['morgan', 'morgan_count', 'padel', 'rdkit']")

    parser.add_argument("--solvent_desc", type=str, default="morgan",
                        help="type of dye descriptor to be used must be in ['morgan', 'morgan_count', 'rdkit']")

    parser.add_argument("--dye_fp_args", type=str, nargs="+", default=["nBits:2048", 'radius:3'],
                        help="arguments for morgan fingerprint generation for dyes ei '--dye_fp_args nBits:2048 radius:3' sets nBits to 2048 and radius to 3")

    parser.add_argument("--solvent_fp_args", type=str, nargs="+", default=["nBits:256", 'radius:3'],
                        help="arguments for morgan fingerprint generation for solvents ei '--dye_fp_args nBits:2048 radius:3' sets nBits to 2048 and radius to 3")

    parser.add_argument("--model_arg_file", type=str, default=None,
                        help="file loc for json file containing model parameters (see readme for info)")

    parser.add_argument("--padel_dye_lookup", type=str, default=None,
                        help="file loc for csv containing padel descriptors for dye smiles")

    parser.add_argument("--save_filename", type=str, default=None,
                        help="file loc to save results. Defaults to not saving when unset")

    parser.add_argument("--drop_smiles", type=str, nargs="+", default=None,
                        help="dye smiles that you want you want to remove from the dataset")

    parser.add_argument("--drop_duplicates", action='store_true',
                        help="drop dye-solvent duplicates")

    parser.add_argument("--verbose", action='store_true',
                        help="drop dye-solvent duplicates")

    args = parser.parse_args()

    # TODO process fp settings and model setting dicts to make the models to pass to the main