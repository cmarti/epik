#!/usr/bin/env python
import argparse
from os.path import exists

import numpy as np
import pandas as pd
import torch
from linear_operator.settings import max_cg_iterations

from epik.src.kernel import get_kernel
from epik.src.model import EpiK
from epik.src.settings import KERNELS
from epik.src.utils import (
    LogTrack,
    encode_seqs,
    guess_space_configuration,
)


def read_test_sequences(test_fpath):
    seqs = np.array([])
    if test_fpath is not None:
        seqs = np.array([line.strip().strip('"') for line in open(test_fpath)])
    return seqs


def main():
    description = "Runs Gaussian-Process regression on sequence space using data"
    description += " from quantitative phenotypes associated to their corresponding"
    description += " sequences. If provided, the variance of the estimated "
    description += " quantitative measure can be incorporated into the model"

    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("data", help="CSV table with genotype-phenotype data")

    options_group = parser.add_argument_group("Kernel options")
    options_group.add_argument(
        "-k",
        "--kernel",
        required=True,
        help="Kernel function to use {}".format(KERNELS),
    )
    help_msg = "pth file with model parameters"
    options_group.add_argument("--params", default=None, help=help_msg)
    help_msg = "CSV File containing log_lambdas to initialize VC model"
    options_group.add_argument("--log_lambdas0", default=None, help=help_msg)
    help_msg = "File containing log-variance to initialize site-product models"
    options_group.add_argument("--log_var0", default=None, help=help_msg)
    help_msg = "CSV File containing theta to initialize site-product models"
    options_group.add_argument("--theta0", default=None, help=help_msg)

    comp_group = parser.add_argument_group("Computational options")
    comp_group.add_argument(
        "--gpu", default=False, action="store_true", help="Use GPU for computation"
    )
    comp_group.add_argument(
        "-m", "--n_devices", default=1, type=int, help="Number of GPUs to use (1)"
    )
    comp_group.add_argument(
        "--max_contrasts",
        default=1,
        type=int,
        help="Maximum number of contrasts to compute simultaneusly (1)",
    )

    training_group = parser.add_argument_group("Training options")
    training_group.add_argument(
        "-n",
        "--n_iter",
        default=200,
        type=int,
        help="Number of iterations for optimization (200)",
    )
    training_group.add_argument(
        "-r",
        "--learning_rate",
        default=0.1,
        type=float,
        help="Learning rate for optimization (0.1)",
    )
    training_group.add_argument(
        "-ptn",
        "--pre_train_n_iter",
        default=0,
        type=int,
        help="Number of iterations to pre-train in a smaller subsample (0)",
    )

    pred_group = parser.add_argument_group("Prediction options")
    pred_group.add_argument(
        "-p", "--pred", help="File containing sequences for predicting genotype"
    )
    pred_group.add_argument(
        "-C",
        "--contrast_matrix",
        help="File containing contrasts for computing the posterior",
    )
    help_msg = "Comma separated list of sequence contexts in which to predict mutational effects"
    help_msg += " and epistatic coefficients (if --calc_epi_coeff is used)"
    pred_group.add_argument("-s", "--seq0", default=None, help=help_msg)
    pred_group.add_argument(
        "--calc_epi_coef",
        default=False,
        action="store_true",
        help="Compute posterior for epistatic coefficients around seq0",
    )
    pred_group.add_argument(
        "--calc_variance",
        default=False,
        action="store_true",
        help="Compute posterior variances",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "-o", "--output", required=True, help="Output file prefix"
    )

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data

    kernel_label = parsed_args.kernel
    params_fpath = parsed_args.params
    log_lambdas0_fpath = parsed_args.log_lambdas0
    log_var0_fpath = parsed_args.log_var0
    theta0_fpath = parsed_args.theta0

    gpu = parsed_args.gpu
    n_devices = parsed_args.n_devices
    n_iter = parsed_args.n_iter
    pre_train_n_iter = parsed_args.pre_train_n_iter
    learning_rate = parsed_args.learning_rate
    max_contrasts = parsed_args.max_contrasts

    pred_fpath = parsed_args.pred
    out_fpath = parsed_args.output
    contrast_matrix_fpath = parsed_args.contrast_matrix
    seq0s = parsed_args.seq0
    calc_variance = parsed_args.calc_variance
    calc_epi_coef = parsed_args.calc_epi_coef

    # Initialize logger
    log = LogTrack()
    log.write("Start analysis")

    # Load data
    data = pd.read_csv(data_fpath, index_col=0).dropna()
    log.write("Loaded {} sequences from {}".format(data.shape[0], data_fpath))

    # Get processed data
    seqs = data.index.values
    test_seqs = read_test_sequences(pred_fpath)
    config = guess_space_configuration(np.append(seqs, test_seqs))
    seq_length = config["seq_length"]
    alleles = np.unique(np.hstack(config["alphabet"]))
    n_alleles = alleles.shape[0]

    X = encode_seqs(seqs, alphabet=alleles)
    y = torch.Tensor(data.values[:, 0])

    if data.shape[1] > 1:
        y_var = data.values[:, 1]
        y_var[y_var < 0.0001] = 0.0001
    else:
        y_var = None

    # Get kernel
    log_lambdas0 = None
    if log_lambdas0_fpath is not None and exists(log_lambdas0_fpath):
        msg = "Loading log_lambdas0 from {}".format(log_lambdas0_fpath)
        log.write(msg)
        log_lambdas0 = torch.Tensor(pd.read_csv(log_lambdas0_fpath)['log_lambdas'])
    
    if log_var0_fpath is not None and exists(log_var0_fpath):
        msg = "Loading log_var0 from {}".format(log_var0_fpath)
        log.write(msg)
        with open(log_var0_fpath) as fhand:
            log_var0 = torch.Tensor(list([float(x.strip()) for x in fhand]))
    else:
        log_var0 = torch.log(((y - y.mean()) ** 2).mean())
        msg = "Initializing kernel with empirical variance: {:.2f}".format(
            torch.exp(log_var0)
        )
        log.write(msg)
    
    theta0 = None
    if theta0_fpath is not None and exists(theta0_fpath):
        msg = "Loading theta0 from {}".format(theta0_fpath)
        log.write(msg)
        theta0 = torch.Tensor(pd.read_csv(theta0_fpath, index_col=0).values)

    log.write("Selected {} kernel".format(kernel_label))
    kernel = get_kernel(
        kernel_label, n_alleles, seq_length, theta0=theta0, log_var0=log_var0,
        log_lambdas0=log_lambdas0,
    )

    # Define device
    device = torch.device("cuda") if gpu else None
    device_label = "GPU" if gpu else "CPU"
    log.write("Running computations on {}".format(device_label))

    # Create model
    log.write("Building model for Gaussian Process regression")
    model = EpiK(
        kernel,
        track_progress=True,
        train_noise=True,
        device=device,
        n_devices=n_devices,
        learning_rate=learning_rate,
    )
    model.set_data(X, y, y_var=y_var)

    # Load hyperparameters if provided
    if params_fpath is not None:
        if not exists(params_fpath):
            log.write("Hyperparameters file not found: {}".format(params_fpath))
        else:
            log.write("Load hyperparameters from {}".format(params_fpath))
            model.load(params_fpath)

    if pre_train_n_iter > 0:
        sample_size = 2000
        n_obs = X.shape[0]

        if n_obs > sample_size:
            idx = torch.Tensor(
                np.random.choice(np.arange(n_obs), sample_size, replace=False)
            ).to(dtype=torch.int)
            model.set_data(
                X[idx, :], y[idx], y_var=y_var[idx] if y_var is not None else y_var
            )
            log.write(
                "Pre-train hyperparameters with {} random points".format(
                    sample_size
                )
            )
            model.fit(n_iter=pre_train_n_iter)
            model.set_data(X, y, y_var=y_var)
        else:
            n_iter += pre_train_n_iter

    if n_iter > 0:
        log.write("Train hyperparameters by maximizing the evidence")
        model.fit(n_iter=n_iter)

    fpath = "{}.model_params.pth".format(out_fpath)
    log.write("Storing model parameteres at {}".format(fpath))
    model.save(fpath)

    # Predict phenotype in new sequences
    if test_seqs.shape[0] > 0:
        log.write("Obtain phenotypic predictions for test data")
        X_test = encode_seqs(test_seqs, alphabet=alleles)
        result = model.predict(
            X_test, calc_variance=calc_variance, labels=test_seqs
        )
        log.write("\tWriting predictions to {}".format(out_fpath))
        result.to_csv(out_fpath)

    if contrast_matrix_fpath is not None:
        if exists(contrast_matrix_fpath):
            log.write(
                "Loading contrast matrix from {}".format(contrast_matrix_fpath)
            )
            contrast_matrix = pd.read_csv(contrast_matrix_fpath, index_col=0)
            results = model.predict_contrasts(
                contrast_matrix,
                alleles,
                calc_variance=calc_variance,
                max_size=max_contrasts,
            )
            fpath = "{}.contrasts.csv".format(out_fpath)
            log.write("\tWriting estimates to {}".format(fpath))
            results.to_csv(fpath)
        else:
            log.write(
                "Contrast matrix not found at {}".format(contrast_matrix_fpath)
            )

    if seq0s is not None:
        for seq0 in seq0s.split(","):
            log.write(
                "Estimating mutational effects and epistatic coefficients around {}".format(
                    seq0
                )
            )
            results = model.predict_mut_effects(
                seq0, alleles, calc_variance=calc_variance, max_size=max_contrasts
            )
            if calc_epi_coef:
                df2 = model.predict_epistatic_coeffs(
                    seq0,
                    alleles,
                    calc_variance=calc_variance,
                    max_size=max_contrasts,
                )
                results = pd.concat([results, df2])
            fpath = "{}.{}_expansion.csv".format(out_fpath, seq0)
            log.write("\tWriting estimates to {}".format(fpath))
            results.to_csv(fpath)

    # Write execution time for tracking performance
    fpath = "{}.time.txt".format(out_fpath)
    log.write("Writing execution times to {}".format(fpath))
    with open(fpath, "w") as fhand:
        fhand.write("fit,{}\n".format(model.fit_time))
        if hasattr(model, "pred_time"):
            fhand.write("pred,{}\n".format(model.pred_time))
        if hasattr(model, "contrast_time"):
            fhand.write("contrast,{}\n".format(model.contrast_time))

    log.finish()


if __name__ == "__main__":
    main()
