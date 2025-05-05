#!/usr/bin/env python
import unittest

import numpy as np
import torch
from scipy.special import comb

from epik.utils import (
    seq_to_one_hot,
    diploid_to_one_hot,
    get_full_space_one_hot,
    one_hot_to_seq,
    encode_seq,
    get_one_hot_subseq_key,
    get_binary_subseq_key,
    encode_seqs,
    calc_decay_rates,
    get_k_mutants,
    get_epistatic_coeffs_contrast_matrix,
    get_mut_effs_contrast_matrix,
    calc_distance_covariance,
    KrawtchoukPolynomials,
    SquaredMatMulOperator,
)


class UtilsTests(unittest.TestCase):
    def test_encode_seq(self):
        # Binary encoding
        seq = "ABBA"
        subseq_key = get_binary_subseq_key(alphabet="AB")
        assert len(subseq_key) == 2

        encoding = encode_seq(seq, subseq_key)
        assert encoding == [1, -1, -1, 1]
        assert len(subseq_key) == 5

        # Odd length
        seq = "ABBAB"
        encoding = encode_seq(seq, subseq_key)
        assert encoding == [1, -1, -1, 1, -1]
        assert len(subseq_key) > 5

        # Fail with missing characters
        seq = "ABBAC"
        try:
            encoding = encode_seq(seq, subseq_key)
            self.fail()
        except ValueError:
            pass

        # One hot encoding
        seq = "ACGT"
        subseq_key = get_one_hot_subseq_key(alphabet=seq)
        assert len(subseq_key) == 4

        encoding = encode_seq(seq, subseq_key)
        assert encoding == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        assert len(subseq_key) == 7

        encoding = encode_seq(seq[:3], subseq_key)
        assert encoding == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

    def test_encode_seqs(self):
        seqs = np.array(["AA", "AB", "BA", "BB"])
        alphabet = "AB"

        onehot = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1.0]])
        X = encode_seqs(seqs, alphabet, encoding_type="one_hot")
        assert np.allclose(X, onehot)

    def test_one_hot_encoding(self):
        X = np.array(["AA", "AB", "BA", "BB"])
        x = seq_to_one_hot(X).numpy()
        onehot = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1.0]])
        assert np.allclose(x - onehot, 0)

    def test_one_hot_encoding_to_seq(self):
        x = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1.0]])
        X = one_hot_to_seq(x, alleles=np.array(["A", "B"]))
        assert np.all(X == np.array(["AA", "AB", "BA", "BB"]))

    def test_get_full_one_hot(self):
        X = get_full_space_one_hot(seq_length=2, n_alleles=2)
        assert np.allclose(X, [[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])

        X = get_full_space_one_hot(seq_length=2, n_alleles=3)
        assert np.allclose(
            X,
            [
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 1],
            ],
        )

    def test_diploid_encoding(self):
        X = np.array(["00", "01", "11", "02", "22"])
        x = diploid_to_one_hot(X).numpy()
        assert x.shape == (5, 2, 3)

        h0 = np.array([[1, 1], [1, 0], [0, 0], [1, 0], [0, 0]])
        ht = np.array([[0, 0], [0, 1], [1, 1], [0, 0], [0, 0]])
        h1 = np.array([[0, 0], [0, 0], [0, 0], [0, 1], [1, 1]])
        y = np.stack([h0, ht, h1], axis=2)
        assert np.allclose(x, y)

    def test_calc_decay_rates(self):
        logit_rho = np.array([[0.0], [-0.69], [0.69]])
        log_p = np.full((3, 3), 1 / 3.0)
        decay_rates = calc_decay_rates(
            logit_rho, log_p, alleles=["A", "B", "C"], positions=[10, 12, 15]
        )

        assert np.all(decay_rates.columns == ["A", "B", "C"])
        assert np.all(decay_rates.index == [10, 12, 15])

        rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
        expected_decay_rates = (1 - (1 - rho) / (1 + 2 * rho)).flatten()
        decay_rates = decay_rates.mean(1).values.flatten()
        assert np.allclose(decay_rates, expected_decay_rates)

        # With sqrt
        decay_rates = calc_decay_rates(
            logit_rho, log_p, sqrt=True, alleles=["A", "B", "C"], positions=[10, 12, 15]
        )
        expected_decay_rates = (1 - np.sqrt((1 - rho) / (1 + 2 * rho))).flatten()
        decay_rates = decay_rates.mean(1).values.flatten()
        assert np.allclose(decay_rates, expected_decay_rates)

    def test_get_k_mutants(self):
        seq0 = "AGCT"
        seqs = get_k_mutants(seq0, alleles=seq0, k=2)

        for seq in seqs:
            d = np.sum([a1 != a2 for a1, a2 in zip(seq, seq0)])
            assert d == 2

        expected_doubles = 3**2 * comb(4, 2)
        assert len(seqs) == expected_doubles

    def test_get_get_contrast_matrices(self):
        seq0 = "ACGT"

        contrast_matrix = get_mut_effs_contrast_matrix(seq0, alleles=seq0)
        n_contrasts, n_seqs = 12, 13
        assert contrast_matrix.shape == (n_contrasts, n_seqs)
        assert np.allclose(contrast_matrix.sum(1), 0)
        assert np.allclose((contrast_matrix != 0).sum(1), 2)

        contrast_matrix = get_epistatic_coeffs_contrast_matrix(seq0, alleles=seq0)
        n_contrasts = 3**2 * comb(4, 2)
        n_seqs = 1 + 3 * 4 + n_contrasts
        assert contrast_matrix.shape == (n_contrasts, n_seqs)
        assert np.allclose(contrast_matrix.sum(1), 0)
        assert np.allclose((contrast_matrix != 0).sum(1), 4)

    def test_calc_distance_covariance(self):
        seq_length = 2
        n = seq_length + 1
        x = np.tril(np.ones((seq_length + 1, seq_length)), k=-1)
        x = np.stack([x, 1 - x], axis=2).reshape(n, 2 * seq_length)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.normal(torch.zeros(x.shape[0]))

        _, ns = calc_distance_covariance(x, y, seq_length)
        assert ns[0] == 3.0
        assert ns[1] == 4
        assert ns[2] == 2

        # Verify chunks in longer sequences
        seq_length = 12
        n = seq_length + 1
        x = get_full_space_one_hot(seq_length=seq_length, n_alleles=2)
        y = torch.normal(torch.zeros(x.shape[0]))
        covs1, ns1 = calc_distance_covariance(x, y, seq_length, chunk_size=None)
        covs2, ns2 = calc_distance_covariance(x, y, seq_length, chunk_size=100)
        assert np.allclose(covs1[0], 1, atol=0.1)
        assert np.allclose(np.abs(covs1[1:]), 0, atol=0.1)
        assert np.allclose(covs1, covs2)
        assert np.allclose(ns1, ns2)

    def test_squared_matmul_operator(self):
        x1 = np.random.normal(size=(10, 3))
        x2 = np.random.normal(size=(10, 3))

        A = x1 @ x2.T
        target = A**2
        v = np.random.normal(size=10)
        u1 = target @ v

        x1, x2 = torch.Tensor(x1), torch.Tensor(x2)
        v = torch.Tensor(v)
        H2 = SquaredMatMulOperator(x1, x2.T)
        u2 = H2 @ v
        assert np.allclose(u1, u2, atol=1e-4)

    def test_krawtchouk_polynomials(self):
        ws = KrawtchoukPolynomials(n_alleles=2, seq_length=3)
        ns = ws.n_alleles ** ws.seq_length

        # Get constant covariance
        log_lambdas = torch.Tensor([0.0, -16, -16, -16])
        w_d = ws.get_w_d(log_lambdas)
        covs = torch.ones(4)
        assert torch.allclose(w_d, covs / ns)

        # Using the interpolating function
        w_d = ws.basis @ ws.get_c_b(log_lambdas)
        assert torch.allclose(w_d, covs / ns)

        w_d = ws.basis @ ws.c_bk @ torch.exp(log_lambdas)
        assert torch.allclose(w_d, covs / ns)

        # Get additive covariance
        log_lambdas = torch.Tensor([-16, 0.0, -16, -16])
        w_d = ws.get_w_d(log_lambdas)
        covs = torch.Tensor([3.0, 1, -1, -3])
        assert torch.allclose(w_d, covs / ns)

        # Using the polynomial in distance
        w_d = ws.basis @ ws.get_c_b(log_lambdas)
        assert torch.allclose(w_d, covs / ns)

        # Check in larger dimensions
        log_lambdas = torch.linspace(0, -6, 9)
        ws = KrawtchoukPolynomials(n_alleles=4, seq_length=8)
        w_d1 = ws.get_w_d(log_lambdas)
        w_d2 = ws.basis @ ws.get_c_b(log_lambdas)
        w_d3 = ws.basis @ ws.c_bk @ torch.exp(log_lambdas)
        assert torch.allclose(w_d1, w_d2, atol=1e-3)
        assert torch.allclose(w_d1, w_d3, atol=1e-3)

        log_lambdas = torch.linspace(0, -10, 5)
        ws = KrawtchoukPolynomials(n_alleles=20, seq_length=4)
        w_d1 = ws.get_w_d(log_lambdas)
        w_d2 = ws.basis @ ws.get_c_b(log_lambdas)
        w_d3 = ws.basis @ ws.c_bk @ torch.exp(log_lambdas)
        assert torch.allclose(w_d1, w_d2, atol=1e-3)
        assert torch.allclose(w_d1, w_d3, atol=1e-3)

        log_lambdas = torch.linspace(0, -10, 17)
        ws = KrawtchoukPolynomials(n_alleles=2, seq_length=16)
        w_d1 = ws.get_w_d(log_lambdas)
        w_d2 = ws.basis @ ws.get_c_b(log_lambdas)
        w_d3 = ws.basis @ ws.c_bk @ torch.exp(log_lambdas)
        assert torch.allclose(w_d1, w_d2, atol=1e-3)
        assert torch.allclose(w_d1, w_d3, atol=1e-3)


if __name__ == "__main__":
    import sys

    sys.argv = ["", "UtilsTests"]
    unittest.main()
