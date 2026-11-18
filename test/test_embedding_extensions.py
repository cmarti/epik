"""Regression checks for masked FA kernels and ESM kernel dispatch."""
import unittest
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from linear_operator import to_dense

from epik.kernel import (
    ConnectednessFactorAnalysisKernel,
    FactorAnalysisKernel,
    JengaFactorAnalysisKernel,
    PositionARDESMKernel,
    PrecomputedKernel,
    get_kernel,
    get_named_kernel,
)
from epik.model import EpiK


class EmbeddingExtensionsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.alphabets = [list("ABC"), list("AB")]
        self.mask = {0: {2}, 1: {1}}
        self.fa_classes = (
            FactorAnalysisKernel,
            JengaFactorAnalysisKernel,
            ConnectednessFactorAnalysisKernel,
        )

    def test_observed_basis_and_covariance(self):
        for cls in self.fa_classes:
            with self.subTest(kernel=cls.__name__):
                kernel = cls(alphabet_list=self.alphabets, unseen_alleles=self.mask)
                self.assertEqual(tuple(kernel.P.shape), (5, 1))
                torch.testing.assert_close(kernel.P[[2, 3, 4]], torch.zeros(3, 1))
                torch.testing.assert_close(kernel.P.T @ kernel.P, torch.eye(1))
                torch.testing.assert_close(kernel.P[:3].sum(0), torch.zeros(1))
                x = torch.eye(5)
                cov = to_dense(kernel(x))
                diff = x[:, None, :] - x[None, :, :]
                squared_distance = torch.einsum(
                    "ijk,kl,ijl->ij", diff, kernel.get_M(), diff
                )
                expected = (kernel.log_var - squared_distance).exp()
                torch.testing.assert_close(cov, expected)
                torch.testing.assert_close(kernel(x, diag=True), cov.diag())
                torch.testing.assert_close(
                    kernel(x[:2], x[2:4], diag=True), cov[:2, 2:4].diag()
                )
                self.assertGreater(torch.linalg.eigvalsh(cov).min().item(), -1e-5)
                cov.sum().backward()
                for name, param in kernel.named_parameters():
                    if param.requires_grad:
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_v_mask_and_empty_subspace(self):
        for cls in self.fa_classes:
            kernel = cls(alphabet_list=self.alphabets, v=torch.tensor([1., 1., 0., 1., 0.]))
            self.assertEqual(tuple(kernel.P.shape), (5, 1))
            kernel = cls(alphabet_list=self.alphabets, v=torch.zeros(5))
            self.assertEqual(tuple(kernel.P.shape), (5, 0))
            torch.testing.assert_close(to_dense(kernel(torch.eye(5))), torch.ones(5, 5))
            kernel.use_keops = True
            product = kernel(torch.eye(5)) @ torch.ones(5, 1)
            torch.testing.assert_close(product, torch.full((5, 1), 5.0))
        with self.assertRaises(ValueError):
            FactorAnalysisKernel(alphabet_list=self.alphabets, v=torch.ones(3))

    def test_factories_and_model_api(self):
        for name in ("FactorAnalysis", "JengaFactorAnalysis", "ConnectednessFactorAnalysis"):
            with self.subTest(kernel=name):
                model = EpiK(
                    name,
                    alphabet_list=self.alphabets,
                    kernel_kwargs={"unseen_alleles": self.mask},
                )
                seqs = np.array(["AA", "BA", "AA", "BA"])
                model.set_data(
                    seqs, torch.tensor([0., 1., 0.2, 0.8]),
                    y_var=torch.full((4,), 0.1),
                )
                mll = model.calc_mll(method="cholesky")
                self.assertTrue(torch.isfinite(mll))
                mll.backward()
                model.fit(n_iter=2, learning_rate=0.01, mll_method="cholesky")
                predictions = model.predict(seqs, method="cholesky")
                self.assertTrue(np.isfinite(predictions["coef"]).all())
        names = (
            "Additive", "Pairwise", "VC", "Geometric", "Exponential",
            "Connectedness", "Jenga", "GeneralProduct", "MahalanobisRBF",
            "FactorAnalysis", "JengaFactorAnalysis", "ConnectednessFactorAnalysis",
        )
        for name in names:
            self.assertEqual(get_kernel(name, 3, 2).n_features, 6)
            self.assertEqual(get_kernel(name, alphabet_list=self.alphabets).n_features, 5)
        self.assertEqual(get_kernel("FactorAnalysis", 3, 2, ndim=None).ndim, 3)

    def test_checkpoint_round_trip(self):
        for cls in self.fa_classes:
            kernel = cls(alphabet_list=self.alphabets, unseen_alleles=self.mask).double()
            restored = cls(alphabet_list=self.alphabets, unseen_alleles=self.mask).double()
            with NamedTemporaryFile(suffix=".pth") as handle:
                kernel.save(handle.name)
                restored.load(handle.name, weights_only=True)
            x = torch.randn(4, 5, dtype=torch.float64)
            torch.testing.assert_close(to_dense(kernel(x)), to_dense(restored(x)))

    def test_esm_covariance_and_gradients(self):
        x = torch.randn(7, 6)
        for name, options in (
            ("Precomputed", {"ard_num_dims": 6}),
            ("PositionARDESM", {"feature_dim": 3}),
        ):
            kernel = get_named_kernel(name, self.alphabets, options)
            cov = to_dense(kernel(x))
            torch.testing.assert_close(kernel(x, diag=True), cov.diag())
            torch.testing.assert_close(kernel(x[:2], x[2:4], diag=True), cov[:2, 2:4].diag())
            cov.sum().backward()
            for param in kernel.parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())
            ndim = 6 if name == "Precomputed" else 3
            self.assertIsInstance(get_kernel(name, 3, 2, ndim=ndim), type(kernel))
        kernel = PositionARDESMKernel(2, 3)
        kernel.raw_outputscale.data.fill_(0.7)
        kernel.log_lengthscales.data.copy_(torch.tensor([0.3, -0.4]))
        scaled = x / kernel.lengthscales.repeat_interleave(3)
        expected = (0.7 - 0.5 * torch.cdist(scaled, scaled).square()).exp()
        torch.testing.assert_close(to_dense(kernel(x)), expected)
        with self.assertRaises(ValueError):
            kernel(torch.randn(3, 4)).to_dense()

    def test_esm_model_api(self):
        features = torch.randn(7, 6)
        y = torch.linspace(-1, 1, 7)
        for name, options in (
            ("Precomputed", {"ard_num_dims": 6}),
            ("PositionARDESM", {"feature_dim": 3}),
        ):
            with self.subTest(kernel=name):
                model = EpiK(name, alphabet_list=self.alphabets, kernel_kwargs=options)
                model.set_data(features, y, y_var=torch.full((7,), 0.1))
                model.fit(n_iter=2, mll_method="cholesky")
                result = model.predict(features[:3], method="cholesky", calc_variance=True)
                self.assertEqual(result.shape[0], 3)
                self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_keops_covariance_and_gradients(self):
        masked = {"alphabet_list": self.alphabets, "unseen_alleles": self.mask}
        configs = [(cls, masked, 5) for cls in self.fa_classes]
        configs.extend([
            (PrecomputedKernel, {"ard_num_dims": 6}, 6),
            (PositionARDESMKernel, {"seq_length": 2, "feature_dim": 3}, 6),
        ])
        for cls, options, width in configs:
            with self.subTest(kernel=cls.__name__):
                dense = cls(**options)
                keops = cls(use_keops=True, **options)
                keops.load_state_dict(dense.state_dict())
                x = torch.randn(7, width)
                rhs = torch.randn(7, 2)
                dense_result = dense(x) @ rhs
                keops_result = keops(x) @ rhs
                torch.testing.assert_close(keops_result, dense_result, atol=1e-5, rtol=1e-4)
                dense_result.sum().backward()
                keops_result.sum().backward()
                parameters = zip(keops.named_parameters(), dense.named_parameters())
                for (name, param), (_, ref) in parameters:
                    if param.requires_grad:
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)
                        torch.testing.assert_close(param.grad, ref.grad, atol=2e-4, rtol=2e-3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class GPUEmbeddingExtensionsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.alphabets = [list("ABC"), list("AB")]
        self.configs = [
            (cls, {"alphabet_list": self.alphabets, "unseen_alleles": {0: {2}}}, 5)
            for cls in (
                FactorAnalysisKernel,
                JengaFactorAnalysisKernel,
                ConnectednessFactorAnalysisKernel,
            )
        ]
        self.configs.extend([
            (PrecomputedKernel, {"ard_num_dims": 6}, 6),
            (PositionARDESMKernel, {"seq_length": 2, "feature_dim": 3}, 6),
        ])

    def test_cuda_covariances_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for cls, options, width in self.configs:
                with self.subTest(kernel=cls.__name__, dtype=dtype):
                    cpu = cls(**options).to(dtype=dtype)
                    dense = cls(**options).to(device="cuda", dtype=dtype)
                    keops = cls(use_keops=True, **options).to(device="cuda", dtype=dtype)
                    dense.load_state_dict(cpu.state_dict())
                    keops.load_state_dict(cpu.state_dict())
                    x = torch.randn(9, width, dtype=dtype)
                    rhs = torch.randn(9, 2, dtype=dtype)
                    expected = to_dense(cpu(x))
                    for kernel in (dense, keops):
                        x_gpu = x.cuda()
                        actual = to_dense(kernel(x_gpu))
                        self.assertTrue(actual.is_cuda)
                        torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=3e-4)
                        singleton = to_dense(kernel(x_gpu[:1], x_gpu[1:2]))
                        torch.testing.assert_close(singleton.cpu(), expected[:1, 1:2])
                        diagonal = kernel(x_gpu[:3], x_gpu[3:6], diag=True)
                        torch.testing.assert_close(diagonal.cpu(), expected[:3, 3:6].diag())
                        loss = (kernel(x_gpu) @ rhs.cuda()).sum()
                        loss.backward()
                    for (name, actual), (_, expected_param) in zip(
                        keops.named_parameters(), dense.named_parameters()
                    ):
                        if actual.requires_grad:
                            self.assertTrue(actual.is_cuda, name)
                            self.assertIsNotNone(actual.grad, name)
                            self.assertTrue(torch.isfinite(actual.grad).all(), name)
                            torch.testing.assert_close(
                                actual.grad, expected_param.grad, atol=3e-4, rtol=3e-3
                            )

    def test_cuda_fit_and_predict(self):
        seqs = np.array(["AA", "BA", "CA", "AB", "BB", "CB"])
        y = torch.tensor([-1., -0.3, 0.1, 0.4, 0.7, 1.1])
        features = torch.randn(6, 6)
        for name, options in (
            ("FactorAnalysis", {"ndim": 2}),
            ("JengaFactorAnalysis", {"ndim": 2}),
            ("ConnectednessFactorAnalysis", {}),
            ("Precomputed", {"ard_num_dims": 6}),
            ("PositionARDESM", {"feature_dim": 3}),
        ):
            with self.subTest(kernel=name):
                X = features if name in ("Precomputed", "PositionARDESM") else seqs
                model = EpiK(
                    name, alphabet_list=self.alphabets, device="cuda",
                    kernel_kwargs=dict(options, use_keops=True),
                )
                model.set_data(X, y, y_var=torch.full((6,), 0.1))
                model.fit(
                    n_iter=2, learning_rate=0.01, mll_method="cg",
                    cg_tol=1e-4, max_cg_iter=100,
                    n_trace_samples=16, n_lanczos_iter=20,
                )
                self.assertTrue(model.X.is_cuda)
                self.assertTrue(next(model.kernel.parameters()).is_cuda)
                self.assertTrue(np.isfinite(model.history["mll"]).all())
                prediction = model.predict(X[:2], method="cholesky", calc_variance=True)
                self.assertTrue(np.isfinite(prediction.to_numpy()).all())

    def test_cuda_masks_and_checkpoint(self):
        for cls, options, width in self.configs[:3]:
            with self.subTest(kernel=cls.__name__):
                v = torch.tensor([1., 1., 0., 1., 0.], device="cuda")
                kernel = cls(alphabet_list=self.alphabets, v=v).cuda()
                torch.testing.assert_close(kernel.P[[2, 3, 4]], torch.zeros(3, 1, device="cuda"))
                with NamedTemporaryFile(suffix=".pth") as handle:
                    kernel.save(handle.name)
                    restored = cls(alphabet_list=self.alphabets, v=v.cpu())
                    restored.load(handle.name, map_location="cpu", weights_only=True)
                x = torch.randn(4, width)
                torch.testing.assert_close(
                    to_dense(kernel(x.cuda())).cpu(), to_dense(restored(x))
                )
                empty = cls(
                    alphabet_list=self.alphabets, v=torch.zeros(5, device="cuda"),
                    use_keops=True,
                ).cuda()
                result = empty(x.cuda()) @ torch.ones(4, 1, device="cuda")
                torch.testing.assert_close(result, torch.full((4, 1), 4., device="cuda"))



class EncodedModelTests(unittest.TestCase):
    def test_encoded_inputs_preserve_dtype_and_predictions(self):
        seqs = np.array(["AA", "BA", "AB", "BB"])
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            with self.subTest(device=device):
                model = EpiK(
                    "FactorAnalysis", seq_length=2, alphabet=list("AB"),
                    device=device, dtype=torch.float64,
                    kernel_kwargs={"ndim": 1},
                )
                y = torch.tensor([0., 0.4, 0.8, 1.2], dtype=torch.float64)
                model.set_data(seqs, y, y_var=torch.full((4,), 0.1))
                sequence_prediction = model.predict(seqs[:2], method="cholesky")
                encoded = model.encode(seqs)
                model.set_data(encoded, y, y_var=torch.full((4,), 0.1))
                encoded_prediction = model.predict(encoded[:2], method="cholesky")
                self.assertEqual(model.X.dtype, torch.float64)
                self.assertEqual(model.y.dtype, torch.float64)
                for parameter in model.gp.parameters():
                    self.assertEqual(parameter.dtype, torch.float64)
                np.testing.assert_allclose(
                    sequence_prediction["coef"], encoded_prediction["coef"],
                    rtol=1e-10, atol=1e-10,
                )
