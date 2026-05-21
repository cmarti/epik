from typing import Any, Generator, List, Optional, Tuple, Union
import sys
import time
from collections import defaultdict
from itertools import combinations, product

import numpy as np
import pandas as pd
import torch
from linear_operator.operators import LinearOperator, MatmulLinearOperator
from scipy.special import logsumexp
from epik.settings import ALPHABETS


def get_random_sequences(n, seq_length, alphabet):
    alleles = np.random.choice(alphabet, size=(n, seq_length), replace=True)
    seqs = np.array(["".join(s) for s in alleles])
    return seqs


def get_simulated_mutagenesis(wt_sequence, alphabet, n_seqs, avg_n_mut):
    seq_length = len(wt_sequence)
    n_alleles = len(alphabet)
    wt_alleles = [alphabet.index(c) for c in wt_sequence]
    mut_prob = avg_n_mut / seq_length
    ps = np.full((seq_length, n_alleles), mut_prob / (n_alleles - 1))
    for i, c in enumerate(wt_alleles):
        ps[i, c] = 1 - mut_prob

    seqs = set()
    while len(seqs) < n_seqs:
        alleles = [np.random.choice(alphabet, p=ps[i]) for i in range(seq_length)]
        seqs.add("".join(alleles))
    seqs = np.array(list(seqs))
    return seqs


def get_one_hot_subseq_key(alphabet, max_l=1):
    subseq_key = {}
    for i, c in enumerate(alphabet):
        z = [0] * len(alphabet)
        z[i] = 1
        subseq_key[c] = z

    if max_l > 1:
        for k in range(2, max_l):
            for alleles in product(alphabet, repeat=k):
                seq = "".join(alleles)
                subseq_key[seq] = []
                for c in alleles:
                    subseq_key[seq] += subseq_key[c]
    return subseq_key


def get_binary_subseq_key(alphabet):
    if len(alphabet) != 2:
        raise ValueError("Alphabet length must be 2")

    subseq_key = {alphabet[0]: [1], alphabet[1]: [-1]}
    return subseq_key


def encode_seq(seq, subseq_key, max_l=4):
    try:
        return subseq_key[seq]
    except KeyError:
        sl = len(seq)
        if sl == 1:
            raise ValueError("Missing characters in `subseq_key`")

        i = sl // 2
        enc1 = encode_seq(seq[:i], subseq_key=subseq_key)
        enc2 = encode_seq(seq[i:], subseq_key=subseq_key)
        encoding = enc1 + enc2

        if sl <= max_l:
            subseq_key[seq] = encoding

        return encoding


def get_alleles(c, alleles=None):
    if alleles is not None:
        return alleles
    else:
        return np.unique(c)


def encode_seqs(seqs, alphabet, encoding_type="one_hot", max_n=500):
    """
    Returns a torch.Tensor with the encoding of the provided sequences

    Parameters
    ----------
    seqs : array-like of shape (n_sequences,)
        Array containing a list of sequences
    alphabet : array-like of shape (n_alleles, )
        Iterable object with the list of alleles in the sequences
    encoding_type : 'one_hot' or 'binary'
        Type of encoding to use. If `encoding_type='one_hot'`,
        sequences will be encoded under the one-hot encoding
        with one column for every allele and site.
        If `encoding_type='binary'` and there are only two possible
        alleles, then sequences will be encoded as binary, with a
        single column for every site taking values (-1, 1)
    max_n : int (500)
        Maximum number of features to allow in the encoding

    Returns
    -------
    X : torch.Tensor of shape (n_sequences, n_features)
        Tensor with the numerical encoding of the input sequences
    """

    max_l = max_n // len(alphabet)
    if encoding_type == "one_hot":
        subseq_key = get_one_hot_subseq_key(alphabet)
    elif encoding_type == "binary":
        subseq_key = get_binary_subseq_key(alphabet)
    else:
        raise ValueError("encoding_type can only be `one_hot` or `binary`")

    X = get_tensor(
        [encode_seq(seq, subseq_key=subseq_key, max_l=max_l) for seq in seqs]
    )
    return X


def seq_to_one_hot(X, alleles=None, alleles_list=None):
    if alleles_list is not None and alleles is not None:
        raise ValueError("Provide only one of `alleles` or `alleles_list`")

    m = np.array([[a for a in x] for x in X])
    onehot = []
    for i in range(m.shape[1]):
        c = m[:, i]
        for allele in get_alleles(c, alleles=alleles):
            onehot.append(get_tensor(c == allele))
    onehot = torch.stack(onehot, 1).contiguous()
    return onehot


def seq_to_binary(X, ref):
    m = np.array([[a for a in x] for x in X])
    onehot = []
    for i in range(m.shape[1]):
        c = m[:, i]
        onehot.append(2 * get_tensor(c == ref) - 1)
    onehot = torch.stack(onehot, 1).contiguous()
    return onehot


def one_hot_to_seq(x, alleles):
    ncol = x.shape[1]
    alpha = alleles.shape[0]
    if ncol % alpha != 0:
        raise ValueError("Check that the number of alleles is correct")
    seq_length = int(ncol / alpha)
    a = np.hstack([alleles] * seq_length)
    X = ["".join(a[i == 1]) for i in x]
    return X


def diploid_to_one_hot(X, dtype=torch.float32):
    m = torch.tensor([[int(a) for a in x] for x in X])
    onehot = torch.stack([m == 0, m == 1, m == 2], 2).to(dtype=dtype)
    return onehot


def to_device(tensor, output_device=None):
    if output_device is not None and output_device != -1:
        tensor = tensor.to(output_device)
    return tensor


def get_tensor(ndarray, dtype=torch.float32, device=None):
    if not torch.is_tensor(ndarray):
        ndarray = torch.tensor(ndarray, dtype=dtype)
    if ndarray.dtype != dtype:
        ndarray = ndarray.to(dtype=dtype)
    return to_device(ndarray, output_device=device)


def to_numpy(v):
    u = v.detach()
    if u.is_cuda:
        u = u.cpu()
    return u.numpy()


def get_gpu_memory(device=None):
    mem = torch.cuda.memory_allocated(device) / 2**20
    suffix = "MB"

    if mem < 1:
        mem = mem * 2**10
        suffix = "KB"

    return "{:.0f}{}".format(mem, suffix)


class LogTrack(object):
    def __init__(self, fhand=None):
        if fhand is None:
            fhand = sys.stderr
        self.fhand = fhand
        self.start = time.time()

    def write(self, msg, add_time=True):
        if add_time:
            msg = "[ {} ] {}\n".format(time.ctime(), msg)
        else:
            msg += "\n"
        self.fhand.write(msg)
        self.fhand.flush()

    def finish(self):
        t = time.time() - self.start
        self.write("Finished succesfully. Time elapsed: {:.1f} s".format(t))


def guess_space_configuration(seqs):
    """
    Guess the sequence space configuration from a collection of sequences
    This allows to have different number of alleles per site and maintain
    the order in which alleles appear in the sequences when enumerating the
    alleles per position

    Parameters
    ----------
    seqs: array-like of shape (n_genotypes,)
        Vector or list containing the sequences from which we want to infer
        the space configuration


    Returns
    -------
    config: dict with keys {'seq_length', 'n_alleles', 'alphabet'}
            Returns a dictionary with the inferred configuration of the discrete
            space where the sequences come from.

    """

    alleles = defaultdict(dict)
    for seq in seqs:
        for i, a in enumerate(seq):
            alleles[i][a] = 1
    length = len(alleles)
    config = {
        "seq_length": length,
        "n_alleles": [len(alleles[i]) for i in range(length)],
        "alphabet": [[a for a in alleles[i].keys()] for i in range(length)],
    }
    return config


def split_training_test(X, y, y_var=None, ptrain=0.8, dtype=None):
    """
    Splits a dataset into training and test subsets

    Parameters
    ----------
    X : torch.Tensor of shape (n_sequences, n_features)
        Tensor with the numerical encoding of the input sequences
    y : torch.Tensor of shape (n_sequence,)
            Tensor containing the phenotypic measurements for each
            sequence in `X`
    y_var : torch.Tensor of shape (n_sequence,) or None
        If `y_var=None` it is assumed that there is no uncertainty
        in the measurements. Otherwise, Tensor containing the
        variance of the measurements in `y`.
    ptrain : float (0.8)
        Proportion of the dataset to keep as training set
    dtype : torch.dtype (torch.float32)
        dtype to use for storing the output tensors

    Returns
    -------
    output: tuple of size (5,)
        Tuple containing (train_X, train_y, test_X, test_y, test_y_var),
        where `X` is the matrix encoding the input sequences,
        `y` the associated measurements and `y_var` the variance of
        those measurements.
    """

    ps = np.random.uniform(size=X.shape[0])
    train = ps <= ptrain
    train_x, train_y = X[train], y[train]

    if y_var is None:
        train_y_var = None
    else:
        train_y_var = y_var[train]

    test = ps > ptrain
    test_x, test_y = X[test], y[test]

    output = [train_x, train_y, test_x, test_y, train_y_var]
    if dtype is not None:
        output = [get_tensor(a, dtype=dtype) if a is not None else None for a in output]
    return output


def ps_to_variances(ps):
    v = 1 / ps
    v = (v.T / v.sum(1)).T
    return v


def get_full_space_one_hot(seq_length, n_alleles, dtype=torch.float32):
    n = n_alleles**seq_length
    i = torch.arange(n)

    c = i
    one_hot = []
    for _ in range(seq_length):
        r = c % n_alleles
        for j in range(n_alleles):
            one_hot.append(r == j)
        c = torch.div(c, n_alleles, rounding_mode="floor")
    X = torch.vstack(one_hot).T.to(dtype=dtype).contiguous()
    return X


def get_full_space_binary(seq_length, dtype=torch.float32):
    if seq_length == 1:
        return torch.tensor([[1], [-1]], dtype=dtype)
    else:
        b1 = get_full_space_binary(seq_length - 1, dtype=dtype)
        ones = torch.ones((b1.shape[0], 1), dtype=dtype)

        return torch.vstack([torch.hstack([b1, ones]), torch.hstack([b1, -ones])])


def log1mexp(x):
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    two = torch.tensor([2.0], dtype=x.dtype).to(device=x.device)
    mask = -torch.log(two) < x  # x < 0
    return torch.where(mask, (-x.expm1()).log(), (-x.exp()).log1p())


def calc_vandermonde_inverse(values, n=None):
    s = values.shape[0]
    if n is None:
        n = s

    B = np.zeros((n, s))
    idx = np.arange(s)
    sign = np.zeros((n, s))

    for k in range(s):
        v_k = values[idx != k]
        norm_factor = 1 / np.prod(v_k - values[k])

        for power in range(n):
            v_k_combs = list(combinations(v_k, s - power - 1))
            p = np.sum([np.prod(v) for v in v_k_combs])
            B[power, k] = sign[power, k] * p * norm_factor
            sign[power, k] = (-1) ** (power)
    return B


def log_factorial(x):
    return torch.lgamma(x + 1)


def log_comb(n, k):
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)


def calc_decay_rates(logit_rho, log_p, sqrt=False, alleles=None, positions=None):
    rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
    p = np.exp(log_p - np.expand_dims(logsumexp(log_p, axis=1), 1))
    eta = (1 - p) / p

    decay_factors = (1 - rho) / (1 + eta * rho)
    if sqrt:
        decay_factors = np.sqrt(decay_factors)

    decay_rates = pd.DataFrame(1 - decay_factors, index=positions, columns=alleles)
    return decay_rates


def get_k_mutants(seq0, alleles, k=2):
    positions = np.arange(len(seq0))
    seqs = []
    for sites in combinations(positions, k):
        for new_alleles in product(alleles, repeat=k):
            new_seq = [c for c in seq0]
            skip = False
            for a, s in zip(new_alleles, sites):
                if a == seq0[s]:
                    skip = True
                    break
                new_seq[s] = a
            if not skip:
                seqs.append("".join(new_seq))
    return seqs


def get_mutant_seq(seq, sites, alleles):
    mut = [c for c in seq]
    for p, a in zip(sites, alleles):
        mut[p] = a
    return "".join(mut)


def get_epistatic_coeffs_contrast_matrix(seq: str, alphabet_list: List[List[str]]):
    """
    Generate a contrast matrix for epistatic coefficients around a
    reference sequence.

    Parameters
    ----------
    seq : str
        The reference sequence for which the contrast matrix is generated.
    alphabet_list : List[List[str]]
        A list of lists where each inner list represents the alphabet of
        possible alleles for the corresponding position in the sequence.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the contrast matrix. Rows correspond to
        mutations, and columns correspond to sequences. The values indicate
        the contrast coefficients for each sequence.
    """

    # Check input
    if len(seq) != len(alphabet_list):
        msg = f"Ensure seq lenght {len(seq)} matches the size of the "
        msg += f"alphabet_list ({len(alphabet_list)})"
        raise ValueError(msg)

    for p, (c, alphabet) in enumerate(zip(seq, alphabet_list)):
        if c not in alphabet:
            msg = f"Unexpected allele {c} found at position {p} "
            msg += f"with alphabet {alphabet}"
            raise ValueError(msg)

    # Prepare contrasts
    contrasts = {}
    pos_alphabets = list(enumerate(alphabet_list))
    for (s1, alphabet1), (s2, alphabet2) in combinations(pos_alphabets, 2):
        c1, c2 = seq[s1], seq[s2]
        alleles1 = [a for a in alphabet1 if a != c1]
        alleles2 = [a for a in alphabet2 if a != c2]

        for a1, a2 in product(alleles1, alleles2):
            seqs = [
                seq,
                get_mutant_seq(seq, sites=[s1], alleles=[a1]),
                get_mutant_seq(seq, sites=[s2], alleles=[a2]),
                get_mutant_seq(seq, sites=[s1, s2], alleles=[a1, a2]),
            ]
            label = "{}{}{}_{}{}{}".format(c1, s1, a1, c2, s2, a2)
            values = [1, -1, -1, 1]
            contrasts[label] = dict(zip(seqs, values))
    contrasts_matrix = pd.DataFrame(contrasts).fillna(0).T
    return contrasts_matrix


def get_mut_effs_contrast_matrix(
    seq: str, alphabet_list: List[List[str]]
) -> pd.DataFrame:
    """
    Generate a contrast matrix for mutational effects around a
    reference sequence.

    Parameters
    ----------
    seq : str
        The reference sequence for which the contrast matrix is generated.
    alphabet_list : List[List[str]]
        A list of lists where each inner list represents the alphabet of
        possible alleles for the corresponding position in the sequence.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the contrast matrix. Rows correspond to
        mutations, and columns correspond to sequences. The values indicate
        the contrast coefficients for each sequence.
    """

    if len(seq) != len(alphabet_list):
        msg = f"Ensure seq lenght {len(seq)} matches the size of the "
        msg += f"alphabet_list ({len(alphabet_list)})"
        raise ValueError(msg)

    contrasts = {}
    for s, (c, alphabet) in enumerate(zip(seq, alphabet_list)):
        if c not in alphabet:
            msg = f"Unexpected allele {c} found at position {s} "
            msg += f"with alphabet {alphabet}"
            raise ValueError(msg)

        for a in [a for a in alphabet_list[s] if a != c]:
            seqs = [seq, get_mutant_seq(seq, sites=[s], alleles=[a])]
            label = "{}{}{}".format(c, s, a)
            values = [-1, 1]
            contrasts[label] = dict(zip(seqs, values))

    contrasts_matrix = pd.DataFrame(contrasts).fillna(0).T
    return contrasts_matrix


def calc_distance_covariance(X, y, seq_length, chunk_size=None):
    if chunk_size is None:
        chunk_size = X.shape[0]

    cov = torch.zeros(seq_length + 1)
    ns = torch.zeros(seq_length + 1)

    nchunks = int(X.shape[0] / chunk_size) + 1
    ones = torch.ones_like(y)

    for i in range(nchunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        s = X[start:end, :] @ X.T
        chunk_y = y[start:end]
        chunk_ones = torch.ones_like(chunk_y)

        for v in range(seq_length + 1):
            d = int(seq_length - v)
            A = (s == v).to(dtype=torch.float)
            ns_d = torch.dot(chunk_ones, A @ ones).item()
            ns[d] += ns_d
            cov[d] += torch.dot(chunk_y, A @ y).item()
    cov = cov / ns
    cov[torch.isnan(cov)] = 0.0
    return (cov, ns)


class KrawtchoukPolynomials(object):
    def __init__(self, n_alleles, seq_length, max_k=None):
        if max_k is None or max_k > seq_length:
            max_k = seq_length

        self.n_alleles = n_alleles
        self.seq_length = seq_length
        self.max_k = max_k

        self.calc_log_w_dkq()
        self.calc_c_bk()

    def calc_log_w_dkq(self):
        size1 = self.seq_length + 1
        size2 = self.max_k + 1
        log_a_minus_1 = np.log(self.n_alleles - 1)
        # l_log_a = self.seq_length * np.log(self.n_alleles)
        d = torch.arange(size1).reshape((size1, 1, 1))
        k = torch.arange(size2).reshape((1, size2, 1))
        q = torch.arange(size2).reshape((1, 1, size2))

        self.log_w_dkq = (
            (k - q) * log_a_minus_1
            + log_comb(d, q)
            + log_comb(self.seq_length - d, k - q)
            # - l_log_a
        )
        self.w_dkq_sign = (-1.0) ** q
        self.w_dk = (self.w_dkq_sign * torch.exp(self.log_w_dkq)).sum(-1)

    def calc_w_d(self, w_dkq_sign, log_w_dkq, log_lambdas):
        w_dkq = torch.exp(log_w_dkq + log_lambdas[None, :, None])
        return (w_dkq_sign * w_dkq).sum((1, 2))

    def calc_c_bk(self):
        theta = -np.log(0.1) / self.seq_length
        x = torch.arange(self.seq_length + 1).to(dtype=torch.float)
        self.basis = torch.exp(-theta * torch.abs(x.unsqueeze(0) - x.unsqueeze(1)))
        L = torch.linalg.cholesky(self.basis)
        self.c_bd = torch.cholesky_inverse(L)
        self.c_bk = self.c_bd @ self.w_dk

    def get_w_d(self, log_lambdas):
        return self.calc_w_d(self.w_dkq_sign, self.log_w_dkq, log_lambdas)

    def get_c_b(self, log_lambdas):
        w_d = self.get_w_d(log_lambdas)
        return self.c_bd @ w_d

    def calc_lambdas(self, covs, ns=None):
        if ns is None:
            lambdas = torch.linalg.solve(self.w_dk, covs)
        else:
            WD = self.w_dk.T * ns.unsqueeze(0)
            A = WD @ self.w_dk
            b = WD @ covs
            lambdas = torch.linalg.solve(A, b)
        return lambdas


class WkAligner(torch.nn.Module):
    def __init__(self, n_alleles, seq_length, max_k=None):
        super().__init__()
        self.n_alleles = n_alleles
        self.seq_length = seq_length
        self.ws = KrawtchoukPolynomials(n_alleles, seq_length, max_k)

    def set_data(self, cov, ns=None):
        self.cov = cov
        self.ns = ns
        if self.ns is None:
            self.ns = torch.ones_like(cov)

        b = self.cov[1] / (self.cov[0] - self.cov[1])
        beta0 = np.log(self.cov[0]) + self.seq_length * (
            np.log(1 + self.n_alleles * b) - np.log(1 + b)
        )
        beta1 = np.log(1 + self.n_alleles * b)
        k = torch.arange(self.ws.max_k + 1).to(dtype=torch.float)
        log_lambdas0 = beta0 - beta1 * k
        self.log_lambdas = torch.nn.Parameter(log_lambdas0)

    def predict(self, log_lambdas):
        return self.ws.get_w_d(log_lambdas)

    def calc_loss(self, log_lambdas):
        w_d = self.predict(log_lambdas)
        rmse = torch.sum(torch.square(self.cov - w_d) * self.ns) / self.ns.sum()
        return rmse

    def fit(self, n_iter=20, lr=0.5):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for i in range(n_iter):
            optimizer.zero_grad()
            loss = self.calc_loss(self.log_lambdas)
            loss.backward()
            optimizer.step()


class SquaredMatMulOperator(LinearOperator):
    def __init__(self, x1, x2):
        # Match batch dimensions
        self.x1 = x1
        self.x2 = x2
        super().__init__(x1, x2)

    def _size(self):
        return (self.x1.shape[0], self.x2.shape[1])

    def _matmul(self, rhs):
        u = torch.einsum("iy,yx,xm,zx,iz->im", self.x1, self.x2, rhs, self.x2, self.x1)
        return u

    def _t_matmul(self, rhs):
        return self._matmul(rhs)


class HammingDistanceCalculator(object):
    def __init__(self, seq_length, shift, scale):
        self.seq_length = seq_length
        self.shift = shift
        self.scale = scale

    def __call__(self, x1, x2):
        v = self.shift + self.scale * self.seq_length
        ones1 = torch.ones((x1.shape[0], 1), device=x1.device)
        ones2 = torch.ones((1, x2.shape[0]), device=x1.device)
        c = MatmulLinearOperator(v * ones1, ones2)
        return c + MatmulLinearOperator(x1, -self.scale * x2.T)


def inner_product(x1, x2, metric=None, diag=False):
    if diag:
        min_size = min(x1.shape[0], x2.shape[0])
        if metric is None:
            return (x1[:min_size, :] * x2[:min_size, :]).sum(1)
        else:
            return ((metric @ x1[:min_size, :].T).T * x2[:min_size, :]).sum(1)
    else:
        if metric is None:
            return x1 @ x2.T
        else:
            return x1 @ metric @ x2.T


def diff_inner_prod(
    v1: torch.Tensor, v2: torch.Tensor, diag: bool = False
) -> torch.Tensor:
    """
    Computes the inner product difference matrix
    for two tensors v1 and v2. If `diag=True`, it calculates
    diag((v1 - v2)^T (v1 - v2)).

    This computation is done as v1^2 + v2^2 - 2 v1 v2 for
    tensorized computation.

    Parameters
    ----------
    v1 : torch.Tensor
        A 2D tensor of shape (N, M), where N is the number of rows
        and M is the number of columns.
    v2 : torch.Tensor
        A 2D tensor of shape (N, M), where N is the number of rows
        and M is the number of columns.
    diag : bool
        If True, computes only the diagonal elements of the difference matrix.

    Returns
    -------
    torch.Tensor
        A tensor containing the inner product difference matrix or its diagonal.
    """
    if diag:
        min_size = min(v1.shape[0], v2.shape[0])
        z1 = torch.sum(torch.square(v1[:min_size, :]), axis=1)
        z2 = torch.sum(torch.square(v2[:min_size, :]), axis=1)
        z = (v1[:min_size, :] * v2[:min_size, :]).sum(axis=1)
        res = z1 + z2 - 2 * z
    else:
        z1 = torch.sum(torch.square(v1), axis=1).unsqueeze(1)
        z2 = torch.sum(torch.square(v2), axis=1).unsqueeze(0)
        res = z1 + z2 - 2 * v1 @ v2.T
    return res


def cov2corr(cov):
    v = 1 / torch.sqrt(torch.diag(cov))
    corr = v.unsqueeze(0) * cov * v.unsqueeze(1)
    return corr


def validate_alphabet(
    seq_length: Optional[int] = None,
    alphabet_type: Optional[str] = None,
    alphabet: Optional[List[str]] = None,
    alphabet_list: Optional[List[List[str]]] = None,
) -> List[List]:
    """
    Validate and normalize alphabet inputs for sequence sites.
    This function centralizes validation logic for specifying alphabets used to
    describe per-site character sets for sequences. It accepts either a global
    alphabet (or an alphabet type name that maps to a predefined alphabet) together
    with a sequence length, or a fully-specified per-site alphabet_list. It
    returns a list of per-site alphabets suitable for downstream processing.

    seq_length : Optional[int]
        The number of sites/sequences positions. Required when `alphabet_list` is
        not provided so that a single `alphabet` can be expanded to every site.

    alphabet_type : Optional[str]
        Key name for a predefined alphabet in the module-level ALPHABETS mapping
        (e.g. 'dna', 'rna', 'protein'). If provided, the corresponding alphabet is
        used. When `alphabet_type` is given, `alphabet` must be None.

    alphabet : Optional[List[str]]
        A list of characters specifying the alphabet to apply to all sites. When
        provided with `seq_length`, the result is `[alphabet] * seq_length`. Must
        be None if `alphabet_type` or `alphabet_list` is provided.
    alphabet_list : Optional[List[List[str]]]

        An explicit per-site list of alphabets. When provided, its length must
        already match the intended sequence length and neither `seq_length` nor
        `alphabet`/`alphabet_type` should be specified.

    Returns
    -------
    List[List[str]]
        A normalized per-site list of alphabets (one list of allowed characters
        per site). If a single `alphabet` (or `alphabet_type`) and `seq_length`
        are given, the returned list repeats that alphabet for each site.

    """

    if alphabet_type is not None:
        if alphabet_type not in ALPHABETS:
            msg = f"`alphabet_type` must be one of {list(ALPHABETS.keys())}"
            raise ValueError(msg)
        if alphabet is not None:
            msg = (
                "When `alphabet_type` is provided, `alphabet` should not be specified."
            )
            raise ValueError(msg)
        alphabet = ALPHABETS[alphabet_type]

    if alphabet_list is None:
        if seq_length is None or alphabet is None:
            msg = "When `alphabet_list` is not provided, both `seq_length` and"
            msg += " `alphabet` or `alphabet_type` must be specified."
            raise ValueError(msg)
        alphabet_list = [alphabet] * seq_length
    else:
        msg = "When `alphabet_list` is provided, neither `seq_length` nor `alphabet`"
        msg += " or `alphabet_type` should be specified."
        assert seq_length is None and alphabet is None, msg

    if any(len(site_alphabet) <= 1 for site_alphabet in alphabet_list):
        raise ValueError("All sites must have at least two alleles")

    return alphabet_list


def get_one_hot_encoding(
    seqs: Union[np.ndarray, torch.Tensor], alphabet_list: List[List]
) -> torch.Tensor:
    """
    One-hot encode a list of sequences using per-position alphabets.

    Parameters
    ----------
    seqs : np.ndarray | torch.Tensor
        Iterable of sequences (strings or sequence-like of alleles). All sequences
        must have the same length.
    alphabet_list : List[List]
        Sequence of alphabets for each position. The length of alphabet_list must
        equal the sequence length. The order of alleles in each alphabet determines
        the order of one-hot columns for that position.

    Returns
    -------
    torch.Tensor
        Float tensor of shape (n_sequences, total_features) where
        total_features = sum(len(alphabet) for alphabet in alphabet_list).
        Columns are ordered by increasing position and, within each position,
        by the order of alleles in the corresponding alphabet. Entries are 1.0
        when the sequence has that allele at that position and 0.0 otherwise.

    Raises
    ------
    ValueError
        If `seqs` is empty or if sequence lengths are inconsistent with
        `alphabet_list`.
    """
    if seqs.shape[0] == 0:
        raise ValueError("seqs is empty")

    alleles = np.array([[c for c in s] for s in seqs])
    X = []
    for p, alphabet in enumerate(alphabet_list):
        alleles_p = alleles[:, p]

        unique_alleles = np.unique(alleles[:, p])
        if np.any(~np.isin(unique_alleles, alphabet)):
            msg = f"Unexpected alleles found at position {p}: {unique_alleles}"
            msg += f". Make sure they are chose from alphabet {alphabet}"
            raise ValueError(msg)

        for allele in alphabet:
            X.append(alleles_p == allele)

    X = torch.Tensor(np.vstack(X).T.astype(float))
    return X
