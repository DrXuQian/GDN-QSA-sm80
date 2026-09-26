"""Scalar gate factorization, with explicit vector-gate exclusion. CPU only.

FP64 proves the real-arithmetic transformation, NOT GPU rounding or races.
Actual SASS + device CPU-oracle admission remains required.
"""
import unittest
import torch

torch.set_num_threads(1)


def post_dot(q, k, g, beta, valid, plant=None):
    prefix = g.cumsum(0)
    row = torch.arange(64)[:, None]
    col = torch.arange(64)[None, :]
    difference = prefix[:, None] - prefix[None, :]
    if plant == "wrong-gate-row":
        difference = prefix[None, :] - prefix[:, None]
    decay = difference.exp()
    mask = (row >= col) & (row < valid) & (col < valid)
    if plant == "future-is-live":
        mask = (row < valid) & (col < valid)
    qk = torch.where(mask, (q @ k.T) * decay, 0.)
    factor = beta[None, :] if plant == "beta-column" else beta[:, None]
    kk = torch.where(mask, (k @ k.T) * decay * factor, 0.)
    return qk, kk


def gated_dot(q, k, g, beta, valid):
    # Deliberately distinct ordering: gate operands, then reduce K. Same scalar
    # identity but not the same expression as the implementation under test.
    p = g.cumsum(0)
    out = [torch.zeros(64,64,dtype=torch.float64) for _ in range(2)]
    for row in range(valid):
        for col in range(row + 1):
            gain = (p[row] - p[col]).exp()
            out[0][row,col] = torch.dot(q[row] * gain, k[col])
            out[1][row,col] = torch.dot(k[row] * gain, k[col]) * beta[row]
    return out


class ScalarAux(unittest.TestCase):
    def test_all_tail_sizes_and_gate_regimes(self):
        gen = torch.Generator().manual_seed(919)
        q, k = [torch.randn(64,128,generator=gen,dtype=torch.float64)*.05 for _ in range(2)]
        beta = torch.rand(64,generator=gen,dtype=torch.float64)*.8+.1
        count = 0
        for scale in (0., -.1, -1.):
            g = scale * torch.rand(64,generator=gen,dtype=torch.float64)
            full = gated_dot(q,k,g,beta,64)
            for valid in range(1,65):
                want = [x.clone() for x in full]
                for x in want:
                    x[valid:,:] = 0.; x[:,valid:] = 0.
                for a, b in zip(post_dot(q,k,g,beta,valid), want):
                    torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-13)
                count += 1
        self.assertEqual(count,192)

    def test_three_seam_negatives(self):
        gen = torch.Generator().manual_seed(920)
        q, k = [torch.randn(64,128,generator=gen,dtype=torch.float64)*.05 for _ in range(2)]
        beta = torch.rand(64,generator=gen,dtype=torch.float64)
        g = -torch.rand(64,generator=gen,dtype=torch.float64)*.1
        want = gated_dot(q,k,g,beta,61)
        for plant in ("wrong-gate-row", "beta-column", "future-is-live"):
            with self.subTest(plant=plant), self.assertRaises(AssertionError):
                for a,b in zip(post_dot(q,k,g,beta,61,plant),want):
                    torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-13)


if __name__ == "__main__":
    unittest.main()
