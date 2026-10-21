"""Runs the NumPy code that the lessons teach and checks every shape, value and error they claim.

    npm run verify:py      (uses .venv; see README)

PyTorch claims in the weight-decay lesson are checked separately only if torch is installed.
"""
import numpy as np

rng = np.random.default_rng(0)
checks = 0


def check(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def raises(fn, what):
    try:
        fn()
    except ValueError:
        check(True, what)
        return
    raise AssertionError(f"expected a ValueError: {what}")


# ---------- Matrices: dot products ----------
check(np.array([2, -1, 3]) @ np.array([4, 0, 1]) == 11, "mx-1 dot = 11")
a, b = np.array([3.0, 1.0]), np.array([1.0, 2.0])
check(a @ b == 5.0 and (a * b).tolist() == [3.0, 2.0], "mx-1 a@b vs a*b")
check(np.isclose(np.sqrt(a @ a), np.linalg.norm(a)), "mx-1 norm")
check((np.ones(5) @ np.ones(5)).shape == (), "mx-1 1-D @ 1-D is 0-d")
check((np.ones(3) * np.ones(3)).shape == (3,), "mx-1 elementwise keeps shape")

# ---------- Matrices: matrix times vector ----------
A = np.array([[1, 2, 0], [3, -1, 4]])
check((A @ np.array([2, 1, 1]))[1] == 9, "mx-2 entry 1 = 9")
check((np.zeros((3, 5)) @ np.zeros(5)).shape == (3,), "mx-2 (3,5)@(5,) -> (3,)")
raises(lambda: np.zeros((3, 5)) @ np.zeros(3), "mx-2 (3,5)@(3,) fails")
A3 = np.arange(9).reshape(3, 3)
check((A3 @ np.array([0, 1, 0])).tolist() == A3[:, 1].tolist(), "mx-2 one-hot picks a column")
Aw, xw = np.array([[2, 1], [-1, 3], [0, 4]]), np.array([3, 1])
check(np.array_equal(Aw @ xw, 3 * Aw[:, 0] + 1 * Aw[:, 1]), "mx-2 row view == column view")

# ---------- Matrices: matrix times matrix ----------
A = np.array([[1, 3, -1], [2, 0, 4]])
B = np.array([[2, 1], [0, -2], [4, 5]])
C = A @ B
check(C.tolist() == [[-2, -10], [20, 22]], "mx-3 widget product")
check(C[1, 0] == 20, "mx-3 C[1,0] = 20")
check((np.zeros((500, 20)) @ np.zeros((20, 3))).shape == (500, 3), "mx-3 (500,20)@(20,3)")
raises(lambda: np.zeros((20, 3)) @ np.zeros((500, 20)), "mx-3 (20,3)@(500,20) fails")
A2, B2 = np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]])
check((A2 @ B2).tolist() == [[1, 0], [0, 0]] and (B2 @ A2).tolist() == [[0, 0], [0, 1]], "mx-3 AB != BA example")
check((np.array([[1, 2], [3, 4]]) @ np.array([[0, 1], [1, 0]]))[0, 0] == 2, "mx-3 (AB)[0,0] = 2")
x = rng.normal(size=4)
M1, M2 = rng.normal(size=(3, 3)), rng.normal(size=(3, 4))
check(np.allclose((M1 @ M2) @ x, M1 @ (M2 @ x)), "mx-3 (AB)x = A(Bx)")

# ---------- Matrices: transpose & the data matrix ----------
X4 = np.array([[2, 1], [-1, -1], [1, 2], [-2, -2]])
check(np.allclose(X4.mean(axis=0), 0), "mx-4 example data is centered")
check((X4.T @ X4).tolist() == [[10, 9], [9, 10]], "mx-4 XtX example")
check(np.allclose(X4.T @ X4 / 3, np.cov(X4, rowvar=False)), "mx-4 XtX/(n-1) is the covariance")
check(np.zeros((500, 20)).T.shape == (20, 500), "mx-4 transpose shape")
check((np.zeros((500, 20)) @ np.zeros((500, 20)).T).shape == (500, 500), "mx-4 X X^T shape")
Xr = rng.normal(size=(30, 5))
check(np.allclose(Xr.T @ Xr, (Xr.T @ Xr).T), "mx-4 XtX symmetric")
check(np.allclose((M1 @ M2).T, M2.T @ M1.T), "mx-4 (AB)^T = B^T A^T")

# ---------- Matrices: NumPy shapes & axes ----------
X = np.arange(12).reshape(3, 4)
check(X.shape == (3, 4) and X[0].shape == (4,) and X[:, 0].shape == (3,) and X[0, 0].shape == (), "mx-5 indexing shapes")
check(X.sum(axis=0).tolist() == [12, 15, 18, 21] and X.sum(axis=1).tolist() == [6, 22, 38], "mx-5 axis sums")
check(X.sum(axis=0, keepdims=True).shape == (1, 4) and X.sum(axis=1, keepdims=True).shape == (3, 1), "mx-5 keepdims")
check(np.sum(X, keepdims=True).shape == (1, 1) and X.sum().shape == (), "mx-5 axis=None")
X = np.zeros((100, 5))
check(X.mean(axis=1).shape == (100,) and X.mean(axis=0).shape == (5,), "mx-5 mean shapes")
check((X - X.mean(axis=0)).shape == (100, 5), "mx-5 broadcast (100,5)-(5,)")
raises(lambda: X - X.mean(axis=1), "mx-5 (100,5)-(100,) fails")
check((X - X.mean(axis=1, keepdims=True)).shape == (100, 5), "mx-5 keepdims broadcast")
check(X[:, :2].shape == (100, 2), "mx-5 slice")

# ---------- PCA in NumPy ----------
X = rng.normal(size=(500, 20)) @ rng.normal(size=(20, 20))
n, d = X.shape
mu = X.mean(axis=0)
Xc = X - mu
Cov = Xc.T @ Xc / (n - 1)
check(mu.shape == (20,) and Xc.shape == (500, 20) and Cov.shape == (20, 20), "pca-3 recipe shapes")
vals, vecs = np.linalg.eigh(Cov)
check(vals.shape == (20,) and vecs.shape == (20, 20), "pca-3 eigh shapes")
check(np.all(np.diff(vals) >= 0), "pca-3 eigh is ascending")
check(np.allclose(Cov @ vecs[:, 3], vals[3] * vecs[:, 3]), "pca-3 columns are eigenvectors")
check(not np.allclose(Cov @ vecs[3], vals[3] * vecs[3]), "pca-3 rows are not eigenvectors")
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
W = vecs[:, :3]
Z = Xc @ W
check(W.shape == (20, 3) and Z.shape == (500, 3), "pca-3 W and Z shapes")
check((Z @ W.T + mu).shape == (500, 20), "pca-3 reconstruction shape")
check(np.isclose(np.array([3, -1, 2]) @ np.array([0.6, 0, 0.8]), 3.4), "pca-3 score = 3.4")
pc1 = np.linalg.eigh(X4.T @ X4)[1][:, -1]
check(np.allclose(np.abs(pc1), [0.7071, 0.7071], atol=1e-3), "pca-3 widget PC1 ~ [0.71, 0.71]")

# ---------- SVD in NumPy ----------
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
check(U.shape == (500, 20) and S.shape == (20,) and Vt.shape == (20, 20), "svd-3 thin shapes")
check(np.linalg.svd(Xc)[0].shape == (500, 500), "svd-3 full_matrices=True U shape")
check(np.all(np.diff(S) <= 0), "svd-3 S is descending")
check(np.allclose(U * S @ Vt, Xc) and np.allclose((U * S) @ Vt, U @ np.diag(S) @ Vt), "svd-3 U*S@Vt and precedence")
k = 3
check((U[:, :k] * S[:k] @ Vt[:k]).shape == (500, 20) and Vt[:k].shape == (3, 20), "svd-3 rank-k shapes")
check(np.allclose(U[:, :k] * S[:k], Xc @ Vt[:k].T), "svd-3 scores two ways")
check(np.allclose(S**2 / (n - 1), vals), "svd-3 S^2/(n-1) = PCA eigenvalues")
check(np.allclose(np.abs(Vt[:k]), np.abs(vecs[:, :k].T)), "svd-3 rows of Vt = principal directions")
check(np.isclose(np.array([20, 10, 5])[1] ** 2 / (101 - 1), 1), "svd-3 PC2 variance = 1")

# ---------- Ridge in NumPy ----------
X = rng.normal(size=(200, 8))
y = rng.normal(size=200)
w = rng.normal(size=8)
lam = 0.1
r = X @ w - y
check(r.shape == (200,) and (X.T @ r).shape == (8,) and (r @ r).shape == (), "l2-3 shapes")
loss = lambda v: (X @ v - y) @ (X @ v - y) + lam * (v @ v)
grad = 2 * X.T @ r + 2 * lam * w
eps = 1e-6
numeric = np.array([(loss(w + eps * e) - loss(w - eps * e)) / (2 * eps) for e in np.eye(8)])
check(np.allclose(grad, numeric, rtol=1e-5, atol=1e-5), "l2-3 gradient formula")
A = X.T @ X + lam * np.eye(8)
check(A.shape == (8, 8), "l2-3 A shape")
check(np.allclose(np.linalg.solve(A, X.T @ y), np.linalg.inv(A) @ X.T @ y), "l2-3 solve == inv")
Xdup = np.column_stack([X[:, 0], X[:, 0], X[:, 1:]])
check(np.linalg.matrix_rank(Xdup.T @ Xdup) < 9, "l2-3 duplicate feature makes XtX singular")
check(np.linalg.matrix_rank(Xdup.T @ Xdup + 1e-3 * np.eye(9)) == 9, "l2-3 adding lambda*I fixes it")
check(50 * 1000 * 50 == 2_500_000, "l2-3 multiplication count")

# ---------- Weight decay (arithmetic only) ----------
check(784 * 128 + 128 == 100_480, "wd-3 parameter count")
check(np.isclose(0.99**100, 0.366, atol=0.001), "wd-1 0.99^100")

# ---------- Dropout in NumPy ----------
h = rng.normal(size=(32, 256))
keep = 0.8
mask = rng.random(h.shape) < keep
check(mask.shape == (32, 256) and (h * mask / keep).shape == (32, 256), "dr-3 mask shapes")
big = rng.random((2000, 256)) < keep
check(abs(big.mean() - keep) < 0.01, "dr-3 fraction kept ~ 0.8")
hs = np.full((20000, 4), 2.0)
check(np.allclose((hs * (rng.random(hs.shape) < keep) / keep).mean(axis=0), 2.0, atol=0.05), "dr-3 expected value preserved")
x = np.zeros((32, 256))
check(((x * (rng.random(x.shape) < 0.9) / 0.9) @ np.zeros((128, 256)).T).shape == (32, 128), "dr-3 dropout then layer")
check(np.isclose(2.0 / 0.8, 2.5), "dr-3 survivor value")

# ---------- PyTorch (optional) ----------
try:
    import torch
    from torch import nn

    layer = nn.Linear(784, 128)
    check(tuple(layer.weight.shape) == (128, 784) and tuple(layer.bias.shape) == (128,), "wd-3 Linear shapes")
    out = layer(torch.zeros(64, 784))
    check(tuple(out.shape) == (64, 128), "wd-3 forward shape")
    out.sum().backward()
    check(layer.weight.grad.shape == layer.weight.shape, "wd-3 grad shape")
    check(sum(p.numel() for p in layer.parameters()) == 100_480, "wd-3 numel")
    torch_note = "PyTorch claims checked too."
except ImportError:
    torch_note = "PyTorch not installed, so its 4 shape claims were not run (the arithmetic was)."

print(f"All {checks} NumPy checks passed. {torch_note}")
