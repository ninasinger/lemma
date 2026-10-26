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
k, h0 = 0.8, 2.0
samples = h0 * (rng.random(400_000) < k) / k
check(abs(samples.mean() - h0) < 0.01 and abs(samples.var() - h0**2 * (1 - k) / k) < 0.02, "dr-3 mean h, variance h^2(1-k)/k")
check(np.isclose(h0**2 * (1 - k) / k, 1.0), "dr-3 variance example = 1.0")
sig = lambda z: 1 / (1 + np.exp(-z))
check(abs(sig(samples).mean() - sig(h0)) > 0.01, "dr-3 E[f(z)] != f(E[z]) for a nonlinearity")

# ---------- Matrices: the silent broadcasting bug ----------
check((np.zeros((100, 1)) - np.zeros(100)).shape == (100, 100), "mx-5 (n,1) - (n,) -> (n,n)")
check((np.zeros((100, 1)).squeeze(1) - np.zeros(100)).shape == (100,), "mx-5 squeeze fixes it")

# ---------- SVD precision fixes ----------
check(np.linalg.matrix_rank(np.diag([3.0, 0.0])) == 1 and np.linalg.matrix_rank(np.zeros((2, 2))) == 0, "svd sigma2=0 gives rank 1 only if sigma1>0")
I2 = np.eye(2)
a, b = np.array([1.0, 0.0]), np.array([1.0, 1.0])
check(np.allclose(I2 @ a, a) and np.allclose(I2 @ b, b) and a @ b != 0, "repeated eigenvalue: eigenvectors need not be orthogonal")
A = rng.normal(size=(30, 20))
U, S, Vt = np.linalg.svd(A, full_matrices=False)
k_ = 5
Ak = U[:, :k_] * S[:k_] @ Vt[:k_]
check(np.isclose(np.linalg.norm(A - Ak, "fro"), np.sqrt((S[k_:] ** 2).sum())), "Eckart-Young Frobenius error")
check(np.isclose(np.linalg.norm(A - Ak, 2), S[k_]), "Eckart-Young spectral error")
other = rng.normal(size=(30, k_)) @ rng.normal(size=(k_, 20))
check(np.linalg.norm(A - other, "fro") > np.linalg.norm(A - Ak, "fro"), "a random rank-k matrix is worse")

# ---------- Ridge: units, sum vs mean, intercepts, conditioning ----------
n, d = 1000, 6
X = rng.normal(size=(n, d)) + 3.0
y = X @ rng.normal(size=d) + 2.0 + rng.normal(size=n)
ridge = lambda X, y, lam: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
lam_b = 0.01
w_mean = np.linalg.solve(X.T @ X / n + lam_b * np.eye(d), X.T @ y / n)
check(np.allclose(ridge(X, y, n * lam_b), w_mean), "l2-3 lambda_sum = n * lambda_mean")
lam = 5.0
w_ridge = ridge(X, y, lam)
grad = 2 * X.T @ (X @ w_ridge - y) + 2 * lam * w_ridge
check(np.allclose(grad, 0, atol=1e-6), "l2-3 closed form zeroes the gradient")
Aaug = np.vstack([X, np.sqrt(lam) * np.eye(d)])
baug = np.concatenate([y, np.zeros(d)])
check(np.allclose(np.linalg.lstsq(Aaug, baug, rcond=None)[0], w_ridge), "l2-3 augmented lstsq = ridge")
Xa = np.column_stack([np.ones(n), X])
P = np.eye(d + 1)
P[0, 0] = 0
w_int = np.linalg.solve(Xa.T @ Xa + lam * P, Xa.T @ y)
mx_, my_ = X.mean(axis=0), y.mean()
w_c = ridge(X - mx_, y - my_, lam)
check(np.allclose(w_int[1:], w_c) and np.isclose(w_int[0], my_ - mx_ @ w_c), "l2-3 unpenalized intercept = centering")

# ---------- Generalization: bias-variance decomposition ----------
f = lambda x: np.sin(np.pi * x)
sigma, N, K, deg, x0 = 0.35, 20, 4000, 3, 0.4
preds, errs = np.empty(K), np.empty(K)
for i in range(K):
    xs = rng.uniform(-1, 1, N)
    w = np.polyfit(xs, f(xs) + rng.normal(0, sigma, N), deg)
    preds[i] = np.polyval(w, x0)
    errs[i] = (f(x0) + rng.normal(0, sigma) - preds[i]) ** 2
decomp = (f(x0) - preds.mean()) ** 2 + preds.var() + sigma**2
check(abs(errs.mean() - decomp) / decomp < 0.06, "gen-1 E[(y - f_hat)^2] = bias^2 + var + sigma^2")
check(np.isclose(0.04 + 0.09 + 0.25, 0.38), "gen-1 example sum")
check(1 - 0.02 == 0.98, "gen-3 majority-class accuracy")

# ---------- Weight decay: five optimizers (same rules as the optim widget) ----------
def run_opt(opt, grad_fn, w0, steps, eta=0.01, lam=0.1, b1=0.9, b2=0.999, eps=1e-8, mu=0.9):
    w = np.array(w0, float)
    m, v, buf = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
    for t in range(1, steps + 1):
        g = grad_fn(w)
        if opt == "sgd":
            w = w - eta * (g + lam * w)
        elif opt == "mom":
            buf = mu * buf + (g + lam * w)
            w = w - eta * buf
        elif opt == "momdec":
            buf = mu * buf + g
            w = w - eta * buf - eta * lam * w
        elif opt == "adaml2":
            gg = g + lam * w
            m, v = b1 * m + (1 - b1) * gg, b2 * v + (1 - b2) * gg**2
            w = w - eta * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        else:  # AdamW, PyTorch order: decay first, then the Adam step on the data gradient
            w = w * (1 - eta * lam)
            m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g**2
            w = w - eta * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return w

one = lambda opt: run_opt(opt, lambda w: np.array([1.0, 0.0]), [1.0, 1.0], 1)
check(np.allclose(one("adaml2"), [0.99, 0.99]), "wd-2 Adam+L2 first step = [0.99, 0.99]")
check(np.allclose(one("adamw"), [0.989, 0.999]), "wd-2 AdamW first step = [0.989, 0.999]")
quiet = {o: run_opt(o, lambda w: np.zeros(1), [1.0], 200)[0] for o in ["sgd", "mom", "momdec", "adaml2", "adamw"]}
check([round(quiet[o], 3) for o in ["sgd", "mom", "momdec", "adaml2", "adamw"]] == [0.819, 0.120, 0.819, 0.016, 0.819], "wd-2 unused weight after 200 steps")
busy = {o: run_opt(o, lambda w: w - 2.0, [1.0], 2000)[0] for o in ["sgd", "mom", "momdec", "adaml2", "adamw"]}
check(all(np.isclose(busy[o], 2 / 1.1, atol=1e-3) for o in ["sgd", "mom", "adaml2"]), "wd-2 L2 rules converge to 2/(1+lambda)")
check(round(busy["momdec"], 3) == 1.980 and np.isclose(busy["momdec"], 2 / (1 + 0.1 * (1 - 0.9)), atol=1e-3), "wd-2 decoupled momentum = L2 with lambda*(1-mu)")
check(round(busy["adamw"], 2) == 1.98, "wd-2 AdamW busy weight ~ 1.98")
check(np.isclose(np.prod(1 - 0.1 * 1e-3 * np.ones(1000)), 0.905, atol=5e-4), "wd-2 constant-lr total decay ~ 0.905")
cos_lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * np.arange(1000) / 1000))
check(np.isclose(np.prod(1 - 0.1 * cos_lr), 0.951, atol=5e-4), "wd-2 cosine total decay ~ 0.951")
check(np.isclose((1 - 0.05 * 1e-3) ** 2000, 0.9048, atol=1e-4), "wd-2 numeric question 0.9048")

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

# ---------- Derivatives ----------
def deriv(f, *args, wrt=0):
    """d f / d args[wrt] by the complex step: exact to rounding for the smooth operations used here."""
    a = [complex(x) for x in args]
    a[wrt] += 1e-20j
    return np.imag(f(*a)) / 1e-20


check(np.isclose(deriv(lambda x: x**2, 3), 6), "dv-1 d(x^2)/dx at 3")
check(np.isclose(3.01**2 - 9, 0.0601) and abs(6 * 0.01 - 0.0601) < 0.002, "dv-1 nudge estimate")
check(np.isclose(deriv(lambda a, b: a * b, 4, 7, wrt=0), 7) and np.isclose(deriv(lambda b, x: b + x, 5.0, 2.0, wrt=1), 1), "dv-2 product and sum rules")
check(np.isclose(deriv(lambda x: (2 * x) ** 2, 3), 24), "dv-3 chain a = 2x, y = a^2 at 3")
check(np.isclose(deriv(lambda x: (3 * x + 1) ** 2, 1), 24), "dv-3 three links")
check(np.isclose(deriv(lambda x: 3 * (2 * x + 5), 4), 6), "dv-4 indirect path")
check(np.isclose(deriv(lambda x: 2 * x + x**2, 3), 8), "dv-5 branches add")
check(np.isclose(deriv(lambda x: (3 * x) * (4 * x), 2), 48), "dv-5 product branches")
g = 5
check(np.isclose(g * deriv(lambda a, b: a * b, 2, 3, wrt=0), 15) and np.isclose(g * deriv(lambda a, b: a * b, 2, 3, wrt=1), 10), "dv-6 incoming x local")
check(np.isclose(3 * deriv(lambda x: x**2, 4), 24) and np.isclose(2.5 * deriv(lambda a: 4 * a, 1.0), 10), "dv-6 drills")
check(np.isclose(deriv(lambda a: (a - 4) ** 2, 6), 4) and np.isclose(deriv(lambda w: (2 * w - 4) ** 2, 3), 8), "dv-7 squared loss")
check(np.isclose(deriv(lambda x: x * x, 3), 6) and np.isclose(deriv(lambda a: a * 5 + a, 2), 6), "dv-8 engine traces")
Wd = np.array([[2, -1, 0], [1, 3, 4]])
check(np.isclose(deriv(lambda *x: (Wd @ np.array(x))[0], 1.0, 1.0, 1.0, wrt=1), -1), "dv-10 Jacobian of Wx is W")
Wv, gv = np.array([[1, 2], [3, 4]]), np.array([1, -1])
check((Wv.T @ gv)[0] == -2 and np.allclose(Wv.T @ gv, [deriv(lambda *x: gv @ (Wv @ np.array(x)), 0.0, 0.0, wrt=j) for j in range(2)]), "dv-10 VJP = W^T g")

# ---------- Chain rule review ----------
check(np.isclose(deriv(lambda x: 4 * (2 * x) ** 2 + 3 * x, 2), 67), "ad-1 worked example: 64 + 3")
check(np.isclose(deriv(lambda x: 4 * (2 * x) ** 2 + 3 * 2, 2), 64) and np.isclose(deriv(lambda a: a**2, 4), 8), "ad-1 indirect path and the 2a edge")
check(np.isclose(deriv(lambda x: 2 * (3 * x) ** 2 + 5 * x, 1), 41) and np.isclose(2 * 67, 134), "ad-4 drills")
check(np.isclose(deriv(lambda x: (3 * (x + 1)) ** 2 + 2 * x, 1), 38), "ad-4 shifted graph")
check(np.isclose(deriv(lambda x: 3 * (2 * x) ** 2 + 4 * x, -1), -20), "ad-4 negative x")
check(np.isclose(-1 * deriv(lambda x: 2 * (2 * x) ** 2 - x, 3), -47), "ad-4 incoming -1")
check(np.isclose(deriv(lambda x: (2 * x + 1) ** 2, 1), 12) and np.isclose(2 * deriv(lambda x: 2 * x + x**2, 3), 16), "ad-3 chain and branch drills")
check(np.isclose(4 * deriv(lambda a, b: a * b, -3, 2, wrt=1), -12) and np.isclose(3 * deriv(lambda x: x**2, -2), -12), "ad-2 drills")

# ---------- Optimization ----------
ws = np.linspace(-10, 10, 200001)
check(np.isclose(ws[np.argmin(2 * (ws - 3) ** 2 + 1)], 3) and np.isclose((( ws + 2) ** 2 + 5).min(), 5), "opt-1 convex minima")
cubic = lambda w: w**3 - 3 * w
check(np.isclose(deriv(cubic, 1), 0) and np.isclose(deriv(cubic, -1), 0) and cubic(1) < cubic(1.1) and cubic(1) < cubic(0.9) and cubic(-1) > cubic(-1.1) and cubic(-1) > cubic(-0.9), "opt-1 w^3 - 3w: min at 1, max at -1")
check(np.isclose(2 - 0.1 * 4, 1.6) and np.isclose(5 - 0.1 * deriv(lambda w: (w - 3) ** 2, 5), 4.6) and np.isclose(-1 - 0.05 * -6, -0.7), "opt-2 gradient-descent steps")
import itertools
gs = [4, -2, 6, 0]
check(np.mean(gs) == 2 and np.mean([gs[0], gs[2]]) == 5 and np.isclose(np.mean([np.mean(b) for b in itertools.combinations(gs, 2)]), np.mean(gs)), "opt-3 mini-batch gradient is unbiased")
check(32000 / 32 == 1000 and len(range(0, 32000, 1024)) == 32 and np.isclose(32000 / 1024, 31.25) and 256 / 32 == 8, "opt-4 updates per epoch")
check(abs(rng.normal(0, 2, size=(40000, 16)).mean(axis=1).std() - 0.5) < 0.02, "opt-4 batch noise shrinks like sigma/sqrt(B)")
check((0**2 - 1**2) == -1 and np.isclose(deriv(lambda x, y: x**2 - y**2, 1, 2, wrt=1), -4) and np.allclose([deriv(lambda x, y: x**2 - y**2, 0.0, 0.0, wrt=i) for i in (0, 1)], 0), "opt-5 saddle")

# ---------- ML CI/CD ----------
half = 1.96 * 100 * np.sqrt(0.9 * 0.1 / 200)
check(abs(half - 4.16) < 0.3 and abs(1.96 * 100 * (rng.binomial(200, 0.9, 400000) / 200).std() - half) < 0.05, "ops-4 interval on 200 examples at 90%")
check(2 < half / 1.5 and 8 > 1.5 * half, "ops-4 a 2-point drop is noise-sized, an 8-point drop is not")
need = next(n for n in range(1, 10000) if 1.96 * np.sqrt(0.9 * 0.1 / n) <= 0.02)
check(need == 865, "ops-4 examples for a +/-2-point interval")

# ---------- Generated questions and cards ----------
# Questions and flashcards with fresh numbers carry a Python twin of each answer. Run a seeded
# sample of every generator through NumPy and compare with the answer the app would accept.
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent.parent
samples = json.loads(subprocess.run(["node", str(root / "scripts/dump_variants.mjs"), "25"], check=True, capture_output=True, text=True).stdout)
for v in samples:
    what = f"{v['where']} (seed {v['seed']}): {v['py']!r}"
    ctx = {"np": np, "rng": np.random.default_rng(v["seed"]), "deriv": deriv}
    try:
        exec(v["py"], ctx)
        got = ctx["ans"]
    except ValueError:
        got = "error"
    want, tol = v["a"], v["tol"]
    if isinstance(want, str):
        expected = "error" if want == "error" else tuple(int(x) for x in re.findall(r"\d+", want))
        check(got == expected if got == "error" or expected == "error" else tuple(got) == expected, f"{what} gave {got}, the app expects {want}")
    else:
        check(np.allclose(got, want, rtol=1e-9, atol=tol + 1e-9), f"{what} gave {got}, the app expects {want}")
variant_note = f"{len(samples)} generated variants included."

print(f"All {checks} NumPy checks passed. {variant_note} {torch_note}")
