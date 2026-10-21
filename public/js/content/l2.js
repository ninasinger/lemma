import { num, list, big, shape, py } from '../vary.js';

// Fresh numbers every time a question or card is shown (see ../vary.js).
const LAMS = [0.01, 0.05, 0.1, 0.5, 1, 2];
const sq = (w) => w.reduce((s, x) => s + x * x, 0);
const gen = {
  penalty(v) {
    const w = [v.nz(-5, 5), v.nz(-5, 5)], lam = v.pick(LAMS), a = +(lam * sq(w)).toFixed(6);
    return { q: `Weights w = (${w.map(num).join(', ')}) and λ = ${num(lam)}. What is the penalty λ‖w‖²?`, a,
      ex: `‖w‖² = ${w.map((x) => num(x * x)).join(' + ')} = ${sq(w)}, and ${num(lam)} × ${sq(w)} = ${num(a)}.`,
      py: `w = np.array(${py(w)})\nans = ${lam} * (w @ w)` };
  },
  shrink(v) {
    const s = v.pick([1, 2, 3, 4, 5, 10]), lam = v.pick([0.5, 1, 2, 4, 9, 16, 25]), f = (s * s) / (s * s + lam);
    return { q: `A direction has singular value σ = ${s}, and λ = ${num(lam)}. What is its shrinkage factor σ² / (σ² + λ)?`, a: f,
      ex: `${s * s} / (${s * s} + ${num(lam)}) = ${num(f, 3)}, so that direction keeps ${num(100 * f, 1)}% of its weight.`,
      py: `U, S, Vt = np.linalg.svd(np.diag([${s}.0, 1.0]))\nans = S[0]**2 / (S[0]**2 + ${lam})` };
  },
  xtr(v) {
    const [n, d] = v.sizes(2, [50, 100, 200, 500, 1000, 3, 5, 8, 10, 20]);
    return { code: `X = np.zeros((${n}, ${d}))\nr = np.zeros(${n})\nX.T @ r`, a: shape(d), ex: `(${d}, ${n}) @ (${n},) → (${d},): one number per weight.`,
      py: `X, r = np.zeros((${n}, ${d})), np.zeros(${n})\nans = (X.T @ r).shape` };
  },
  rr(v) {
    const n = v.pick([50, 100, 200, 500, 1000]), op = v.pick(['@', '*']);
    return { q: `What is the shape of <code>r ${op} r</code>?`, code: `r = np.zeros(${n})\nr ${op} r`, a: op === '@' ? '()' : shape(n),
      ex: op === '@' ? 'r dotted with itself: the sum of squared residuals, a single number.' : `Elementwise: each residual squared, still ${n} numbers. Add <code>.sum()</code> to get the loss.`,
      py: `r = np.zeros(${n})\nans = (r ${op} r).shape` };
  },
  ridgeA(v) {
    const [n, d] = v.sizes(2, [50, 100, 200, 500, 3, 5, 8, 10, 20]), lam = v.pick(LAMS);
    return { code: `X = np.zeros((${n}, ${d}))\nA = X.T @ X + ${lam} * np.eye(${d})`, a: shape(d, d),
      ex: `(${d}, ${n}) @ (${n}, ${d}) → (${d}, ${d}), plus a (${d}, ${d}) identity scaled by λ.`,
      py: `X = np.zeros((${n}, ${d}))\nans = (X.T @ X + ${lam} * np.eye(${d})).shape` };
  },
  cost(v) {
    const n = v.pick([100, 500, 1000, 2000, 10000]), d = v.pick([5, 10, 20, 50, 100]), a = d * n * d;
    return { q: `X has shape (${big(n)}, ${d}). How many multiplications does <code>X.T @ X</code> take?`, a,
      ex: `(${d}, ${big(n)}) @ (${big(n)}, ${d}): m·n·p = ${d} · ${big(n)} · ${d} = ${big(a)}.`, py: `ans = ${d} * ${n} * ${d}` };
  },
  meanSum(v) {
    const n = v.pick([100, 500, 1000, 2000, 10000]), lb = v.pick([0.001, 0.005, 0.01, 0.05, 0.1]), a = +(n * lb).toFixed(6);
    return { q: `Loss A is Σᵢ rᵢ² + λ_A‖w‖² and loss B is (1/n)·Σᵢ rᵢ² + λ_B‖w‖², with n = ${big(n)}. What λ_A gives the same solution as λ_B = ${num(lb)}?`, a,
      ex: `Multiply loss B by n: Σ r² + n·λ_B‖w‖². So λ_A = n·λ_B = ${num(a)}. Whether your library sums or averages the loss changes what a given λ means.`,
      py: `g = np.random.default_rng(1)\nX, y = g.normal(size=(${n}, 3)), g.normal(size=${n})\nwB = np.linalg.solve(X.T @ X / ${n} + ${lb} * np.eye(3), X.T @ y / ${n})\nwA = np.linalg.solve(X.T @ X + ${a} * np.eye(3), X.T @ y)\nassert np.allclose(wA, wB)\nans = ${a}` };
  },

  cLoss(v) {
    const w = [v.nz(-4, 4), v.nz(-4, 4), v.nz(-4, 4)], lam = v.pick(LAMS), a = +(lam * sq(w)).toFixed(6);
    return { front: `Write the loss with L2 regularization. What penalty does w = ${list(w)} pay with λ = ${num(lam)}?`,
      back: `data loss + λ‖w‖², where ‖w‖² = Σⱼ wⱼ². Here ${num(lam)} × ${sq(w)} = <b>${num(a)}</b>.`, a, py: `w = np.array(${py(w)})\nans = ${lam} * (w @ w)` };
  },
  cGrad(v) {
    const w = [v.nz(-4, 4), v.nz(-4, 4)], lam = v.pick([0.1, 0.5, 1, 2]), g = w.map((x) => +(2 * lam * x).toFixed(6));
    return { front: `What is the gradient of λ‖w‖²? Evaluate it at w = ${list(w)} with λ = ${num(lam)}.`,
      back: `2λw = 2 × ${num(lam)} × ${list(w)} = <b>${list(g)}</b>. Written as (λ/2)‖w‖², it’s just λw.`, a: g,
      py: `w, h = np.array(${py(w)}, dtype=float), 1e-6\nf = lambda u: ${lam} * (u @ u)\nans = [(f(w + h * e) - f(w - h * e)) / (2 * h) for e in np.eye(2)]` };
  },
  cShrink(v) {
    const s = v.pick([1, 2, 3, 5, 10]), lam = v.pick([1, 4, 9, 25, 100]), f = (s * s) / (s * s + lam);
    return { front: `How much does ridge shrink the component along a singular direction with σ = ${s} when λ = ${lam}?`,
      back: `By the factor σᵢ² / (σᵢ² + λ) = ${s * s} / (${s * s} + ${lam}) = <b>${num(f, 3)}</b>. Well-determined (large-σ) directions barely change; poorly determined ones get crushed. Large σ means X varies there, not that it predicts y.`, a: f,
      py: `ans = ${s}**2 / (${s}**2 + ${lam})` };
  },
  cShapes(v) {
    const [n, d] = v.sizes(2, [50, 100, 200, 500, 1000, 3, 5, 10, 20]);
    return { front: `Shapes in linear regression with ${n} samples and ${d} features: X, y, w and <code>X @ w</code>?`,
      back: `X (${n}, ${d}), y (${n},), w (${d},), <code>X @ w</code> <b>(${n},)</b>.`, a: shape(n), py: `ans = (np.zeros((${n}, ${d})) @ np.zeros(${d})).shape` };
  },
  cNpGrad(v) {
    const [n, d] = v.sizes(2, [50, 100, 200, 500, 3, 5, 10, 20]);
    return { front: `Gradient of ‖Xw − y‖² + λ‖w‖² in NumPy, and its shape for X of shape (${n}, ${d})?`,
      back: `<code>2 * X.T @ (X @ w - y) + 2 * lam * w</code>, shape <b>(${d},)</b>, same as w.`, a: shape(d),
      py: `X, y, w, lam = np.zeros((${n}, ${d})), np.zeros(${n}), np.zeros(${d}), 0.1\nans = (2 * X.T @ (X @ w - y) + 2 * lam * w).shape` };
  },
  cXt(v) {
    const [n, d] = v.sizes(2, [50, 100, 200, 500, 3, 5, 10, 20]);
    return { front: `Why does <code>X.T</code> appear in the gradient? Follow the shapes for X of shape (${n}, ${d}).`,
      back: `It maps ${n} residuals back to ${d} weights: (${d}, ${n}) @ (${n},) → <b>(${d},)</b>. Entry j sums residual × feature j over all samples.`, a: shape(d),
      py: `ans = (np.zeros((${n}, ${d})).T @ np.zeros(${n})).shape` };
  },
  cMean(v) {
    const n = v.pick([100, 500, 1000, 5000]), lb = v.pick([0.001, 0.01, 0.05, 0.1]), a = +(n * lb).toFixed(6);
    return { front: `Loss Σr² + λ_A‖w‖² vs (1/n)Σr² + λ_B‖w‖² with n = ${big(n)} and λ_B = ${num(lb)}: what λ_A gives the same solution?`,
      back: `λ_A = n·λ_B = <b>${num(a)}</b>. Whether the data term is summed or averaged changes what a given λ means.`, a, py: `ans = ${n} * ${lb}` };
  },
};

export default {id:'l2',title:'L2 regularization',sub:'Charge the model for every large weight.',c:'l2',
 guide:`<h3>The idea</h3><div class="eq">Loss = data loss + λ‖w‖²</div><ul><li>Large weights cost extra, so the model prefers smoother, simpler fits.</li><li>Gradient of the penalty: 2λw (or λw when written as (λ/2)‖w‖²).</li><li>λ too small: overfitting. λ too large: underfitting. Tune it on validation data.</li><li>Shrinks weights toward 0 but rarely to exactly 0 (that is L1).</li><li>The bias term is usually left unpenalized.</li></ul>
 <h3>Ridge regression</h3><div class="eq">w = (XᵀX + λI)⁻¹ Xᵀy</div><ul><li>In the SVD basis of X, each direction is scaled by σᵢ² / (σᵢ² + λ).</li><li>Weakly supported (small σ) directions shrink the most.</li></ul><div class="eq">Ridge(alpha=1.0)</div><p class="note">scikit-learn calls λ “alpha”.</p>
 <h3>Shapes</h3><pre class="py">X (n, d), y (n,), w (d,), X @ w (n,)
grad = 2 * X.T @ (X @ w - y) + 2 * lam * w        # (d,)
w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)</pre>`,
 lessons:[
 {id:'l2-1',title:'Taming wiggles',steps:[
  {t:'learn',title:'Overfitting, live',body:`<p>Twelve noisy points and a degree-11 polynomial. With no penalty the curve threads every point and swings wildly in between: it has memorized the noise.</p><p><b>L2 regularization</b> adds a cost for large weights:</p><div class="eq">Loss = data error + λ · ‖w‖²</div><p>where <span class="m">‖w‖² = w₁² + w₂² + …</span>. Drag λ up and watch the curve and the error chart.</p>`,w:'ridge'},
  {t:'task',title:'Find the sweet spot',body:`<p>The dashed curve is the true function, which you never get in real life. Because we know it here, the widget can show each fit’s <b>true error</b>. Find a λ that gets true error under <b>0.025</b>.</p>`,w:'ridge',wo:{goal:true},hint:`Too small and the curve wiggles; too big and it flattens. Look for the bottom of the true-error valley.`,ex:`Train error only rises as λ grows, but true error dips first. That valley is the bias–variance trade-off.`},
  {t:'mc',q:`As λ → ∞, the weights…`,o:[`shrink toward 0, and the model underfits`,`grow without bound`,`drop to exactly 0 one at a time`,`stay the same`],ex:`With a huge penalty any non-zero weight costs too much, so the fit flattens toward a constant (the bias isn’t penalized).`},
  {t:'mc',q:`λ too small leads to ___; λ too large leads to ___.`,o:[`overfitting; underfitting`,`underfitting; overfitting`,`faster training; slower training`,`underfitting; underfitting`],ex:`Small λ lets the model chase noise; large λ forces it to be too simple. Pick λ with a validation set.`},
  {t:'tf',q:`L2 regularization usually drives many weights to exactly zero.`,a:false,ex:`That is L1 (the lasso). L2 shrinks weights smoothly toward zero but rarely makes them exactly zero.`},
  {t:'num',q:`Weights w = (3, −4) and λ = 0.1. What is the penalty λ‖w‖²?`,a:2.5,tol:.001,ex:`‖w‖² = 9 + 16 = 25, and 0.1 × 25 = 2.5.`,gen:gen.penalty}
 ],
 cards:[
  { id: 'l2-loss', front: `Write the loss with L2 regularization.`, back: `data loss + λ‖w‖², where ‖w‖² = Σⱼ wⱼ².`, gen: gen.cLoss },
  { id: 'l2-lambda', front: `What happens when λ is too small, and too large?`, back: `Too small: overfits, wiggling through the noise. Too large: underfits, with weights shrunk toward 0 and a model that’s too flat.` },
  { id: 'l2-vs-l1', front: `Does L2 regularization make weights exactly zero?`, back: `Rarely. It shrinks them smoothly. L1 (the lasso) is what produces exact zeros.` },
  { id: 'l2-select', front: `How do you choose λ in practice?`, back: `On held-out validation data or by cross-validation. Touch the test set once, at the very end. (The widget’s “true error” is an oracle you never have.)` },
  { id: 'l2-bias', front: `Is the bias usually included in the L2 penalty?`, back: `No. Shifting every prediction up or down doesn’t make the model any wigglier.` },
 ]},
 {id:'l2-2',title:'The geometry of ridge',steps:[
  {t:'learn',title:'Pulled toward the origin',body:`<p>The gray ellipses are contours of the data loss, centered on the unpenalized best fit <span class="m">w*</span>. The dashed circle is a contour of the penalty.</p><p>As λ grows, the solution slides from <span class="m">w*</span> toward 0, always sitting where an ellipse just touches a circle. Watch which direction gives way first.</p>`,w:'contour'},
  {t:'mc',q:`Ridge shrinks the weights most along directions where…`,o:[`the loss is flat, so the data gives little evidence`,`the loss is steep`,`the w₁-axis points`,`the weights are positive`],ex:`Where the loss curves gently, moving the weight barely hurts the fit, so the penalty wins. Steep directions are pinned down by the data and barely move.`},
  {t:'learn',title:'Back to SVD',body:`<p>For linear regression the ridge solution is</p><div class="eq">w = (XᵀX + λI)⁻¹ Xᵀy</div><p>In the SVD basis of <span class="m">X</span>, each direction gets scaled by</p><div class="eq">σᵢ² / (σᵢ² + λ)</div><p>Directions where X varies a lot (large σ) are pinned down well by the data and barely shrink; directions where X barely varies (small σ) are poorly determined and get crushed. Ridge is a soft version of dropping the small principal components. Note that this is about X alone: a direction with lots of variance need not carry any signal about y.</p>`},
  {t:'num',q:`A direction has singular value σ = 3, and λ = 1. What is its shrinkage factor σ² / (σ² + λ)?`,a:.9,tol:.001,ex:`9 / (9 + 1) = 0.9, so that direction keeps 90% of its weight.`,gen:gen.shrink},
  {t:'mc',q:`The gradient of λ‖w‖² with respect to w is…`,o:[`2λw`,`λ`,`λ‖w‖`,`2w`],ex:`The derivative of λw² is 2λw. Many papers write (λ/2)‖w‖² so the gradient comes out as plain λw.`},
  {t:'tf',q:`The bias (intercept) is usually left out of the L2 penalty.`,a:true,ex:`Shifting every prediction up or down doesn’t make the function any wigglier, so penalizing it only adds bias. The polynomial demo left the constant term unpenalized.`}
 ],
 cards:[
  { id: 'l2-grad', front: `What is the gradient of λ‖w‖²?`, back: `2λw. Written as (λ/2)‖w‖², it’s just λw.`, gen: gen.cGrad },
  { id: 'l2-closed', front: `Ridge regression in closed form?`, back: `w = (XᵀX + λI)⁻¹ Xᵀy.` },
  { id: 'l2-shrink', front: `How much does ridge shrink the component along singular direction i?`, back: `By the factor σᵢ² / (σᵢ² + λ). Well-determined (large-σ) directions barely change; poorly determined ones get crushed. Large σ means X varies there, not that it predicts y.`, gen: gen.cShrink },
  { id: 'l2-geo', front: `Which directions does ridge shrink the most?`, back: `Where the loss is flat (small singular values): the data gives little evidence there, so the penalty wins.` },
 ]},
 {id:'l2-3',title:'Ridge in NumPy',steps:[
  {t:'learn',title:'Loss and gradient, with shapes',body:`<pre class="py">X.shape, y.shape            # (n, d), (n,)
w = np.zeros(d)             # (d,)   one weight per feature
pred = X @ w                # (n,)   one prediction per sample
r = pred - y                # (n,)   residuals
loss = r @ r + lam * (w @ w)          # ()     a single number
grad = 2 * X.T @ r + 2 * lam * w      # (d, n) @ (n,) -> (d,)</pre><p>The gradient has one entry per weight, exactly like w. <code>X.T</code> is what carries the n residuals back to the d weights.</p>`},
  {t:'shape',q:'What is the shape of <code>X.T @ r</code>?',code:'X = np.zeros((200, 8))\nr = np.zeros(200)\nX.T @ r',a:'(8,)',ex:'(8, 200) @ (200,) → (8,): one number per weight.',gen:gen.xtr},
  {t:'mc',q:'Ignoring the penalty, what does <code>grad[j]</code> add up?',o:['Over all samples: residual times feature j, times 2','The residual of sample j','Weight j squared','The sum of row j of X'],ex:'grad[j] = 2·Σᵢ X[i, j]·r[i] = 2 × (column j of X) · r. If feature j is big exactly where the errors are big, w[j] gets pushed hardest.'},
  {t:'shape',q:'What is the shape of <code>r @ r</code>?',code:'r = np.zeros(200)\nr @ r',a:'()',ex:'r dotted with itself: the sum of squared residuals, a single number.',gen:gen.rr},
  {t:'learn',title:'The closed form',body:`<pre class="py">A = X.T @ X + lam * np.eye(d)   # (d, d)
b = X.T @ y                     # (d,)
w = np.linalg.solve(A, b)       # (d,): solves A w = b</pre><p><code>np.linalg.solve</code> is faster and more accurate than <code>np.linalg.inv(A) @ b</code>. Adding λ to the diagonal also keeps A invertible when features are collinear.</p>`},
  {t:'shape',q:'What is the shape of A?',code:'X = np.zeros((200, 8))\nA = X.T @ X + 0.1 * np.eye(8)',a:'(8, 8)',ex:'(8, 200) @ (200, 8) → (8, 8), plus an (8, 8) identity scaled by λ.',gen:gen.ridgeA},
  {t:'mc',q:'Why use <code>np.linalg.solve(A, b)</code> instead of <code>np.linalg.inv(A) @ b</code>?',o:['It’s faster and numerically more accurate','<code>inv</code> doesn’t work on square matrices','<code>solve</code> adds the penalty for you','There’s no difference'],ex:'Solving never forms the inverse, which costs more and amplifies rounding error.'},
  {t:'tf',q:'For any λ &gt; 0, XᵀX + λI is invertible, even if two features are identical.',a:true,ex:'XᵀX has eigenvalues ≥ 0. Adding λI lifts every eigenvalue to at least λ &gt; 0, so none is zero.'},
  {t:'num',q:'X has shape (1000, 50). How many multiplications does <code>X.T @ X</code> take?',a:2500000,tol:0,ex:'(50, 1000) @ (1000, 50): m·n·p = 50 · 1000 · 50 = 2,500,000.',gen:gen.cost},
  {t:'mc',q:'Two features: height in meters and income in dollars, not standardized. What does ridge with a single λ do?',o:['Penalizes the height coefficient far more than an equally important income effect','Treats both features equally','Ignores the units','Only affects the intercept'],ex:'A 1-meter change is huge while $1 is tiny, so height needs a large coefficient, and λw² hits it hard. Standardize features so one λ means the same thing for each.'},
  {t:'num',q:'Loss A is Σᵢ rᵢ² + λ_A‖w‖² and loss B is (1/n)·Σᵢ rᵢ² + λ_B‖w‖², with n = 1,000. What λ_A gives the same solution as λ_B = 0.01?',a:10,tol:0,ex:'Multiply loss B by n: Σ r² + n·λ_B‖w‖². So λ_A = n·λ_B = 10. Whether your library sums or averages the loss changes what a given λ means.',gen:gen.meanSum},
  {t:'learn',title:'Intercepts and conditioning',body:`<p>To leave the intercept unpenalized, either center X and y first, or add a column of ones and zero its entry in the penalty:</p>
<pre class="py">Xa = np.column_stack([np.ones(n), X])   # (n, d+1); column 0 is the intercept
P = np.eye(d + 1); P[0, 0] = 0          # no penalty on the intercept
w = np.linalg.solve(Xa.T @ Xa + lam * P, Xa.T @ y)   # (d+1,)</pre><p>Forming XᵀX squares the condition number. When features are nearly collinear, solve the same problem as least squares on an augmented system instead:</p>
<pre class="py">A = np.vstack([X, np.sqrt(lam) * np.eye(d)])   # (n + d, d)
b = np.concatenate([y, np.zeros(d)])           # (n + d,)
w = np.linalg.lstsq(A, b, rcond=None)[0]       # same ridge solution, no X.T @ X</pre>`},
  {t:'explain',kind:'derive',secs:90,q:'Derive the ridge solution from the objective ‖Xw − y‖² + λ‖w‖².',model:`<p>Take the gradient with respect to w and set it to zero:</p><div class="eq">2Xᵀ(Xw − y) + 2λw = 0  ⇒  (XᵀX + λI) w = Xᵀy</div><p>For λ &gt; 0, XᵀX + λI is positive definite (every eigenvalue is at least λ), so the objective is strictly convex and this w is its unique minimizer:</p><div class="eq">w = (XᵀX + λI)⁻¹ Xᵀy</div><p>In code, solve the system rather than forming the inverse. The same objective also comes from a Gaussian prior on w (MAP estimation), which the likelihood unit will cover.</p>`,rubric:['Differentiates both terms correctly: 2Xᵀ(Xw − y) and 2λw','Sets the gradient to zero and rearranges to (XᵀX + λI)w = Xᵀy','Argues uniqueness: XᵀX + λI is positive definite for λ &gt; 0','Notes to solve the system rather than invert']}
 ],
 cards:[
  { id: 'l2-np-shapes', front: `Shapes in linear regression: X, y, w and <code>X @ w</code>?`, back: `X (n, d), y (n,), w (d,), <code>X @ w</code> (n,).`, gen: gen.cShapes },
  { id: 'l2-np-grad', front: `Gradient of ‖Xw − y‖² + λ‖w‖² in NumPy, and its shape?`, back: `<code>2 * X.T @ (X @ w - y) + 2 * lam * w</code>, shape (d,), same as w.`, gen: gen.cNpGrad },
  { id: 'l2-np-solve', front: `The ridge solution in one NumPy call?`, back: `<code>np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)</code>` },
  { id: 'l2-np-xt', front: `Why does <code>X.T</code> appear in the gradient?`, back: `It maps n residuals back to d weights: (d, n) @ (n,) → (d,). Entry j sums residual × feature j over all samples.`, gen: gen.cXt },
  { id: 'l2-np-units', front: `Why standardize features before ridge?`, back: `The penalty acts on coefficients, whose size depends on each feature’s units. Without standardizing, one λ penalizes features unevenly.` },
  { id: 'l2-np-mean', front: `Loss Σr² + λ_A‖w‖² vs (1/n)Σr² + λ_B‖w‖²: how do the λs relate?`, back: `λ_A = n·λ_B. Whether the data term is summed or averaged changes what a given λ means.`, gen: gen.cMean },
  { id: 'l2-np-aug', front: `How do you solve ridge without forming XᵀX?`, back: `Least squares on the augmented system [X; √λ·I] w ≈ [y; 0], e.g. with <code>np.linalg.lstsq</code>.` },
  { id: 'l2-derive-closed', kind: 'derive', secs: 90, front: `Derive the ridge solution from ‖Xw − y‖² + λ‖w‖².`, back: `Gradient 2Xᵀ(Xw − y) + 2λw = 0 gives (XᵀX + λI)w = Xᵀy. XᵀX + λI is positive definite for λ &gt; 0, so the minimizer is unique.`, rubric: ['Correct gradient of both terms', 'Rearranges to (XᵀX + λI)w = Xᵀy', 'Uniqueness from positive definiteness', 'Solve rather than invert'] },
  { id: 'l2-np-solve-why', front: `Why <code>np.linalg.solve(A, b)</code> rather than <code>inv(A) @ b</code>?`, back: `It’s faster and more accurate, and never forms the inverse.` },
 ]}
]};
