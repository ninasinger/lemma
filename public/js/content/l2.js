export default {id:'l2',title:'L2 regularization',sub:'Charge the model for every large weight.',c:'l2',
 guide:`<h3>The idea</h3><div class="eq">Loss = data loss + λ‖w‖²</div><ul><li>Large weights cost extra, so the model prefers smoother, simpler fits.</li><li>Gradient of the penalty: 2λw (or λw when written as (λ/2)‖w‖²).</li><li>λ too small: overfitting. λ too large: underfitting. Tune it on validation data.</li><li>Shrinks weights toward 0 but rarely to exactly 0 (that is L1).</li><li>The bias term is usually left unpenalized.</li></ul>
 <h3>Ridge regression</h3><div class="eq">w = (XᵀX + λI)⁻¹ Xᵀy</div><ul><li>In the SVD basis of X, each direction is scaled by σᵢ² / (σᵢ² + λ).</li><li>Weakly supported (small σ) directions shrink the most.</li></ul><div class="eq">Ridge(alpha=1.0)</div><p class="note">scikit-learn calls λ “alpha”.</p>
 <h3>Shapes</h3><pre class="py">X (n, d), y (n,), w (d,), X @ w (n,)
grad = 2 * X.T @ (X @ w - y) + 2 * lam * w        # (d,)
w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)</pre>`,
 lessons:[
 {id:'l2-1',title:'Taming wiggles',steps:[
  {t:'learn',title:'Overfitting, live',body:`<p>Twelve noisy points and a degree-11 polynomial. With no penalty the curve threads every point and swings wildly in between: it has memorized the noise.</p><p><b>L2 regularization</b> adds a cost for large weights:</p><div class="eq">Loss = data error + λ · ‖w‖²</div><p>where <span class="m">‖w‖² = w₁² + w₂² + …</span>. Drag λ up and watch the curve and the error chart.</p>`,w:'ridge'},
  {t:'task',title:'Find the sweet spot',body:`<p>The dashed curve is the true function, and <b>test error</b> measures how far your fit is from it. Find a λ that gets test error under <b>0.025</b>.</p>`,w:'ridge',wo:{goal:true},hint:`Too small and the curve wiggles; too big and it flattens. Look for the bottom of the test-error valley.`,ex:`Train error only rises as λ grows, but test error dips first. That valley is the bias–variance trade-off.`},
  {t:'mc',q:`As λ → ∞, the weights…`,o:[`shrink toward 0, and the model underfits`,`grow without bound`,`drop to exactly 0 one at a time`,`stay the same`],ex:`With a huge penalty any non-zero weight costs too much, so the fit flattens toward a constant (the bias isn’t penalized).`},
  {t:'mc',q:`λ too small leads to ___; λ too large leads to ___.`,o:[`overfitting; underfitting`,`underfitting; overfitting`,`faster training; slower training`,`underfitting; underfitting`],ex:`Small λ lets the model chase noise; large λ forces it to be too simple. Pick λ with a validation set.`},
  {t:'tf',q:`L2 regularization usually drives many weights to exactly zero.`,a:false,ex:`That is L1 (the lasso). L2 shrinks weights smoothly toward zero but rarely makes them exactly zero.`},
  {t:'num',q:`Weights w = (3, −4) and λ = 0.1. What is the penalty λ‖w‖²?`,a:2.5,tol:.001,ex:`‖w‖² = 9 + 16 = 25, and 0.1 × 25 = 2.5.`}
 ],
 cards:[
  { id: 'l2-loss', front: `Write the loss with L2 regularization.`, back: `data loss + λ‖w‖², where ‖w‖² = Σⱼ wⱼ².` },
  { id: 'l2-lambda', front: `What happens when λ is too small, and too large?`, back: `Too small: overfits, wiggling through the noise. Too large: underfits, with weights shrunk toward 0 and a model that’s too flat.` },
  { id: 'l2-vs-l1', front: `Does L2 regularization make weights exactly zero?`, back: `Rarely. It shrinks them smoothly. L1 (the lasso) is what produces exact zeros.` },
  { id: 'l2-bias', front: `Is the bias usually included in the L2 penalty?`, back: `No. Shifting every prediction up or down doesn’t make the model any wigglier.` },
 ]},
 {id:'l2-2',title:'The geometry of ridge',steps:[
  {t:'learn',title:'Pulled toward the origin',body:`<p>The gray ellipses are contours of the data loss, centered on the unpenalized best fit <span class="m">w*</span>. The dashed circle is a contour of the penalty.</p><p>As λ grows, the solution slides from <span class="m">w*</span> toward 0, always sitting where an ellipse just touches a circle. Watch which direction gives way first.</p>`,w:'contour'},
  {t:'mc',q:`Ridge shrinks the weights most along directions where…`,o:[`the loss is flat, so the data gives little evidence`,`the loss is steep`,`the w₁-axis points`,`the weights are positive`],ex:`Where the loss curves gently, moving the weight barely hurts the fit, so the penalty wins. Steep directions are pinned down by the data and barely move.`},
  {t:'learn',title:'Back to SVD',body:`<p>For linear regression the ridge solution is</p><div class="eq">w = (XᵀX + λI)⁻¹ Xᵀy</div><p>In the SVD basis of <span class="m">X</span>, each direction gets scaled by</p><div class="eq">σᵢ² / (σᵢ² + λ)</div><p>Large-σ directions (strong signal) are almost untouched; small-σ directions (weak, noisy) get crushed. Ridge is a soft version of dropping the small principal components.</p>`},
  {t:'num',q:`A direction has singular value σ = 3, and λ = 1. What is its shrinkage factor σ² / (σ² + λ)?`,a:.9,tol:.001,ex:`9 / (9 + 1) = 0.9, so that direction keeps 90% of its weight.`},
  {t:'mc',q:`The gradient of λ‖w‖² with respect to w is…`,o:[`2λw`,`λ`,`λ‖w‖`,`2w`],ex:`The derivative of λw² is 2λw. Many papers write (λ/2)‖w‖² so the gradient comes out as plain λw.`},
  {t:'tf',q:`The bias (intercept) is usually left out of the L2 penalty.`,a:true,ex:`Shifting every prediction up or down doesn’t make the function any wigglier, so penalizing it only adds bias. The polynomial demo left the constant term unpenalized.`}
 ],
 cards:[
  { id: 'l2-grad', front: `What is the gradient of λ‖w‖²?`, back: `2λw. Written as (λ/2)‖w‖², it’s just λw.` },
  { id: 'l2-closed', front: `Ridge regression in closed form?`, back: `w = (XᵀX + λI)⁻¹ Xᵀy.` },
  { id: 'l2-shrink', front: `How much does ridge shrink the component along singular direction i?`, back: `By the factor σᵢ² / (σᵢ² + λ). Strong directions barely change; weak ones get crushed.` },
  { id: 'l2-geo', front: `Which directions does ridge shrink the most?`, back: `Where the loss is flat (small singular values): the data gives little evidence there, so the penalty wins.` },
 ]},
 {id:'l2-3',title:'Ridge in NumPy',steps:[
  {t:'learn',title:'Loss and gradient, with shapes',body:`<pre class="py">X.shape, y.shape            # (n, d), (n,)
w = np.zeros(d)             # (d,)   one weight per feature
pred = X @ w                # (n,)   one prediction per sample
r = pred - y                # (n,)   residuals
loss = r @ r + lam * (w @ w)          # ()     a single number
grad = 2 * X.T @ r + 2 * lam * w      # (d, n) @ (n,) -> (d,)</pre><p>The gradient has one entry per weight, exactly like w. <code>X.T</code> is what carries the n residuals back to the d weights.</p>`},
  {t:'shape',q:'What is the shape of <code>X.T @ r</code>?',code:'X = np.zeros((200, 8))\nr = np.zeros(200)\nX.T @ r',a:'(8,)',ex:'(8, 200) @ (200,) → (8,): one number per weight.'},
  {t:'mc',q:'Ignoring the penalty, what does <code>grad[j]</code> add up?',o:['Over all samples: residual times feature j, times 2','The residual of sample j','Weight j squared','The sum of row j of X'],ex:'grad[j] = 2·Σᵢ X[i, j]·r[i] = 2 × (column j of X) · r. If feature j is big exactly where the errors are big, w[j] gets pushed hardest.'},
  {t:'shape',q:'What is the shape of <code>r @ r</code>?',code:'r = np.zeros(200)\nr @ r',a:'()',ex:'r dotted with itself: the sum of squared residuals, a single number.'},
  {t:'learn',title:'The closed form',body:`<pre class="py">A = X.T @ X + lam * np.eye(d)   # (d, d)
b = X.T @ y                     # (d,)
w = np.linalg.solve(A, b)       # (d,): solves A w = b</pre><p><code>np.linalg.solve</code> is faster and more accurate than <code>np.linalg.inv(A) @ b</code>. Adding λ to the diagonal also keeps A invertible when features are collinear.</p>`},
  {t:'shape',q:'What is the shape of A?',code:'X = np.zeros((200, 8))\nA = X.T @ X + 0.1 * np.eye(8)',a:'(8, 8)',ex:'(8, 200) @ (200, 8) → (8, 8), plus an (8, 8) identity scaled by λ.'},
  {t:'mc',q:'Why use <code>np.linalg.solve(A, b)</code> instead of <code>np.linalg.inv(A) @ b</code>?',o:['It’s faster and numerically more accurate','<code>inv</code> doesn’t work on square matrices','<code>solve</code> adds the penalty for you','There’s no difference'],ex:'Solving never forms the inverse, which costs more and amplifies rounding error.'},
  {t:'tf',q:'For any λ &gt; 0, XᵀX + λI is invertible, even if two features are identical.',a:true,ex:'XᵀX has eigenvalues ≥ 0. Adding λI lifts every eigenvalue to at least λ &gt; 0, so none is zero.'},
  {t:'num',q:'X has shape (1000, 50). How many multiplications does <code>X.T @ X</code> take?',a:2500000,tol:0,ex:'(50, 1000) @ (1000, 50): m·n·p = 50 · 1000 · 50 = 2,500,000.'}
 ],
 cards:[
  { id: 'l2-np-shapes', front: `Shapes in linear regression: X, y, w and <code>X @ w</code>?`, back: `X (n, d), y (n,), w (d,), <code>X @ w</code> (n,).` },
  { id: 'l2-np-grad', front: `Gradient of ‖Xw − y‖² + λ‖w‖² in NumPy, and its shape?`, back: `<code>2 * X.T @ (X @ w - y) + 2 * lam * w</code>, shape (d,), same as w.` },
  { id: 'l2-np-solve', front: `The ridge solution in one NumPy call?`, back: `<code>np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)</code>` },
  { id: 'l2-np-xt', front: `Why does <code>X.T</code> appear in the gradient?`, back: `It maps n residuals back to d weights: (d, n) @ (n,) → (d,). Entry j sums residual × feature j over all samples.` },
  { id: 'l2-np-solve-why', front: `Why <code>np.linalg.solve(A, b)</code> rather than <code>inv(A) @ b</code>?`, back: `It’s faster and more accurate, and never forms the inverse.` },
 ]}
]};
