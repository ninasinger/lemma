import { num, term, list, shape, dot, dotTerms, py } from '../vary.js';

// Fresh numbers every time a question or card is shown (see ../vary.js).
// Three eigenvalues, largest first, whose total divides 100 so percentages come out whole.
function eigenvalues(v) {
  for (;;) {
    const T = v.pick([10, 20, 25, 50]), l1 = v.int(Math.ceil(T / 3) + 1, T - 3), l2 = v.int(2, T - l1 - 1), l3 = T - l1 - l2;
    if (l1 > l2 && l2 > l3 && l3 > 0) return [l1, l2, l3];
  }
}
const UNIT_DIRS = [[0.6, 0.8], [0.8, 0.6], [0.28, 0.96], [0.96, 0.28]];
const gen = {
  ratio(v) {
    const L = eigenvalues(v), T = L[0] + L[1] + L[2], k = v.int(1, 2), pct = (100 * L[k - 1]) / T;
    return { q: `The covariance eigenvalues are <span class="m">${L.join(', ')}</span>. What percent of the variance does PC${k} explain?`, a: pct,
      ex: `${L[k - 1]} / (${L.join(' + ')}) = ${num(L[k - 1] / T)}, so ${num(pct)}%.`, py: `vals = np.array(${py(L)})\nans = 100 * vals[${k - 1}] / vals.sum()` };
  },
  cov(v) {
    const [n, d] = v.sizes(2, [50, 100, 200, 500, 1000, 3, 5, 10, 20, 64]);
    return { code: `X = rng.normal(size=(${n}, ${d}))\nXc = X - X.mean(axis=0)\nC = Xc.T @ Xc / (${n} - 1)`, a: shape(d, d),
      ex: `(${d}, ${n}) @ (${n}, ${d}) → (${d}, ${d}): one entry per pair of features.`,
      py: `X = rng.normal(size=(${n}, ${d}))\nXc = X - X.mean(axis=0)\nans = (Xc.T @ Xc / (${n} - 1)).shape` };
  },
  scores(v) {
    const [n, d] = v.sizes(2, [100, 200, 500, 1000, 20, 50, 64]), k = v.int(2, Math.min(5, d - 1));
    return { code: `Xc.shape        # (${n}, ${d})\nW = vecs[:, :${k}]\nZ = Xc @ W`, a: shape(n, k),
      ex: `(${n}, ${d}) @ (${d}, ${k}) → (${n}, ${k}): ${k} numbers per sample instead of ${d}.`,
      py: `Xc = np.zeros((${n}, ${d}))\nvecs = np.zeros((${d}, ${d}))\nans = (Xc @ vecs[:, :${k}]).shape` };
  },
  score(v) {
    const [p, q] = v.pick(UNIT_DIRS), zero = v.int(0, 2), w = [0, 0, 0], others = [0, 1, 2].filter((i) => i !== zero);
    w[others[0]] = p * v.sign(); w[others[1]] = q * v.sign();
    const x = v.vec(3, -4, 5), s = dot(x, w);
    return { q: `A centered sample is <span class="m">${list(x)}</span> and a unit direction is <span class="m">w = ${list(w)}</span>. What is the sample’s score along w?`, a: +s.toFixed(6),
      ex: `${dotTerms(x, w)} = ${num(s)}.`, py: `ans = np.array(${py(x)}) @ np.array(${py(w)})` };
  },
  recon(v) {
    const [n, d] = v.sizes(2, [100, 200, 500, 1000, 20, 50, 64]), k = v.int(2, Math.min(5, d - 1));
    return { code: `Z.shape     # (${n}, ${k})\nW.shape     # (${d}, ${k})\nmu.shape    # (${d},)\nX_hat = Z @ W.T + mu`, a: shape(n, d),
      ex: `(${n}, ${k}) @ (${k}, ${d}) → (${n}, ${d}), and mu broadcasts across the rows. You’re back in the original ${d} features, minus what the dropped components carried.`,
      py: `Z, W, mu = np.zeros((${n}, ${k})), np.zeros((${d}, ${k})), np.zeros(${d})\nans = (Z @ W.T + mu).shape` };
  },

  cRatio(v) {
    const L = eigenvalues(v), T = L[0] + L[1] + L[2], k = v.int(1, 3);
    return { front: `What is the explained-variance ratio of PCᵢ? Find it for PC${k} when the eigenvalues are ${L.join(', ')}.`,
      back: `λᵢ / (λ₁ + λ₂ + … + λ_d). Here ${L[k - 1]} / ${T} = <b>${num(L[k - 1] / T)}</b>.`, a: L[k - 1] / T,
      py: `vals = np.array(${py(L)})\nans = vals[${k - 1}] / vals.sum()` };
  },
  cCov(v) {
    const [n, d] = v.sizes(2);
    return { front: `X has shape (${n}, ${d}). What is the shape of <code>Xc.T @ Xc / (${n} - 1)</code>?`, back: `<b>(${d}, ${d})</b>: the covariance matrix, one entry per pair of features.`,
      a: shape(d, d), py: `Xc = np.zeros((${n}, ${d}))\nans = (Xc.T @ Xc / (${n} - 1)).shape` };
  },
  cEigh(v) {
    const d = v.int(3, 8), k = v.int(1, 2);
    return { front: `After <code>vals, vecs = np.linalg.eigh(C)</code> for a (${d}, ${d}) covariance matrix C, which column of <code>vecs</code> is PC${k}?`,
      back: `<b><code>vecs[:, ${d - k}]</code></b>. Eigenvectors are columns, and eigenvalues come back in <b>ascending</b> order, so PC1 is the last column. Reverse them to put PC1 first.`, a: d - k,
      py: `vals, vecs = np.linalg.eigh(np.diag(np.arange(1.0, ${d + 1})))\nans = int(np.argsort(vals)[::-1][${k - 1}])` };
  },
  cProj(v) {
    const [n, d] = v.sizes(2, [100, 200, 500, 1000, 20, 50, 64]), k = v.int(2, Math.min(5, d - 1));
    return { front: `Shapes in <code>Z = Xc @ W</code> with ${k} components, for ${n} samples with ${d} features?`, back: `Xc (${n}, ${d}) @ W (${d}, ${k}) → Z <b>(${n}, ${k})</b>.`,
      a: shape(n, k), py: `ans = (np.zeros((${n}, ${d})) @ np.zeros((${d}, ${k}))).shape` };
  },
  cZij(v) {
    const i = v.int(0, 99), j = v.int(0, 2);
    return { front: `In PCA, what does Z[${i}, ${j}] mean?`, back: `The coordinate of sample ${i} along principal direction ${j} (PC${j + 1}): centered sample ${i} dotted with column ${j} of W.` };
  },
  cRecon(v) {
    const [n, d] = v.sizes(2, [100, 200, 500, 1000, 20, 50, 64]), k = v.int(2, Math.min(5, d - 1));
    return { front: `How do you reconstruct X from ${k} components, for ${n} samples with ${d} features?`,
      back: `<code>Z @ W.T + mu</code>: (${n}, ${k}) @ (${k}, ${d}) → <b>(${n}, ${d})</b>, then add the mean back.`, a: shape(n, d),
      py: `ans = (np.zeros((${n}, ${k})) @ np.zeros((${d}, ${k})).T + np.zeros(${d})).shape` };
  },
};

export default {id:'pca',title:'PCA',sub:'Find the directions where your data actually varies.',c:'pca',
 guide:`<h3>The recipe</h3><ol><li>Center each feature (subtract its mean). Standardize too if features use different units.</li><li>Compute the covariance matrix <span class="m">C = XᵀX / (n−1)</span>.</li><li>Its eigenvectors are the principal components, sorted by eigenvalue.</li><li>Project onto the top k to get a k-dimensional version of the data.</li></ol>
 <h3>Key facts</h3><ul><li>Eigenvalue λᵢ is the variance along PCᵢ. Explained-variance ratio = λᵢ / Σλ.</li><li>PCs are orthogonal, and the projected features are uncorrelated.</li><li>Maximizing captured variance is the same as minimizing reconstruction error.</li><li>PCA is linear and unsupervised: it never looks at labels.</li></ul>
 <h3>In code</h3><div class="eq">PCA(n_components=0.90).fit_transform(X)</div><p class="note">In scikit-learn, a fraction keeps enough components to explain that share of variance.</p>
 <h3>Shapes</h3><pre class="py">X (n, d) → mu = X.mean(axis=0) (d,) → Xc (n, d) → C (d, d)
vals, vecs = np.linalg.eigh(C)   # (d,), (d, d), ascending; column i = eigenvector i
W = vecs[:, ::-1][:, :k]         # (d, k), top k directions
Z = Xc @ W                       # (n, k)
X_hat = Z @ W.T + mu             # (n, d)</pre>`,
 lessons:[
 {id:'pca-1',title:'Directions of spread',steps:[
  {t:'learn',title:'Data hides in fewer directions',body:`<p>Two columns, <em>height in cm</em> and <em>height in inches</em>, really carry one number. Most real datasets are like that, only less obviously: the points spread mostly along a few directions.</p><p><b>Principal Component Analysis</b> (PCA) finds those directions and ranks them by how much the data spreads along each one.</p>`},
  {t:'learn',title:'Spin the line',body:`<p>Every dot is projected onto the line (the small colored dots). <b>Captured variance</b> measures how spread out those projections are. Drag anywhere on the plot or use the slider.</p>`,w:'pca',wo:{start:150}},
  {t:'task',title:'Find PC1',body:`<p>Rotate the line until it captures as much variance as it possibly can.</p>`,w:'pca',wo:{start:120},hint:`Keep turning. The captured bar can still grow: aim along the direction the cloud is stretched.`,ex:`That direction is the <b>first principal component</b> (PC1): the one line that keeps the most spread.`},
  {t:'mc',q:`PC1 is the direction that…`,o:[`maximizes the variance of the projected data`,`minimizes the variance of the projected data`,`is always parallel to the x-axis`,`connects the two most distant points`],ex:`PC1 maximizes projected variance. It depends on the whole cloud, not on an axis or on any single pair of points.`},
  {t:'learn',title:'Same line, two stories',body:`<p>The gray segments now show what each projection <em>throws away</em>. For every centered point, Pythagoras gives <span class="m">‖x‖² = kept² + thrown away²</span>.</p><p>So maximizing captured variance is the same as <b>minimizing the leftover (reconstruction) error</b>. Rotate and watch: the two bars always add up to 100%.</p>`,w:'pca',wo:{start:100,residuals:true,reveal:true}},
  {t:'tf',q:`Before running PCA, you should center the data by subtracting each feature’s mean.`,a:true,ex:`PCA measures spread around the mean. Without centering, the first component tends to point at the mean instead of along the spread.`},
  {t:'mc',q:`Two features: <b>income</b> in dollars (20,000–200,000) and <b>age</b> in years (18–90). Without any scaling, PC1 will point mostly along…`,o:[`income, because its numbers vary far more`,`age, because it is more informative`,`a 45° blend of the two`,`neither, since PCA ignores units`],ex:`Variance depends on units. Dollars vary by tens of thousands, years by tens, so income dominates. Standardize (z-score) features when their units differ.`}
 ],
 cards:[
  { id: 'pca-def', front: `Which direction is PC1?`, back: `The direction along which the centered data spreads the most: the projections onto it have the largest variance.` },
  { id: 'pca-recon', front: `Why does maximizing captured variance also minimize reconstruction error?`, back: `For each centered point, ‖x‖² = (kept)² + (thrown away)². The total is fixed, so more captured variance means less leftover error.` },
  { id: 'pca-center', front: `Why center the data before PCA?`, back: `PCA measures spread around the mean. Without centering, PC1 tends to point at the mean instead of along the spread.` },
  { id: 'pca-scale', front: `When should you standardize features before PCA?`, back: `When they use different units or scales. Otherwise the feature with the biggest numbers dominates PC1.` },
 ]},
 {id:'pca-2',title:'Components & compression',steps:[
  {t:'learn',title:'More components',body:`<p><b>PC2</b> is the direction of greatest remaining variance that is <b>perpendicular</b> to PC1. PC3 is perpendicular to both, and so on.</p><p>Formally, the PCs are the <b>eigenvectors</b> of the covariance matrix <span class="m">C = XᵀX / (n−1)</span>, and each <b>eigenvalue</b> λᵢ is the variance along PCᵢ.</p><p>The share of variance that PCᵢ explains is <span class="m">λᵢ / (λ₁ + λ₂ + …)</span>.</p>`},
  {t:'tf',q:`Principal components are always orthogonal (perpendicular) to each other.`,a:true,ex:`The covariance matrix is symmetric, so its eigenvectors can be chosen orthonormal. Eigenvectors for different eigenvalues are automatically orthogonal; for a repeated eigenvalue you pick an orthogonal basis of its eigenspace. That also makes the projected features uncorrelated.`},
  {t:'num',q:`The covariance eigenvalues are <span class="m">6, 3, 1</span>. What percent of the variance does PC1 explain?`,a:60,tol:.5,suf:'%',ex:`6 / (6 + 3 + 1) = 0.6, so 60%.`,gen:gen.ratio},
  {t:'task',title:'How many to keep?',body:`<p>These are the eigenvalues of a 6-feature dataset, drawn as a <b>scree plot</b>. Tap a bar to keep that many components. Find the <b>smallest</b> k that keeps at least 90% of the variance.</p>`,w:'scree',hint:`Add components until the total crosses the 90% tick, then stop.`,ex:`k = 3 keeps 91.1%. You squeezed 6 numbers per row into 3 and kept almost all of the structure.`},
  {t:'match',q:`Match each piece of PCA to what it means.`,p:[[`Eigenvector of C`,`A principal direction`],[`Eigenvalue of C`,`Variance along that direction`],[`Top-k projection`,`Compressed representation`],[`Centering`,`Subtracting the mean`]],ex:`That is the whole PCA vocabulary.`},
  {t:'mc',q:`Which of these is a real limitation of PCA?`,o:[`It can only find linear (straight-line) structure`,`It needs labeled data`,`It always increases the number of features`,`It only works with 2 or 3 features`],ex:`PCA is unsupervised and works in any dimension, but it only finds straight directions. Curved structure needs tools like kernel PCA, t-SNE or UMAP.`}
 ],
 cards:[
  { id: 'pca-eig', front: `PCA takes the eigenvectors of which matrix?`, back: `The covariance matrix C = XcᵀXc / (n − 1). Eigenvectors are the principal directions; eigenvalue λᵢ is the variance along PCᵢ.` },
  { id: 'pca-ratio', front: `What is the explained-variance ratio of PCᵢ?`, back: `λᵢ / (λ₁ + λ₂ + … + λ_d).`, gen: gen.cRatio },
  { id: 'pca-orth', front: `Why are principal components orthogonal?`, back: `The covariance matrix is symmetric, and symmetric matrices have an orthonormal basis of eigenvectors (automatic for distinct eigenvalues, a choice within a repeated one). So the projected features are uncorrelated.` },
  { id: 'pca-limit', front: `What is PCA’s main limitation?`, back: `It only finds linear (straight-line) structure. Curved structure needs kernel PCA, t-SNE or UMAP.` },
 ]},
 {id:'pca-3',title:'PCA in NumPy',steps:[
  {t:'learn',title:'The whole recipe, with shapes',body:`<p>Here is PCA on 500 samples with 20 features. Read the shape comments: they are the whole story.</p>
<pre class="py">n, d = X.shape                    # (500, 20)
mu = X.mean(axis=0)               # (20,)      one mean per feature
Xc = X - mu                       # (500, 20)  broadcasting
C = Xc.T @ Xc / (n - 1)           # (20, 20)   covariance
vals, vecs = np.linalg.eigh(C)    # (20,), (20, 20), ascending order!
order = np.argsort(vals)[::-1]    # biggest first
vals, vecs = vals[order], vecs[:, order]
W = vecs[:, :3]                   # (20, 3)    top 3 directions, as columns
Z = Xc @ W                        # (500, 3)   the projected data (scores)
X_hat = Z @ W.T + mu              # (500, 20)  reconstruction</pre>`},
  {t:'shape',q:'What is the shape of <code>C</code>?',code:'X = rng.normal(size=(500, 20))\nXc = X - X.mean(axis=0)\nC = Xc.T @ Xc / (500 - 1)',a:'(20, 20)',ex:'(20, 500) @ (500, 20) → (20, 20): one entry per pair of features.',gen:gen.cov},
  {t:'mc',q:'After <code>vals, vecs = np.linalg.eigh(C)</code>, where is eigenvector i?',o:['<code>vecs[:, i]</code>, column i','<code>vecs[i]</code>, row i','<code>vals[i]</code>','<code>vecs[i, i]</code>'],ex:'NumPy returns eigenvectors as columns. <code>vecs[i]</code> is row i, which mixes one piece of every eigenvector.'},
  {t:'mc',q:'Why reverse the order after <code>eigh</code>?',o:['<code>eigh</code> returns eigenvalues in ascending order, and PC1 has the largest','<code>eigh</code> returns them in random order','To make the eigenvalues positive','It isn’t needed'],ex:'<code>np.linalg.eigh</code> sorts ascending, so PC1 is the last column. Sorting descending puts it first.'},
  {t:'shape',q:'What is the shape of <code>Z</code>?',code:'Xc.shape        # (500, 20)\nW = vecs[:, :3]\nZ = Xc @ W',a:'(500, 3)',ex:'(500, 20) @ (20, 3) → (500, 3): 3 numbers per sample instead of 20.',gen:gen.scores},
  {t:'learn',title:'What one entry of Z means',body:`<p><span class="m">Z[i, j]</span> = row i of Xc · column j of W: the <b>coordinate of sample i along principal direction j</b>. These are the 4 centered samples from the Matrices unit, projected onto their PC1 ≈ [0.71, 0.71]. Tap the entries, then try Column view.</p>`,w:'matmul',wo:{A:[[2,1],[-1,-1],[1,2],[-2,-2]],B:[[0.71],[0.71]],names:['Xc','W','Z'],aRows:['s0','s1','s2','s3'],k:['height','weight'],bCols:['PC1'],explain:(i)=>`Z[${i}, 0] is how far sample ${i} lies along PC1: its dot product with the unit direction [0.71, 0.71].`}},
  {t:'num',q:'A centered sample is <span class="m">[3, −1, 2]</span> and a unit direction is <span class="m">w = [0.6, 0, 0.8]</span>. What is the sample’s score along w?',a:3.4,tol:0.001,ex:'3·0.6 + (−1)·0 + 2·0.8 = 1.8 + 1.6 = 3.4.',gen:gen.score},
  {t:'shape',q:'What is the shape of <code>X_hat</code>?',code:'Z.shape     # (500, 3)\nW.shape     # (20, 3)\nmu.shape    # (20,)\nX_hat = Z @ W.T + mu',a:'(500, 20)',ex:'(500, 3) @ (3, 20) → (500, 20), and mu broadcasts across the rows. You’re back in the original 20 features, minus what the dropped components carried.',gen:gen.recon},
  {t:'mc',q:'In code, what fraction of variance do the top k components keep?',o:['<code>vals[:k].sum() / vals.sum()</code>','<code>vals[:k] / k</code>','<code>vecs[:, :k].sum()</code>','<code>W.sum() / d</code>'],ex:'Each eigenvalue is the variance along its component, so the kept share is the sum of the top k over the total.'}
 ],
 cards:[
  { id: 'pca-np-cov', front: `X has shape (n, d). What is the shape of <code>Xc.T @ Xc / (n - 1)</code>?`, back: `(d, d): the covariance matrix, one entry per pair of features.`, gen: gen.cCov },
  { id: 'pca-np-eigh', front: `After <code>vals, vecs = np.linalg.eigh(C)</code>, where is eigenvector i, and in what order?`, back: `Column <code>vecs[:, i]</code>. Eigenvalues come back in <b>ascending</b> order, so reverse them to put PC1 first.`, gen: gen.cEigh },
  { id: 'pca-np-proj', front: `Shapes in <code>Z = Xc @ W</code> with k components?`, back: `Xc (n, d) @ W (d, k) → Z (n, k).`, gen: gen.cProj },
  { id: 'pca-np-zij', front: `In PCA, what does Z[i, j] mean?`, back: `The coordinate of sample i along principal direction j: centered sample i dotted with PC j.`, gen: gen.cZij },
  { id: 'pca-np-recon', front: `How do you reconstruct X from k components?`, back: `<code>Z @ W.T + mu</code>: (n, k) @ (k, d) → (n, d), then add the mean back.`, gen: gen.cRecon },
 ]}
]};
