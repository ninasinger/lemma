import { num, big, shape, py } from '../vary.js';

// Fresh numbers every time a question or card is shown (see ../vary.js).
const DATA = [100, 200, 500, 1000, 3, 5, 10, 20, 64];
const gen = {
  storage(v) {
    const m = v.pick([64, 80, 100, 120, 128, 200, 256, 300]), n = v.pick([48, 60, 80, 96, 100, 150, 240]), k = v.pick([1, 2, 3, 5, 8, 10, 20]), a = k * (m + n + 1);
    return { q: `A ${m}×${n} image approximated at rank k = ${k} stores <span class="m">k·(m + n + 1)</span> numbers. How many is that?`, a,
      ex: `${k} × (${m} + ${n} + 1) = ${big(a)} numbers, versus ${big(m * n)} for the full image.`, py: `ans = ${k} * (${m} + ${n} + 1)` };
  },
  pcVar(v) {
    const n1 = v.pick([20, 25, 50, 100, 200]), s = v.pick([5, 10, 15, 20, 30, 40]), a = (s * s) / n1;
    return { q: `Centered data has n = ${n1 + 1} samples and σ₁ = ${s}. What is the variance along PC1?`, a,
      ex: `σ₁² / (n − 1) = ${s * s} / ${n1} = ${num(a)}.`, py: `ans = ${s}**2 / (${n1 + 1} - 1)` };
  },
  sShape(v) {
    const [n, d] = v.sizes(2, DATA), k = Math.min(n, d);
    return { code: `Xc = rng.normal(size=(${n}, ${d}))\nU, S, Vt = np.linalg.svd(Xc, full_matrices=False)\nS`, a: shape(k),
      ex: `min(${n}, ${d}) = ${k} singular values, returned as a 1-D array.`,
      py: `U, S, Vt = np.linalg.svd(rng.normal(size=(${n}, ${d})), full_matrices=False)\nans = S.shape` };
  },
  uFull(v) {
    const n = v.pick([50, 100, 200, 500]), d = v.pick([3, 5, 10, 20]);
    return { code: `U, S, Vt = np.linalg.svd(Xc)   # Xc is (${n}, ${d})\nU`, a: shape(n, n),
      ex: `The default <code>full_matrices=True</code> makes U square. Only its first ${d} columns ever meet a singular value; the rest are wasted memory.`,
      py: `U, S, Vt = np.linalg.svd(np.zeros((${n}, ${d})))\nans = U.shape` };
  },
  vtRows(v) {
    const d = v.pick([5, 10, 20, 50, 64]), k = v.int(2, 4);
    return { code: `Vt.shape    # (${d}, ${d})\nk = ${k}\nVt[:k]`, a: shape(k, d),
      ex: `The first ${k} rows: ${k} principal directions, each with a weight for all ${d} features.`, py: `ans = np.zeros((${d}, ${d}))[:${k}].shape` };
  },
  pcVar2(v) {
    const n1 = v.pick([25, 50, 100]), S = v.pick([[20, 10, 5], [30, 20, 10], [40, 15, 5], [50, 25, 10], [10, 5, 2]]), j = v.int(1, 3), a = (S[j - 1] ** 2) / n1;
    return { q: `Centered data has n = ${n1 + 1} samples and <code>S = [${S.join(', ')}]</code>. What is the variance along PC${j}?`, a,
      ex: `σ${'₁₂₃'[j - 1]}² / (n − 1) = ${S[j - 1]}² / ${n1} = ${num(a)}.`, py: `S = np.array(${py(S)})\nans = S[${j - 1}]**2 / (${n1 + 1} - 1)` };
  },

};

export default {id:'svd',title:'SVD',sub:'Every matrix is a rotation, a stretch, and another rotation.',c:'svd',
 guide:`<h3>The decomposition</h3><div class="eq">A = U Σ Vᵀ</div><ul><li>Exists for any m×n matrix.</li><li>U (m×m) and V (n×n) are orthogonal: rotations, possibly with a reflection.</li><li>Σ is diagonal with σ₁ ≥ σ₂ ≥ … ≥ 0.</li><li>Rank = number of non-zero σ’s.</li></ul>
 <h3>Truncation</h3><ul><li>Keep the top k terms, <span class="m">A ≈ U_k Σ_k V_kᵀ</span>: the best rank-k approximation (Eckart–Young).</li><li>Storage drops from m·n to k(m + n + 1).</li></ul>
 <h3>Link to PCA</h3><ul><li>For centered X = UΣVᵀ, the PCs are the columns of V.</li><li>Variance along PCᵢ = σᵢ² / (n−1).</li><li>Projected data (scores) = UΣ.</li></ul><div class="eq">U, S, Vt = np.linalg.svd(X, full_matrices=False)</div>
 <h3>Shapes</h3><pre class="py">U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
# Xc (n, d), n > d  →  U (n, d), S (d,) 1-D and descending, Vt (d, d)
Vt[:k]                      # (k, d): top k principal directions as rows
U[:, :k] * S[:k] @ Vt[:k]   # (n, d): rank-k reconstruction
S**2 / (n - 1)              # (d,): PCA eigenvalues</pre>`,
 lessons:[
 {id:'svd-1',title:'Rotate, stretch, rotate',steps:[
  {t:'learn',title:'Matrices are motions',body:`<p>A 2×2 matrix <span class="m">A</span> moves every point of the plane. The <b>Singular Value Decomposition</b> says every such motion splits into three simple ones:</p><div class="eq">A = U Σ Vᵀ</div><p><span class="m">Vᵀ</span> rotates, <span class="m">Σ</span> stretches along the axes, <span class="m">U</span> rotates again. Press the three steps in order.</p>`,w:'svd'},
  {t:'mc',q:`The circle became an ellipse. Its two semi-axes have lengths…`,o:[`σ₁ and σ₂, the singular values`,`the entries of the matrix`,`the eigenvalues of A`,`always 1 and 2`],ex:`The unit circle always maps to an ellipse whose semi-axes are σ₁ and σ₂. That is the geometric meaning of singular values.`},
  {t:'mc',q:`The singular values in Σ are always…`,o:[`non-negative, sorted largest first`,`negative when A rotates`,`complex numbers`,`equal to the diagonal of A`],ex:`By convention σ₁ ≥ σ₂ ≥ … ≥ 0. Any flip or sign is absorbed into U and V.`},
  {t:'learn',title:'Try other matrices',body:`<p>Switch presets and replay the steps. A <b>rotation</b> keeps the circle a circle, so both σ’s are 1. <b>Rank 1</b> squashes the whole plane onto a line, so σ₂ = 0.</p>`,w:'svd',wo:{preset:3}},
  {t:'mc',q:`If σ₁ &gt; 0 but σ₂ = 0 for a 2×2 matrix, the unit circle is mapped to…`,o:[`a line segment, because the matrix has rank 1`,`a single point`,`a bigger circle`,`a square`],ex:`The number of non-zero singular values is the rank. One non-zero σ means everything lands on a line.`},
  {t:'tf',q:`Every matrix has an SVD, including non-square ones.`,a:true,ex:`Unlike eigendecomposition, the SVD exists for any m×n matrix. That is a big reason it shows up everywhere.`},
  {t:'match',q:`Match each part of A = UΣVᵀ.`,p:[[`Vᵀ`,`Rotates the input`],[`Σ`,`Stretches along the axes`],[`U`,`Rotates the output`],[`Rank`,`Count of non-zero σ’s`]],ex:`Rotate, stretch, rotate.`}
 ],
 cards:[
  { id: 'svd-def', front: `What is the SVD of a matrix A?`, back: `A = U Σ Vᵀ. U and V are orthogonal (rotations, maybe with a flip); Σ is diagonal with σ₁ ≥ σ₂ ≥ … ≥ 0.` },
  { id: 'svd-geo', front: `What does A do to the unit circle, in SVD terms?`, back: `Vᵀ rotates it, Σ stretches the axes by σ₁ and σ₂, U rotates again. The result is an ellipse with semi-axes σ₁ and σ₂.` },
  { id: 'svd-rank', front: `How do you read the rank of A from its SVD?`, back: `Count the non-zero singular values.` },
  { id: 'svd-exists', front: `Which matrices have an SVD?`, back: `Every matrix, of any shape. Eigendecomposition needs a square matrix and doesn’t always exist.` },
 ]},
 {id:'svd-2',title:'Compression & the PCA link',steps:[
  {t:'learn',title:'Keep the big σ’s',body:`<p>An image is a matrix of pixel brightness. The SVD writes it as a sum of rank-1 layers:</p><div class="eq">A = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + σ₃u₃v₃ᵀ + …</div><p>Keeping only the first <b>k</b> layers gives the <b>best possible</b> rank-k approximation, measured by total squared error (the Frobenius norm) and also by the spectral norm. That’s the Eckart–Young theorem. Slide k and watch the face appear.</p>`,w:'img',wo:{k:1}},
  {t:'task',title:'Compress it',body:`<p>Find the <b>smallest k</b> that brings the error under <b>10%</b>.</p>`,w:'img',wo:{k:1,goal:true},hint:`Watch the error readout and stop at the first k where it drops below 10%.`,ex:`Ten rank-1 layers rebuild the face with under 10% error, using 970 numbers instead of 2,304.`},
  {t:'num',q:`A 100×80 image approximated at rank k = 5 stores <span class="m">k·(m + n + 1)</span> numbers. How many is that?`,a:905,tol:0,ex:`5 × (100 + 80 + 1) = 905 numbers, versus 8,000 for the full image.`,gen:gen.storage},
  {t:'learn',title:'PCA is SVD in disguise',body:`<p>Center your data matrix <span class="m">X</span> (n rows, d features) and take its SVD, <span class="m">X = UΣVᵀ</span>. Then:</p><ul><li>The columns of <span class="m">V</span> are the <b>principal directions</b>.</li><li>The variance along PCᵢ is <span class="m">σᵢ² / (n−1)</span>.</li><li>The projected data (the scores) is <span class="m">UΣ</span>.</li></ul><p>Working from the SVD of X avoids forming <span class="m">XᵀX</span>, which squares the condition number and loses precision in the small components. scikit-learn’s PCA uses SVD-based solvers (exact, randomized or iterative, depending on settings); check which one your library picks.</p>`},
  {t:'mc',q:`You run SVD on centered data, X = UΣVᵀ. The principal directions are…`,o:[`the columns of V`,`the columns of U`,`the diagonal of Σ`,`the rows of X`],ex:`V is d×d and lives in feature space, so its columns are directions in feature space: the PCs. U holds per-sample coordinates.`},
  {t:'num',q:`Centered data has n = 101 samples and σ₁ = 20. What is the variance along PC1?`,a:4,tol:.01,ex:`σ₁² / (n − 1) = 400 / 100 = 4.`,gen:gen.pcVar},
  {t:'tf',q:`Keeping the top k singular values gives the best rank-k approximation in total squared error (Frobenius norm).`,a:true,ex:`That is the Eckart–Young theorem. Truncated SVD is optimal, which is why PCA compression works so well.`}
 ],
 cards:[
  { id: 'svd-trunc', front: `What is the best rank-k approximation of A, and in what sense?`, back: `Keep the top k terms: σ₁u₁v₁ᵀ + … + σₖuₖvₖᵀ. It minimizes the error in Frobenius norm (total squared error) and in spectral norm: the Eckart–Young theorem.` },
  { id: 'svd-storage', front: `How many numbers does a rank-k approximation of an m×n matrix need?`, back: `k(m + n + 1): k columns of U, k rows of Vᵀ and k singular values.` },
  { id: 'svd-pca', front: `Centered X = UΣVᵀ. Where are the principal directions and their variances?`, back: `The principal directions are the columns of V (rows of Vᵀ). Variance along PCᵢ = σᵢ² / (n − 1).` },
  { id: 'svd-why', front: `Why compute PCA from the SVD of X instead of the eigenvectors of XᵀX?`, back: `Forming XᵀX squares the condition number, so small components lose precision. The SVD of X avoids that. Which solver a given library uses varies.` },
 ]},
 {id:'svd-3',title:'SVD in NumPy',steps:[
  {t:'learn',title:'Three arrays, three shapes',body:`<pre class="py">Xc.shape                                         # (500, 20)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
U.shape, S.shape, Vt.shape                       # (500, 20), (20,), (20, 20)
np.allclose(U * S @ Vt, Xc)                      # True</pre><p><b>S is 1-D</b> and sorted largest first; it is not a matrix. <code>U * S</code> scales column j of U by S[j] (broadcasting (500, 20) with (20,)). It equals <code>U @ np.diag(S)</code> without building the 20×20 diagonal matrix.</p>`},
  {t:'shape',q:'What is the shape of <code>S</code>?',code:'Xc = rng.normal(size=(500, 20))\nU, S, Vt = np.linalg.svd(Xc, full_matrices=False)\nS',a:'(20,)',ex:'min(500, 20) = 20 singular values, returned as a 1-D array.',gen:gen.sShape},
  {t:'shape',q:'Without <code>full_matrices=False</code>, what is the shape of <code>U</code>?',code:'U, S, Vt = np.linalg.svd(Xc)   # Xc is (500, 20)\nU',a:'(500, 500)',ex:'The default <code>full_matrices=True</code> makes U square. Only its first 20 columns ever meet a singular value; the rest are wasted memory.',gen:gen.uFull},
  {t:'mc',q:'Which array holds the principal directions?',o:['The rows of <code>Vt</code>','The columns of <code>Vt</code>','The columns of <code>U</code>','<code>S</code>'],ex:'<code>Vt</code> is Vᵀ, so its rows are the columns of V: the principal directions, already sorted by S.'},
  {t:'learn',title:'Rank k in one line',body:`<pre class="py">k = 3
Xk = U[:, :k] * S[:k] @ Vt[:k]   # (500, 3) * (3,) -> (500, 3); @ (3, 20) -> (500, 20)
Z = U[:, :k] * S[:k]             # (500, 3): the PCA scores, same as Xc @ Vt[:k].T
var = S**2 / (n - 1)             # (20,): the PCA eigenvalues, largest first</pre><p><code>*</code> and <code>@</code> have the same precedence and run left to right, so the first line is <code>(U[:, :k] * S[:k]) @ Vt[:k]</code>.</p>`},
  {t:'shape',q:'What is the shape of <code>Vt[:k]</code>?',code:'Vt.shape    # (20, 20)\nk = 3\nVt[:k]',a:'(3, 20)',ex:'The first 3 rows: three principal directions, each with a weight for all 20 features.',gen:gen.vtRows},
  {t:'num',q:'Centered data has n = 101 samples and <code>S = [20, 10, 5]</code>. What is the variance along PC2?',a:1,tol:0.001,ex:'σ₂² / (n − 1) = 10² / 100 = 1.',gen:gen.pcVar2},
  {t:'tf',q:'<code>np.linalg.svd</code> returns singular values sorted from largest to smallest.',a:true,ex:'Unlike <code>eigh</code> (ascending), <code>svd</code> is already sorted descending, so no reordering is needed.'},
  {t:'mc',q:'U has shape (500, 20) and S has shape (20,). What does <code>U * S</code> do?',o:['Scales column j of U by S[j]','Scales row j of U by S[j]','Takes a dot product','Raises an error'],ex:'Broadcasting lines up the trailing 20s, so S is reused for every row: entry [i, j] becomes U[i, j]·S[j].'}
 ],
 cards:[
  { id: 'svd-np-shapes', front: `<code>U, S, Vt = np.linalg.svd(X, full_matrices=False)</code> for X of shape (n, d) with n &gt; d. Shapes?`, back: `U (n, d), S (d,), Vt (d, d).` },
  { id: 'svd-np-s1d', front: `Is S from <code>np.linalg.svd</code> a matrix?`, back: `No: a 1-D array of singular values, largest first. <code>np.diag(S)</code> builds the diagonal matrix.` },
  { id: 'svd-np-us', front: `What does <code>U * S</code> compute?`, back: `It scales column j of U by S[j] (broadcasting). Same as <code>U @ np.diag(S)</code>, but cheaper.` },
  { id: 'svd-np-rankk', front: `Rank-k reconstruction in NumPy?`, back: `<code>U[:, :k] * S[:k] @ Vt[:k]</code>: (n, k) @ (k, d) → (n, d).` },
  { id: 'svd-np-full', front: `What does <code>full_matrices=True</code> (the default) change?`, back: `U becomes (n, n): square, with extra columns that never meet a singular value. Use <code>full_matrices=False</code> for data.` },
  { id: 'svd-np-order', front: `Sort order of <code>np.linalg.eigh</code> vs <code>np.linalg.svd</code>?`, back: `eigh: ascending. svd: descending (largest σ first).` },
 ]}
]};
