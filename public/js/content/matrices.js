// Unit: matrix foundations. Everything later (PCA, SVD, ridge, layers) is built on these ideas.
// Indices count from 0, like NumPy, so X[i, j] in a lesson is X[i, j] in code.

const features = ['height', 'weight'];

export default {
  id: 'mx',
  title: 'Matrices',
  sub: 'What matrix multiplication really computes, entry by entry.',
  c: 'mx',
  guide: `<h3>Dot product</h3><div class="eq">a·b = a₀b₀ + a₁b₁ + … = ‖a‖ ‖b‖ cos θ</div><ul><li>Positive: same general direction. Zero: perpendicular. Negative: more than 90° apart.</li><li>‖a‖ = √(a·a). Cosine similarity = a·b / (‖a‖‖b‖).</li></ul>
  <h3>Matrix times vector</h3><ul><li>A (m, n) @ x (n,) → (m,).</li><li>Row view: entry i is row i of A dotted with x.</li><li>Column view: a weighted sum of A's columns, weights = entries of x.</li></ul>
  <h3>Matrix times matrix</h3><div class="eq">C[i, j] = Σₖ A[i, k]·B[k, j]</div><ul><li>(m, n) @ (n, p) → (m, p). Inner dimensions match and disappear.</li><li>Not commutative: A B ≠ B A in general.</li><li>(A B) x = A (B x): the matrix next to x acts first.</li><li>Cost: m·n·p multiplications.</li></ul>
  <h3>The data matrix</h3><ul><li>X is (n, d): rows are samples, columns are features. X[i, j] = feature j of sample i.</li><li>XᵀX is (d, d): entry [j, k] = Σᵢ X[i, j]X[i, k]. Centered and divided by n − 1, it's the covariance matrix.</li><li>X Xᵀ is (n, n): dot products between samples.</li><li>(A B)ᵀ = Bᵀ Aᵀ.</li></ul>
  <h3>NumPy</h3>
<pre class="py">X.shape                       # (n, d)
X.mean(axis=0)                # (d,)   one value per column
X.mean(axis=1)                # (n,)   one value per row
X.mean(axis=1, keepdims=True) # (n, 1) broadcasts back against X
a @ b                         # dot product / matrix product
a * b                         # elementwise, never a matrix product</pre>`,
  lessons: [
    {
      id: 'mx-1',
      title: 'Dot products',
      steps: [
        { t: 'learn', title: 'Two lists, one number', body: `<p>A vector is a list of numbers, like <span class="m">a = [3, 1]</span>. The <b>dot product</b> multiplies matching entries and adds them up:</p><div class="eq">a·b = a₀b₀ + a₁b₁ + … + aₙ₋₁bₙ₋₁</div><p>Geometrically it is <span class="m">‖a‖ ‖b‖ cos θ</span>: the two lengths times how much the arrows point the same way. Drag the arrow tips and watch both formulas agree.</p><p>Indices start at 0 here, exactly like NumPy.</p>`, w: 'dot' },
        { t: 'num', q: 'a = [2, −1, 3] and b = [4, 0, 1]. What is a·b?', a: 11, tol: 0, ex: '2·4 + (−1)·0 + 3·1 = 8 + 0 + 3 = 11.' },
        { t: 'task', title: 'Make it zero', body: '<p>Drag the arrows until <b>a·b = 0</b>.</p>', w: 'dot', wo: { a: [3, 1], b: [1, 2] }, hint: 'Make the arrows perpendicular. For a = [3, 1], try b = [−1, 3].', ex: 'A zero dot product means the vectors are <b>orthogonal</b> (perpendicular): neither has any component along the other. PCA’s components are orthogonal in exactly this sense.' },
        { t: 'mc', q: 'a·b is negative. What does that tell you?', o: ['The angle between a and b is more than 90°', 'One of the vectors has only negative entries', 'The vectors have different lengths', 'The vectors are perpendicular'], ex: 'Lengths are never negative, so the sign comes from cos θ, which is negative exactly when θ is over 90°.' },
        { t: 'learn', title: 'Length is a dot product too', body: `<p>Dot a vector with itself: <span class="m">a·a = a₀² + a₁² + …</span>, the squared length. So <span class="m">‖a‖ = √(a·a)</span>.</p><p>Dividing out both lengths leaves <span class="m">cos θ = a·b / (‖a‖ ‖b‖)</span>, the <b>cosine similarity</b> used to compare embeddings.</p>
<pre class="py">a = np.array([3.0, 1.0])   # shape (2,)
b = np.array([1.0, 2.0])   # shape (2,)
a @ b                      # 5.0, same as np.dot(a, b)
a * b                      # array([3., 2.]), elementwise: NOT a dot product
np.sqrt(a @ a)             # 3.162..., same as np.linalg.norm(a)</pre>` },
        { t: 'shape', q: 'What is the shape of <code>a @ b</code>?', code: 'a = np.ones(5)   # shape (5,)\nb = np.ones(5)   # shape (5,)\na @ b', a: '()', ex: 'Two 1-D arrays of the same length give a single number: a 0-d result with shape ().' },
        { t: 'mc', q: 'For two arrays of shape (3,), <code>a * b</code> gives…', o: ['an array of shape (3,), each entry times its partner', 'the dot product, a single number', 'a (3, 3) matrix', 'an error'], ex: '<code>*</code> is always elementwise in NumPy. Use <code>@</code> (or <code>np.dot</code>) for the dot product.' },
      ],
      cards: [
        { id: 'mx-dot-def', front: 'What does the dot product a·b compute, entry by entry?', back: 'Multiply matching entries and add them: a·b = a₀b₀ + a₁b₁ + … . Two vectors of length n give one number.' },
        { id: 'mx-dot-geo', front: 'What is the geometric meaning of a·b?', back: '‖a‖ ‖b‖ cos θ. Positive: same general direction. Zero: perpendicular. Negative: more than 90° apart.' },
        { id: 'mx-dot-norm', front: 'How do you get the length ‖a‖ from a dot product?', back: '‖a‖ = √(a·a). In NumPy: <code>np.sqrt(a @ a)</code> or <code>np.linalg.norm(a)</code>.' },
        { id: 'mx-dot-np', front: 'a and b have shape (3,). What do <code>a @ b</code> and <code>a * b</code> return?', back: '<code>a @ b</code>: the dot product, one number with shape <code>()</code>.<br><code>a * b</code>: elementwise products, shape <code>(3,)</code>.' },
        { id: 'mx-dot-cos', front: 'What is the cosine similarity of a and b?', back: 'cos θ = a·b / (‖a‖ ‖b‖), between −1 and 1. It compares direction only, ignoring length.' },
      ],
    },
    {
      id: 'mx-2',
      title: 'Matrix times vector',
      steps: [
        { t: 'learn', title: 'A stack of dot products', body: `<p>A matrix with m rows and n columns has shape <span class="m">(m, n)</span>. Multiply it by a vector x with n entries and you get m numbers. Entry i is <b>row i of A dotted with x</b>:</p><div class="eq">(A x)[i] = Σₖ A[i, k]·x[k]</div><p>Tap each entry of the result. The widget draws x as a column; in NumPy x is 1-D with shape (2,), and <code>A @ x</code> has shape (3,).</p>`, w: 'matmul', wo: { A: [[2, 1], [-1, 3], [0, 4]], B: [[3], [1]], names: ['A', 'x', 'Ax'], noColumnView: true } },
        { t: 'num', q: 'A = {{1 2 0; 3 −1 4}} and x = {{2; 1; 1}}. What is entry 1 of Ax (the second entry)?', a: 9, tol: 0, ex: 'Row 1 of A is [3, −1, 4]. Dotted with x = [2, 1, 1]: 3·2 + (−1)·1 + 4·1 = 6 − 1 + 4 = 9.' },
        { t: 'learn', title: 'Or: a mix of the columns', body: `<p>Read the same product by columns instead:</p><div class="eq">A x = x[0]·(column 0) + x[1]·(column 1) + …</div><p>The entries of x say <b>how much of each column</b> to use. Switch to <b>Column view</b>. Both views always give the same answer.</p>`, w: 'matmul', wo: { A: [[2, 1], [-1, 3], [0, 4]], B: [[3], [1]], names: ['A', 'x', 'Ax'] } },
        { t: 'mc', q: 'A has shape (3, 5). For <code>A @ x</code> to work, x must have shape…', o: ['(5,)', '(3,)', '(5, 3)', '(3, 3)'], ex: 'Each row of A has 5 entries, so x needs 5 entries to dot with. The vector’s length must equal the number of columns.' },
        { t: 'shape', q: 'What is the shape of <code>A @ x</code>?', code: 'A = np.zeros((3, 5))\nx = np.zeros(5)\nA @ x', a: '(3,)', ex: 'One output per row of A: 3 rows give shape (3,).' },
        { t: 'tf', q: 'If x = [0, 1, 0], then <code>A @ x</code> is exactly column 1 of A.', a: true, ex: 'Column view: 0·col₀ + 1·col₁ + 0·col₂ = col₁. Multiplying by a one-hot vector selects a column, the same trick behind embedding lookups.' },
        { t: 'shape', q: 'Does this work? If so, what shape?', code: 'A = np.zeros((3, 5))\nx = np.zeros(3)\nA @ x', a: 'error', err: true, ex: 'NumPy raises a ValueError: A has 5 columns but x has 3 entries. The inner dimensions must agree.' },
      ],
      cards: [
        { id: 'mx-mv-shape', front: 'A has shape (m, n) and x has shape (n,). What is the shape of <code>A @ x</code>?', back: '<code>(m,)</code>: one number per row of A.' },
        { id: 'mx-mv-row', front: 'Row view of A x: what is entry i?', back: 'Row i of A dotted with x: Σₖ A[i, k]·x[k].' },
        { id: 'mx-mv-col', front: 'Column view of A x?', back: 'A weighted sum of A’s columns: x[0]·col₀ + x[1]·col₁ + … . The entries of x are the weights.' },
        { id: 'mx-mv-onehot', front: 'What is A @ x when x is all zeros except a 1 at index j?', back: 'Exactly column j of A.' },
        { id: 'mx-mv-match', front: 'Which dimensions must match in <code>A @ x</code>?', back: 'The number of columns of A must equal the length of x. Otherwise NumPy raises a ValueError.' },
      ],
    },
    {
      id: 'mx-3',
      title: 'Matrix times matrix',
      steps: [
        { t: 'learn', title: 'Every entry is one dot product', body: `<p>For <span class="m">C = A B</span>, entry C[i, j] is <b>row i of A dotted with column j of B</b>:</p><div class="eq">C[i, j] = Σₖ A[i, k]·B[k, j]</div><p>The index k walks along a row of A and down a column of B at the same time, so those two lengths must match. Tap entries of C; matching colors get multiplied.</p>`, w: 'matmul', wo: { A: [[1, 3, -1], [2, 0, 4]], B: [[2, 1], [0, -2], [4, 5]] } },
        { t: 'task', title: 'Find the right entry', body: '<p>Tap the entry of C that comes from <b>row 1 of A</b> and <b>column 0 of B</b>.</p>', w: 'matmul', wo: { A: [[1, 3, -1], [2, 0, 4]], B: [[2, 1], [0, -2], [4, 5]], target: [1, 0] }, hint: 'Rows of C come from rows of A, and columns of C come from columns of B.', ex: 'C[1, 0] = row 1 of A · column 0 of B = 2·2 + 0·0 + 4·4 = 20.' },
        { t: 'learn', title: 'The shape rule', body: `<div class="eq">(m, n) @ (n, p) → (m, p)</div><p>The inner dimensions (n and n) must match, and they disappear. The outer ones (m and p) become the shape of the result: <b>rows come from A, columns come from B</b>.</p><p>It also tells you the cost: m·p entries, each a dot product of length n, so <span class="m">m·n·p</span> multiplications.</p>` },
        { t: 'shape', q: 'What is the shape of <code>X @ W</code>?', code: 'X = np.zeros((500, 20))\nW = np.zeros((20, 3))\nX @ W', a: '(500, 3)', ex: '(500, 20) @ (20, 3): the 20s match and vanish, leaving (500, 3). You’ll meet exactly this product as the PCA projection.' },
        { t: 'shape', q: 'And this one?', code: 'W = np.zeros((20, 3))\nX = np.zeros((500, 20))\nW @ X', a: 'error', err: true, ex: '(20, 3) @ (500, 20): the inner dimensions 3 and 500 don’t match. Order matters in matrix multiplication.' },
        { t: 'tf', q: 'For square matrices, A B = B A always.', a: false, ex: 'Matrix multiplication is not commutative. With A = {{0 1; 0 0}} and B = {{0 0; 1 0}}: A B = {{1 0; 0 0}} but B A = {{0 0; 0 1}}.' },
        { t: 'num', q: 'A = {{1 2; 3 4}} and B = {{0 1; 1 0}}. What is (A B)[0, 0]?', a: 2, tol: 0, ex: 'Row 0 of A is [1, 2] and column 0 of B is [0, 1]: 1·0 + 2·1 = 2. (This B swaps the columns of A.)' },
        { t: 'learn', title: 'Multiplying means composing', body: `<p><span class="m">(A B) x = A (B x)</span>: first B transforms x, then A transforms the result. A product of matrices is a single transformation that does <b>B first, then A</b>.</p><p>That’s why SVD’s <span class="m">U Σ Vᵀ</span> reads right to left: rotate by Vᵀ, stretch by Σ, rotate by U.</p>` },
        { t: 'mc', q: 'In (A B) x, which matrix acts on x first?', o: ['B', 'A', 'Whichever is larger', 'They act at the same time'], ex: '(A B) x = A (B x). The matrix closest to x acts first.' },
      ],
      cards: [
        { id: 'mx-mm-entry', front: 'What is the formula for entry C[i, j] of C = A B?', back: 'Row i of A dotted with column j of B: C[i, j] = Σₖ A[i, k]·B[k, j].' },
        { id: 'mx-mm-shape', front: '(m, n) @ (n, p) gives what shape?', back: '(m, p). The inner dimensions must match and disappear; the outer ones remain.' },
        { id: 'mx-mm-comm', front: 'Is A B = B A in general?', back: 'No. Matrix multiplication is not commutative, and the shapes may not even allow B A.' },
        { id: 'mx-mm-order', front: 'In A B x, which matrix is applied to x first?', back: 'B. (A B) x = A (B x), so read right to left.' },
        { id: 'mx-mm-cost', front: 'How many multiplications does an (m, n) @ (n, p) product take?', back: 'm·n·p: m·p entries, each a dot product of length n.' },
        { id: 'mx-mm-k', front: 'In Σₖ A[i, k]·B[k, j], what does the shared index k connect?', back: 'Column k of A with row k of B. It walks along row i of A and down column j of B together.' },
      ],
    },
    {
      id: 'mx-4',
      title: 'Transpose & the data matrix',
      steps: [
        { t: 'learn', title: 'Rows are samples, columns are features', body: `<p>Machine-learning data is usually a matrix <span class="m">X</span> with shape <span class="m">(n, d)</span>: n samples (rows) and d features (columns). <b>X[i, j] is feature j of sample i.</b></p><p>The transpose flips rows and columns: <span class="m">Xᵀ</span> has shape (d, n) and <span class="m">Xᵀ[j, i] = X[i, j]</span>. In NumPy it’s <code>X.T</code>.</p>` },
        { t: 'shape', q: 'What is the shape of <code>X.T</code>?', code: 'X = np.zeros((500, 20))\nX.T', a: '(20, 500)', ex: 'Transposing swaps the two axes: (500, 20) becomes (20, 500).' },
        { t: 'learn', title: 'What XᵀX adds up', body: `<p>Four people, two centered features (height and weight, minus their means). <span class="m">XᵀX</span> has shape (d, n) @ (n, d) = <b>(d, d)</b>: one entry per <em>pair of features</em>.</p><div class="eq">(XᵀX)[j, k] = Σᵢ X[i, j]·X[i, k]</div><p>Each entry runs over every sample and adds feature j times feature k. Tap the diagonal and the off-diagonal entries.</p>`, w: 'matmul', wo: { A: [[2, -1, 1, -2], [1, -1, 2, -2]], B: [[2, 1], [-1, -1], [1, 2], [-2, -2]], names: ['Xᵀ', 'X', 'XᵀX'], aRows: features, k: ['s0', 's1', 's2', 's3'], bCols: features, sel: [0, 1], explain: (i, j) => (i === j ? `Diagonal: the sum of squares of ${features[i]} over all 4 samples. Divided by n − 1 = 3 it’s the variance of ${features[i]} (the data is centered).` : `Off-diagonal: ${features[i]} times ${features[j]}, summed over samples. It’s large and positive because they rise and fall together. Divided by n − 1 it’s their covariance.`) } },
        { t: 'mc', q: 'X is centered and has shape (n, d). What is XᵀX / (n − 1)?', o: ['The covariance matrix of the features, shape (d, d)', 'The similarity between samples, shape (n, n)', 'The mean of each feature', 'The identity matrix'], ex: 'Diagonal entries are variances and off-diagonal entries are covariances. PCA takes the eigenvectors of exactly this matrix.' },
        { t: 'shape', q: 'What is the shape of <code>X @ X.T</code>?', code: 'X = np.zeros((500, 20))\nX @ X.T', a: '(500, 500)', ex: '(500, 20) @ (20, 500) → (500, 500): one entry per pair of samples, the dot product of sample i with sample j. That’s a Gram (similarity) matrix.' },
        { t: 'tf', q: 'XᵀX is always symmetric.', a: true, ex: 'Entry [j, k] = Σᵢ X[i, j]X[i, k] is the same sum as entry [k, j]. A symmetric matrix always has an orthonormal basis of eigenvectors, which is why PCA’s components can be taken orthogonal.' },
        { t: 'mc', q: '(A B)ᵀ equals…', o: ['Bᵀ Aᵀ', 'Aᵀ Bᵀ', 'A B', 'B A'], ex: 'Transposing reverses the order. Check with shapes: A B is (m, p), so (A B)ᵀ is (p, m) = (p, n) @ (n, m) = Bᵀ Aᵀ.' },
        { t: 'match', q: 'Match each expression to its meaning (X has shape (n, d)).', p: [['X[i, j]', 'Feature j of sample i'], ['Xᵀ', 'Shape (d, n)'], ['XᵀX', 'Feature-by-feature, (d, d)'], ['X Xᵀ', 'Sample-by-sample, (n, n)']], ex: 'Which side the transpose sits on decides whether you compare features or samples.' },
      ],
      cards: [
        { id: 'mx-x-entry', front: 'In a data matrix X of shape (n, d), what is X[i, j]?', back: 'Feature j of sample i. Rows are samples, columns are features.' },
        { id: 'mx-xtx-entry', front: 'What does entry [j, k] of XᵀX add up?', back: 'Σᵢ X[i, j]·X[i, k]: feature j times feature k, summed over every sample.' },
        { id: 'mx-xtx-cov', front: 'X is centered with shape (n, d). What is XᵀX / (n − 1)?', back: 'The (d, d) covariance matrix: variances on the diagonal, covariances off it.' },
        { id: 'mx-xxt', front: 'Shape and meaning of X Xᵀ, for X of shape (n, d)?', back: '(n, n). Entry [i, j] is the dot product of sample i with sample j: a similarity (Gram) matrix.' },
        { id: 'mx-tr-prod', front: '(A B)ᵀ = ?', back: 'Bᵀ Aᵀ. Transposing reverses the order.' },
        { id: 'mx-xtx-sym', front: 'Why is XᵀX always symmetric?', back: 'Entry [j, k] = Σᵢ X[i, j]X[i, k] is the same sum as entry [k, j].' },
      ],
    },
    {
      id: 'mx-5',
      title: 'NumPy shapes & axes',
      steps: [
        { t: 'learn', title: 'Shapes are tuples', body: `<p>Every NumPy array has a <b>shape</b>: a tuple with one length per axis. A (3, 4) array has axis 0 of length 3 (rows) and axis 1 of length 4 (columns). A 1-D array of 4 numbers has shape <code>(4,)</code>; the comma makes it a tuple. A single number has shape <code>()</code>.</p>
<pre class="py">X = np.arange(12).reshape(3, 4)
X.shape         # (3, 4)
X[0].shape      # (4,)   row 0
X[:, 0].shape   # (3,)   column 0
X[0, 0].shape   # ()     a single number</pre>` },
        { t: 'learn', title: 'Which way does an axis go?', body: '<p>An aggregate like <code>.sum(axis=k)</code> <b>collapses axis k</b>: that axis disappears from the shape. <code>axis=0</code> runs down the rows, so you get one value per column. <code>axis=1</code> runs across, so you get one value per row. Try each one.</p>', w: 'axis', wo: { op: 'sum', axis: 0 } },
        { t: 'task', title: 'One mean per feature', body: '<p>Rows are samples and columns are features. Pick the call that gives the <b>mean of each feature</b>.</p>', w: 'axis', wo: { op: 'sum', axis: null, goal: { op: 'mean', axis: 0 } }, hint: 'You want one value per column, so collapse the rows.', ex: '<code>X.mean(axis=0)</code> collapses the samples and leaves one mean per feature, shape (d,). It’s the first line of PCA.' },
        { t: 'shape', q: 'What is the shape of <code>X.mean(axis=1)</code>?', code: 'X = np.zeros((100, 5))\nX.mean(axis=1)', a: '(100,)', ex: 'Axis 1 (the 5 columns) disappears, leaving one mean per row.' },
        { t: 'learn', title: 'Broadcasting', body: `<p>Arrays with different shapes combine by lining their shapes up <b>from the right</b>. Each pair of lengths must be equal, or one of them must be 1. A missing axis counts as 1.</p>
<pre class="py">X.shape                             # (100, 5)
mu = X.mean(axis=0)                 # (5,)
(X - mu).shape                      # (100, 5): mu is reused for every row
r = X.mean(axis=1)                  # (100,)
X - r                               # error: aligned right, 5 vs 100
r = X.mean(axis=1, keepdims=True)   # (100, 1)
(X - r).shape                       # (100, 5): the 1 stretches across columns</pre>` },
        { t: 'shape', q: 'Does this work? If so, what shape?', code: 'X = np.zeros((100, 5))\nX - X.mean(axis=1)', a: 'error', err: true, ex: 'Aligned from the right, (100, 5) and (100,) compare 5 with 100. No match, so NumPy raises a ValueError. Use <code>keepdims=True</code> to get (100, 1).' },
        { t: 'shape', q: 'And with keepdims?', code: 'X = np.zeros((100, 5))\nX - X.mean(axis=1, keepdims=True)', a: '(100, 5)', ex: '(100, 5) with (100, 1): the 1 stretches across the 5 columns, so each row has its own mean subtracted.' },
        { t: 'shape', q: 'Predictions have shape (n, 1) and labels shape (n,). What is the shape of <code>pred - y</code>?', code: 'pred = np.zeros((100, 1))\ny = np.zeros(100)\npred - y', a: '(100, 100)', ex: 'Aligned from the right, (100, 1) and (100,) become (100, 1) and (1, 100), and both stretch: every prediction minus every label. A loss computed from this is silently wrong. Fix it with <code>pred.squeeze(1)</code> or <code>y[:, None]</code>, and assert shapes before computing a loss.' },
        { t: 'mc', q: 'For two (3, 3) arrays, how do <code>A * B</code> and <code>A @ B</code> differ?', o: ['<code>*</code> multiplies entry by entry; <code>@</code> is matrix multiplication', 'Both are matrix multiplication', '<code>*</code> is matrix multiplication; <code>@</code> is entry by entry', 'Both are entry by entry'], ex: 'Mixing these up gives silently wrong answers when the shapes happen to fit both ways, so it’s worth burning in.' },
        { t: 'shape', q: 'What is the shape of <code>X[:, :2]</code>?', code: 'X = np.zeros((100, 5))\nX[:, :2]', a: '(100, 2)', ex: '<code>:</code> keeps every row and <code>:2</code> keeps columns 0 and 1.' },
      ],
      cards: [
        { id: 'np-axis0', front: '<code>X.sum(axis=0)</code> for X of shape (n, d): shape and meaning?', back: '(d,). Axis 0 (the rows) collapses, leaving one sum per column (feature).' },
        { id: 'np-axis1', front: '<code>X.mean(axis=1)</code> for X of shape (n, d)?', back: '(n,): one mean per row (sample).' },
        { id: 'np-keepdims', front: 'What does <code>keepdims=True</code> do?', back: 'Keeps the collapsed axis with length 1, e.g. (n, 1) instead of (n,), so the result broadcasts back against X.' },
        { id: 'np-bcast', front: 'What is the broadcasting rule?', back: 'Line the shapes up from the right. Each pair of lengths must be equal or contain a 1 (missing axes count as 1). The 1s stretch.' },
        { id: 'np-1d', front: 'What’s the difference between shapes (4,) and (4, 1)?', back: '(4,) is a 1-D array of 4 numbers. (4, 1) is 2-D: 4 rows and 1 column. They broadcast differently.' },
        { id: 'np-bcast-bug', front: 'Why is <code>pred - y</code> dangerous when pred is (n, 1) and y is (n,)?', back: 'It silently broadcasts to (n, n): every prediction minus every label. Squeeze one side or reshape the other, and assert shapes before computing a loss.' },
        { id: 'np-slice', front: 'Shapes of <code>X[0]</code>, <code>X[:, 0]</code> and <code>X[:, :k]</code> for X of shape (n, d)?', back: '<code>X[0]</code>: (d,), row 0.<br><code>X[:, 0]</code>: (n,), column 0.<br><code>X[:, :k]</code>: (n, k).' },
      ],
    },
  ],
};
