export default {id:'wd',title:'Weight decay',sub:'Shrink every weight a little, every single step.',c:'wd',
 guide:`<div class="eq">w ← (1 − ηλ)·w − η·∇L</div><ul><li>Every step, each weight loses the fraction ηλ of itself.</li><li>With zero gradient: wₜ = (1 − ηλ)ᵗ·w₀, exponential decay.</li><li>Plain SGD: identical to L2 with penalty (λ/2)‖w‖².</li><li>Adam: an L2 term gets divided by √v, so the decay becomes uneven. AdamW applies decay separately.</li><li>Common practice: decay the weight matrices, skip biases and normalization parameters.</li></ul><div class="eq">torch.optim.AdamW(params, lr=3e-4, weight_decay=0.1)</div>
 <h3>Shapes</h3><pre class="py">nn.Linear(in, out).weight   # (out, in)
nn.Linear(in, out).bias     # (out,)
layer(x)                    # x @ W.T + b: (batch, in) -> (batch, out)
p.grad.shape == p.shape     # always
decay if p.ndim >= 2, else weight_decay = 0</pre>`,
 lessons:[
 {id:'wd-1',title:'Shrink every step',steps:[
  {t:'learn',title:'A slow leak in every weight',body:`<p>Weight decay changes the update rule itself. Each step, every weight loses a small fraction of itself on top of following the gradient:</p><div class="eq">w ← w − η·∇L − η·λ·w<br>  = (1 − ηλ)·w − η·∇L</div><p>Here only w₁ matters to the loss; w₂–w₅ only see noise. Press Run with decay off, then reset and try it on.</p>`,w:'decay'},
  {t:'num',q:`w = 2.0, learning rate η = 0.1, decay λ = 0.5, and the gradient is 0. What is w after one step?`,a:1.9,tol:.001,ex:`(1 − 0.1 × 0.5) × 2.0 = 0.95 × 2.0 = 1.9.`},
  {t:'task',title:'Clean up the noise',body:`<p>Turn weight decay on and train until every useless weight (w₂ to w₅) is smaller than <b>0.05</b> in size.</p>`,w:'decay',hint:`Weight decay has to be on. Then press Run (or Step many times) and watch w₂–w₅ fade.`,ex:`The gradient keeps w₁ alive, so it settles at 1.2 / (1 + λ) instead of 1.2. Nothing defends the other weights, so they fade to zero.`},
  {t:'mc',q:`With no gradient signal at all, weight decay makes a weight…`,o:[`shrink exponentially toward 0`,`jump straight to 0`,`shrink by the same amount every step`,`oscillate around 0`],ex:`Each step multiplies by (1 − ηλ), so after t steps w = (1 − ηλ)ᵗ·w₀. That is exponential decay, hence the name.`},
  {t:'num',q:`ηλ = 0.01 and the gradient is 0. After 100 steps, the weight has been multiplied by about…`,a:.366,tol:.01,ex:`0.99¹⁰⁰ ≈ 0.366, very close to e⁻¹ ≈ 0.368.`}
 ],
 cards:[
  { id: 'wd-update', front: `Write the weight-decay update rule.`, back: `w ← (1 − ηλ)·w − η·∇L.` },
  { id: 'wd-exp', front: `With zero gradient, what happens to w over t steps of weight decay?`, back: `w_t = (1 − ηλ)ᵗ·w₀: exponential decay toward 0.` },
  { id: 'wd-unused', front: `With weight decay on, what happens to weights the loss doesn’t need?`, back: `Nothing defends them, so they decay to zero. Useful weights settle a little smaller than the loss alone would choose.` },
 ]},
 {id:'wd-2',title:'L2 vs decay vs AdamW',steps:[
  {t:'learn',title:'Twins under SGD',body:`<p>Add the penalty <span class="m">(λ/2)‖w‖²</span> to the loss. Its gradient is <span class="m">λw</span>, so plain SGD gives</p><div class="eq">w ← w − η·(∇L + λw)</div><p>That is exactly the weight-decay update. <b>For SGD, L2 regularization and weight decay are the same thing.</b></p>`},
  {t:'learn',title:'Adam splits them apart',body:`<p>Adam divides each gradient by a running estimate of its size, <span class="m">√v</span>. If the penalty lives in the loss, its gradient <span class="m">λw</span> gets divided too:</p><div class="eq">Adam + L2:  w ← w − η·(∇L + λw) / √v</div><p>Weights with a history of large gradients get <em>less</em> regularization. <b>AdamW</b> (Loshchilov &amp; Hutter, 2019) decouples the decay, so every weight shrinks at the same rate:</p><div class="eq">AdamW:  w ← w − η·(Adam step) − η·λ·w</div>`},
  {t:'tf',q:`With plain SGD, L2 regularization and weight decay give the same update (with λ matched).`,a:true,ex:`The penalty’s gradient λw is exactly the decay term. The two only diverge with adaptive optimizers like Adam.`},
  {t:'mc',q:`Why was AdamW introduced?`,o:[`In Adam, an L2 penalty gets rescaled per weight, so it stops acting like uniform decay`,`Adam cannot compute the gradient of an L2 penalty`,`AdamW runs faster per step`,`AdamW removes the need for a learning rate`],ex:`Folding L2 into Adam’s gradient lets the adaptive denominator weaken it unevenly. Decoupling restores true weight decay and often generalizes better.`},
  {t:'match',q:`Match each term.`,p:[[`L2 regularization`,`Adds λ‖w‖² to the loss`],[`Weight decay`,`Multiplies w by (1 − ηλ) each step`],[`AdamW`,`Adam with decoupled decay`],[`λ`,`Regularization strength`]],ex:`Same goal, different plumbing.`},
  {t:'mc',q:`When training transformers, which parameters usually get <b>no</b> weight decay?`,o:[`Biases and LayerNorm gains`,`Attention weight matrices`,`MLP weight matrices`,`All of them`],ex:`Common practice decays the large weight matrices and skips biases and normalization parameters, where shrinking toward zero doesn’t help generalization.`}
 ],
 cards:[
  { id: 'wd-sgd', front: `Under plain SGD, are L2 regularization and weight decay the same?`, back: `Yes. The gradient of (λ/2)‖w‖² is λw, which is exactly the decay term.` },
  { id: 'wd-adam', front: `Why isn’t L2 the same as weight decay in Adam?`, back: `Adam divides each gradient, including λw, by √v. Weights with a history of large gradients get less regularization.` },
  { id: 'wd-adamw', front: `What does AdamW change?`, back: `It applies decay outside the adaptive step: w ← w − η·(Adam step) − η·λ·w, so every weight shrinks at the same rate.` },
  { id: 'wd-nodecay', front: `Which parameters usually get no weight decay?`, back: `Biases and normalization gains: the 1-D parameters.` },
 ]},
 {id:'wd-3',title:'Shapes in a PyTorch layer',steps:[
  {t:'learn',title:'nn.Linear stores (out, in)',body:`<pre class="py">layer = nn.Linear(in_features=784, out_features=128)
layer.weight.shape   # (128, 784)   (out, in), not (in, out)!
layer.bias.shape     # (128,)
x.shape              # (32, 784)    a batch of 32 inputs
y = layer(x)         # x @ layer.weight.T + layer.bias
y.shape              # (32, 128)</pre><p>Each <b>row</b> of the weight matrix holds the 784 incoming weights of one output unit.</p>`},
  {t:'shape',q:'What is the shape of <code>y</code>?',code:'x = torch.zeros(64, 784)\nlayer = nn.Linear(784, 128)\ny = layer(x)',a:'(64, 128)',ex:'(64, 784) @ (784, 128) → (64, 128), plus the bias broadcast across the 64 rows.'},
  {t:'mc',q:'What is <code>layer.weight[j]</code> (row j)?',o:['The 784 incoming weights of output unit j','The weights leaving input j','The bias of unit j','One input example'],ex:'y[:, j] = x @ weight[j] + bias[j]: row j is dotted with each input to produce output unit j.'},
  {t:'num',q:'How many parameters does <code>nn.Linear(784, 128)</code> have, including the bias?',a:100480,tol:0,ex:'784 · 128 weights + 128 biases = 100,352 + 128 = 100,480.'},
  {t:'learn',title:'One step of decoupled decay',body:`<pre class="py">with torch.no_grad():
    for p in model.parameters():
        p.mul_(1 - lr * wd)    # decay: elementwise, same shape as p
        p.sub_(lr * p.grad)    # gradient step: p.grad has p's shape</pre><p>Decay multiplies every entry of the (128, 784) matrix by the same 1 − ηλ. Nothing about the shape changes.</p>`},
  {t:'shape',q:'What is the shape of the weight’s gradient?',code:'layer = nn.Linear(784, 128)\nloss.backward()\nlayer.weight.grad',a:'(128, 784)',ex:'A gradient always has exactly its parameter’s shape: one partial derivative per entry.'},
  {t:'learn',title:'Choosing what to decay',body:`<pre class="py">decay    = [p for p in model.parameters() if p.ndim >= 2]   # weight matrices
no_decay = [p for p in model.parameters() if p.ndim &lt; 2]    # biases, norm gains
opt = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.1},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=3e-4)</pre><p>The shape does the sorting: weight matrices have 2 dimensions, while biases and LayerNorm gains are 1-D.</p>`},
  {t:'mc',q:'Why does <code>p.ndim &gt;= 2</code> pick out the parameters to decay?',o:['Weight matrices are 2-D; biases and norm gains are 1-D','2-D parameters train faster','PyTorch requires it','It skips the embeddings'],ex:'It’s a shape-based shortcut for the usual rule: decay the big matrices, leave the 1-D vectors alone.'}
 ],
 cards:[
  { id: 'wd-np-linear', front: `<code>nn.Linear(in_features=784, out_features=128)</code>: shapes of weight and bias?`, back: `weight (128, 784), i.e. (out, in). bias (128,).` },
  { id: 'wd-np-forward', front: `What does <code>nn.Linear</code> compute for x of shape (batch, in)?`, back: `<code>x @ W.T + b</code>, giving shape (batch, out).` },
  { id: 'wd-np-row', front: `In <code>nn.Linear</code>, what is row j of the weight matrix?`, back: `The incoming weights of output unit j.` },
  { id: 'wd-np-grad', front: `What shape is a parameter’s gradient?`, back: `Exactly the parameter’s shape: one partial derivative per entry.` },
  { id: 'wd-np-groups', front: `How do you exclude biases and norm gains from weight decay?`, back: `Use two optimizer param groups: <code>p.ndim &gt;= 2</code> gets weight_decay, <code>p.ndim &lt; 2</code> gets 0.` },
 ]}
]};
