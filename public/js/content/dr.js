import { num, big, shape, py } from '../vary.js';
import { sup } from '../util.js';

// Fresh numbers every time a question or card is shown (see ../vary.js).
const DROP = [0.1, 0.2, 0.25, 0.4, 0.5, 0.75], KEEP = [0.5, 0.75, 0.8, 0.9];
// h·m/k, m ~ Bernoulli(k), checked from its two outcomes rather than the formula.
const outcomes = (h, k) => `vals, probs = np.array([${h} / ${k}, 0.0]), np.array([${k}, 1 - ${k}])\nmean = vals @ probs\nvar = ((vals - mean) ** 2) @ probs`;
const gen = {
  scale(v) {
    const p = v.pick(DROP), a = 1 / (1 - p);
    return { q: `The drop rate is p = ${num(p)}. Surviving activations are multiplied by what?`, a,
      ex: `1 / (1 − ${num(p)}) = 1 / ${num(1 - p)} = ${num(a, 3)}.`, py: `${outcomes(1, `(1 - ${p})`)}\nassert np.isclose(mean, 1)\nans = vals[0]` };
  },
  masks(v) {
    const n = v.int(3, 12), a = 2 ** n;
    return { q: `A layer has ${n} droppable neurons. How many different masks (thinned sub-networks) are possible?`, a,
      ex: `Each neuron is either on or off: 2${sup(n)} = ${big(a)}.`, py: `import itertools\nans = len(list(itertools.product([0, 1], repeat=${n})))` };
  },
  maskShape(v) {
    const b = v.pick([16, 32, 64]), h = v.pick([128, 256, 512]), k = v.pick(KEEP);
    return { code: `h = np.zeros((${b}, ${h}))\nmask = rng.random(h.shape) < ${k}\nmask`, a: shape(b, h),
      py: `h = np.zeros((${b}, ${h}))\nans = (rng.random(h.shape) < ${k}).shape` };
  },
  fraction(v) {
    const b = v.pick([16, 32, 64]), h = v.pick([128, 256, 512]), k = v.pick(KEEP);
    return { q: `<code>mask = rng.random((${b}, ${h})) &lt; ${k}</code>. About what fraction of the mask is True?`, a: k,
      ex: `Each entry is True with probability ${k}, so about ${num(100 * k)}% are.`, py: `ans = (rng.random((1000, 1000)) < ${k}).mean()` };
  },
  survivor(v) {
    const h = v.pick([1, 1.5, 2, 3, 4]), k = v.pick(KEEP), a = h / k;
    return { q: `A unit outputs ${num(h)} and keep = ${num(k)}. If it survives, what value does it pass on during training?`, a,
      ex: `${num(h)} / ${num(k)} = ${num(a, 3)}.`, py: `${outcomes(h, k)}\nassert np.isclose(mean, ${h})\nans = vals[0]` };
  },
  inputShape(v) {
    const b = v.pick([16, 32, 64]), i = v.pick([256, 512, 784]), o = v.pick([10, 64, 128]), k = v.pick(KEEP);
    return { code: `x = np.zeros((${b}, ${i}))\nmask = rng.random(x.shape) < ${k}\nW = np.zeros((${o}, ${i}))\n(x * mask / ${k}) @ W.T`, a: shape(b, o),
      ex: `Dropout keeps (${b}, ${i}). Then (${b}, ${i}) @ (${i}, ${o}) → (${b}, ${o}).`,
      py: `x, W = np.zeros((${b}, ${i})), np.zeros((${o}, ${i}))\nmask = rng.random(x.shape) < ${k}\nans = ((x * mask / ${k}) @ W.T).shape` };
  },
  variance(v) {
    const h = v.pick([1, 2, 3]), k = v.pick(KEEP), a = (h * h * (1 - k)) / k;
    return { q: `h = ${num(h)} and keep = ${num(k)}. What is the variance of h·m/keep?`, a,
      ex: `h²(1 − k)/k = ${h * h} × ${num(1 - k)} / ${num(k)} = ${num(a, 3)}.`, py: `${outcomes(h, k)}\nans = var` };
  },

};

export default {id:'dr',title:'Dropout',sub:'Randomly switch off neurons while training.',c:'dr',
 guide:`<ul><li>Training: zero each unit with probability p, using a fresh random mask every step.</li><li>Inverted dropout: scale survivors by 1/(1 − p) so expected activations match test time.</li><li>Inference: no dropping and no scaling. Call <span class="m">model.eval()</span>.</li><li>PyTorch and Keras: p (or rate) is the drop probability. The original 2014 paper used p for the keep probability.</li><li>Typical rates: 0.1 to 0.5.</li><li>Why it helps: no unit can rely on specific partners, and training averages over 2ⁿ thinned networks.</li></ul><div class="eq">nn.Dropout(p=0.1)</div>
 <h3>Shapes</h3><pre class="py">mask = rng.random(h.shape) &lt; keep   # same shape as h
h * mask / keep                     # elementwise, shape unchanged</pre>`,
 lessons:[
 {id:'dr-1',title:'Unplug at random',steps:[
  {t:'learn',title:'Coin flips for neurons',body:`<p>During training, every hidden neuron is switched off (its output set to 0) with probability <b>p</b>, the drop rate. Each training step draws a fresh random mask.</p><p>Press <b>New mask</b> a few times, then switch to <b>Test</b>.</p>`,w:'drop'},
  {t:'mc',q:`Dropout is applied…`,o:[`only during training; at test time every neuron is used`,`only at test time`,`equally during training and testing`,`only to the output layer`],ex:`The randomness is a training-time regularizer. At inference you want one deterministic prediction from the full network.`},
  {t:'learn',title:'Keep the average honest',body:`<p>If half the neurons vanish, the next layer receives about half the signal it will see at test time. <b>Inverted dropout</b> fixes this by scaling the survivors up during training:</p><div class="eq">kept value × 1 / (1 − p)</div><p>With p = 0.5 the survivors are doubled, so the expected value matches test time, when nothing is dropped or scaled.</p>`},
  {t:'num',q:`The drop rate is p = 0.2. Surviving activations are multiplied by what?`,a:1.25,tol:.001,ex:`1 / (1 − 0.2) = 1 / 0.8 = 1.25.`,gen:gen.scale},
  {t:'task',title:'Dial in the scale',body:`<p>Set the drop rate so that surviving activations are scaled by exactly <b>4×</b>.</p>`,w:'drop',wo:{p:.3},hint:`You need 1 / (1 − p) = 4. What fraction of neurons has to survive?`,ex:`1 / (1 − 0.75) = 4. Only a quarter of the neurons survive each step, so each one counts four times.`},
  {t:'tf',q:`In PyTorch, <span class="m">nn.Dropout(p=0.3)</span> zeroes about 30% of activations during training.`,a:true,ex:`PyTorch and Keras use p (rate) as the drop probability. Careful when reading the original 2014 paper: there p meant the probability of <em>keeping</em> a unit.`}
 ],
 cards:[
  { id: 'dr-def', front: `What does dropout do during training?`, back: `Zeroes each unit’s output with probability p, drawing a fresh random mask every step.` },
  { id: 'dr-scale', front: `In inverted dropout, what are the survivors multiplied by?`, back: `1 / (1 − p), so the expected activation matches test time.` },
  { id: 'dr-test', front: `What does dropout do at test time?`, back: `Nothing: no units are dropped and nothing is scaled. Call <code>model.eval()</code>.` },
  { id: 'dr-p', front: `In PyTorch’s <code>nn.Dropout(p=0.3)</code>, what is p?`, back: `The drop probability: about 30% of activations are zeroed. The original 2014 paper used p for the keep probability.` },
 ]},
 {id:'dr-2',title:'Why it works',steps:[
  {t:'learn',title:'Two ways to see it',body:`<p><b>No freeloaders.</b> A neuron can’t count on any particular partner being present, so each feature has to be useful on its own. This breaks up fragile co-adaptations.</p><p><b>A huge ensemble.</b> Each mask defines a different thinned network, and they all share weights. With n droppable neurons there are <span class="m">2ⁿ</span> of them. At test time the full network is a cheap approximation to averaging them: exact for a single linear layer, only approximate once nonlinearities are involved.</p>`},
  {t:'num',q:`A layer has 10 droppable neurons. How many different masks (thinned sub-networks) are possible?`,a:1024,tol:0,ex:`Each neuron is either on or off: 2¹⁰ = 1,024.`,gen:gen.masks},
  {t:'mc',q:`You forget to call <span class="m">model.eval()</span> before inference. What happens?`,o:[`Dropout stays on, so predictions are random and usually worse`,`Nothing changes`,`The model keeps training`,`Inference gets faster`],ex:`model.train() and model.eval() switch dropout (and batch-norm behavior). Forgetting eval() is a classic bug.`},
  {t:'mc',q:`Dropout is best described as…`,o:[`a regularizer that injects noise to reduce overfitting`,`an optimizer`,`a normalization layer`,`a learning-rate schedule`],ex:`Like L2 and weight decay, its job is better generalization, here by training under random noise.`},
  {t:'tf',q:`A higher dropout rate always gives better test accuracy.`,a:false,ex:`Too much dropout starves the network and it underfits. Choose the rate from evidence: raise it when the train–validation gap is large and growing, and lower it when training error itself is too high.`},
  {t:'match',q:`Match the dropout terms.`,p:[[`p (drop rate)`,`Chance a unit is zeroed`],[`1 / (1 − p)`,`Scale for surviving units`],[`model.eval()`,`Turns dropout off`],[`Thinned network`,`One random sub-network`]],ex:`You can now read any dropout code.`}
 ],
 cards:[
  { id: 'dr-coadapt', front: `Why does dropout help? (the co-adaptation view)`, back: `No unit can rely on a specific partner being present, so each feature has to be useful on its own.` },
  { id: 'dr-ensemble', front: `Why does dropout help? (the ensemble view)`, back: `Each mask is a thinned network. Training averages over 2ⁿ weight-sharing sub-networks, and the full network at test time approximates their average.` },
  { id: 'dr-eval', front: `What goes wrong if you forget <code>model.eval()</code> at inference?`, back: `Dropout stays on: predictions are random and usually worse.` },
  { id: 'dr-rate', front: `How do you choose a dropout rate?`, back: `From validation evidence: more dropout when the model overfits (a large, growing gap), less or none when it underfits or the dataset is large relative to the model.` },
 ]},
 {id:'dr-3',title:'Dropout in NumPy',steps:[
  {t:'learn',title:'A mask with the same shape',body:`<pre class="py">h.shape                              # (32, 256): batch of 32, 256 features
p = 0.2                              # drop rate
keep = 1 - p
mask = rng.random(h.shape) &lt; keep    # (32, 256) True/False, one coin per entry
h_train = h * mask / keep            # (32, 256) elementwise, survivors scaled up
h_test = h                           # test time: no mask, no scaling</pre><p>The <code>*</code> here is elementwise, not matrix multiplication: the mask has exactly h’s shape.</p>`},
  {t:'shape',q:'What is the shape of <code>mask</code>?',code:'h = np.zeros((32, 256))\nmask = rng.random(h.shape) < 0.8\nmask',a:'(32, 256)',ex:'One independent coin flip per activation, so the mask has h’s shape.',gen:gen.maskShape},
  {t:'tf',q:'Each example in the batch gets its own random mask.',a:true,ex:'The mask is (32, 256): every row (example) has its own coins.'},
  {t:'num',q:'<code>mask = rng.random((32, 256)) &lt; 0.8</code>. About what fraction of the mask is True?',a:0.8,tol:0.02,ex:'Each entry is True with probability 0.8, so about 80% are.',gen:gen.fraction},
  {t:'mc',q:'Why divide by <code>keep</code> during training?',o:['So each unit’s expected value equals its test-time value','To make the mask binary','To speed up training','To normalize the batch'],ex:'E[h·mask/keep] = h·keep/keep = h. Then at test time no scaling is needed.'},
  {t:'num',q:'A unit outputs 2.0 and keep = 0.8. If it survives, what value does it pass on during training?',a:2.5,tol:0.001,ex:'2.0 / 0.8 = 2.5.',gen:gen.survivor},
  {t:'shape',q:'Dropout on a layer’s input. What shape comes out?',code:'x = np.zeros((32, 256))\nmask = rng.random(x.shape) < 0.9\nW = np.zeros((128, 256))\n(x * mask / 0.9) @ W.T',a:'(32, 128)',ex:'Dropout keeps (32, 256). Then (32, 256) @ (256, 128) → (32, 128).',gen:gen.inputShape},
  {t:'learn',title:'Mean and variance of one dropped unit',body:`<p>Take a fixed activation h and a mask m that is 1 with keep probability k. The unit passes on h·m/k:</p><div class="eq">E[h·m/k] = h·E[m]/k = h<br>Var[h·m/k] = h²·Var[m]/k² = h²·k(1 − k)/k² = h²(1 − k)/k</div><p>Dropout keeps the mean and adds noise, and lower keep probability means more noise.</p>`},
  {t:'num',q:'h = 2.0 and keep = 0.8. What is the variance of h·m/keep?',a:1,tol:0.001,ex:'h²(1 − k)/k = 4 × 0.2 / 0.8 = 1.0.',gen:gen.variance},
  {t:'mc',q:'Inverted dropout makes each unit’s expected value match test time. Does the network’s expected output match too?',o:['Not in general: for a nonlinear f, E[f(z)] ≠ f(E[z])','Yes, always','Only with ReLU activations','Only at the output layer'],ex:'Matching what goes into a nonlinearity on average doesn’t match what comes out on average. Test-time scaling approximates averaging over masks; it is exact only for linear parts.'},
  {t:'learn',title:'eval() and no_grad() are different switches',body:'<p><code>model.eval()</code> changes layer behavior: dropout turns off and batch norm uses its running statistics. <code>torch.no_grad()</code> only stops recording gradients. Inference usually wants both.</p><p><b>Monte Carlo dropout</b> breaks the rule on purpose: it keeps dropout layers active at inference, runs several stochastic passes and uses their spread as an uncertainty estimate. That’s an intentional choice, not the default.</p>'},
  {t:'tf',q:'<code>torch.no_grad()</code> turns dropout off.',a:false,ex:'no_grad only disables gradient tracking. Dropout follows the module’s train/eval mode, so you still need <code>model.eval()</code>.'},
  {t:'explain',kind:'derive',secs:90,q:'Derive the mean and variance of h·m/k for m ~ Bernoulli(k), then explain why matching the mean doesn’t make the whole network’s expected output match test time.',model:`<p>E[m] = k and Var[m] = k(1 − k). So E[h·m/k] = h, and Var[h·m/k] = h²·k(1 − k)/k² = h²(1 − k)/k.</p><p>That fixes each unit’s <em>mean</em>. But the next layer applies a nonlinearity f, and in general E[f(z)] ≠ f(E[z]) (Jensen’s inequality for convex or concave f). So the test-time network, which feeds f the means, is only an approximation to averaging the network’s output over masks. It is exact only where the computation is linear.</p>`,rubric:['E[m] = k and Var[m] = k(1 − k)','Mean of h·m/k is h','Variance is h²(1 − k)/k','Explains E[f(z)] ≠ f(E[z]) for nonlinear f','Concludes test-time scaling is an approximation to averaging']}
 ],
 cards:[
  { id: 'dr-np-mask', front: `How do you build a dropout mask for activations h with keep probability k?`, back: `<code>mask = rng.random(h.shape) &lt; k</code>: same shape as h, one coin per entry.` },
  { id: 'dr-np-apply', front: `Apply inverted dropout to h in NumPy.`, back: `<code>h * mask / keep</code>: elementwise, and the shape doesn’t change.` },
  { id: 'dr-np-expect', front: `Why does inverted dropout divide by keep?`, back: `E[h·mask/keep] = h, so training and test activations match on average.` },
  { id: 'dr-np-var', front: `Variance of a dropped unit h·m/k, with keep probability k?`, back: `h²(1 − k)/k. The mean stays h.` },
  { id: 'dr-np-nonlin', front: `Why isn’t test-time dropout scaling an exact average over masks?`, back: `For a nonlinear f, E[f(z)] ≠ f(E[z]). Matching each unit’s mean doesn’t match the network’s mean output.` },
  { id: 'dr-np-evalnograd', front: `What do <code>model.eval()</code> and <code>torch.no_grad()</code> each do?`, back: `eval(): changes layer behavior (dropout off, batch norm running stats). no_grad(): stops gradient tracking. They’re independent.` },
  { id: 'dr-mc', front: `What is Monte Carlo dropout?`, back: `Keeping dropout on at inference on purpose, running several passes, and using their spread as an uncertainty estimate.` },
  { id: 'dr-derive-meanvar', front: 'What does inverted dropout do to a unit’s mean and its variance?', back: 'The mean stays the same (that’s what dividing by keep is for); the variance goes up.' },
  { id: 'dr-np-shape', front: `Does dropout change an activation’s shape?`, back: `No. It’s elementwise: the mask and the output have the same shape as the input.` },
 ]}
]};
