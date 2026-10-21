// Small helpers shared by the app, the widgets and the lesson content.
// Nothing here touches the DOM at import time, so content files can be loaded in Node tests.

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const reduced = {
  get matches() { return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches; },
};

export function rng(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
export function gauss(r){let u=0;while(!u)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r())}
export function shuffle(a,r=Math.random){a=a.slice();for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
export const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
export const SUB='₀₁₂₃₄₅₆₇₈₉';
export const sup=s=>String(s).replace(/[-0-9]/g,c=>'⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]);
export const minus=s=>String(s).replace('-','−');

let disposers=[];
export const onDispose=f=>disposers.push(f);
export function disposeAll(){disposers.forEach(f=>{try{f()}catch(e){}});disposers=[]}

/* ---------- rich text: matrices and Python ---------- */

// Content can write a matrix as {{1 2; 3 4}}: rows split by ";", entries by spaces or commas.
// A column vector is {{1; 2; 3}}.
export function rich(html){
  return String(html).replace(/\{\{([^}]+)\}\}/g,(_,body)=>{
    const rows=body.split(';').map(r=>r.trim().split(/[\s,]+/).filter(Boolean));
    const cols=Math.max(...rows.map(r=>r.length));
    return `<span class="mtx" style="--cols:${cols}" role="img" aria-label="matrix ${esc(body.trim())}">${rows.map(r=>r.map(c=>`<span>${c}</span>`).join('')).join('')}</span>`;
  });
}

const PY=/(#[^\n]*)|('(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*")|\b(import|from|as|def|return|for|in|if|else|elif|with|lambda|True|False|None|and|or|not|is)\b|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b/g;
// Colors comments, strings, keywords and numbers in every <pre class="py"> inside root.
export function highlight(root){
  root.querySelectorAll('pre.py:not([data-hl])').forEach(pre=>{
    const src=pre.textContent.replace(/^\n+|\s+$/g,'');
    let out='',last=0;
    src.replace(PY,(m,com,str,kw,num,at)=>{
      out+=esc(src.slice(last,at));
      const cls=com?'c':str?'s':kw?'k':'n';
      out+=`<span class="${cls}">${esc(m)}</span>`;last=at+m.length;return m;
    });
    pre.innerHTML=out+esc(src.slice(last));pre.dataset.hl='1';pre.tabIndex=0;
  });
}

// Shapes are compared as lists of dimension names: "(n, d)", "n,d" and "[n, d]" all match,
// "()" is a 0-d result (a single number) and "error" means the line raises.
export function parseShape(text){
  const t=String(text).trim().toLowerCase();
  if(/^(error|err|fails?|valueerror|it fails|an error)$/.test(t))return['error'];
  const inner=t.replace(/^[([]\s*/,'').replace(/\s*[)\]]$/,'');
  return inner.split(',').map(p=>p.replace(/\s+/g,'')).filter(Boolean);
}
export const sameShape=(a,b)=>{const x=parseShape(a),y=parseShape(b);return x.length===y.length&&x.every((v,i)=>v===y[i])};

/* ---------- icons ---------- */
export const ICON={
 check:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
 star:'<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2.8l2.8 5.9 6.4.8-4.7 4.4 1.2 6.4L12 17.2l-5.7 3.1 1.2-6.4L2.8 9.5l6.4-.8z"/></svg>',
 play:'<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.9l10.2-6.5a1 1 0 0 0 0-1.7L9.5 4.6A1 1 0 0 0 8 5.5z"/></svg>',
 trophy:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 4h10v5a5 5 0 0 1-10 0z" fill="currentColor"/><path d="M7 6H4v1.5A3.5 3.5 0 0 0 7.5 11M17 6h3v1.5a3.5 3.5 0 0 1-3.5 3.5M12 14v4M8 21h8M9.5 18h5"/></svg>',
 book:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 19.5V5a2 2 0 0 1 2-2h14v15H6a2 2 0 0 0-2 2zM4 19.5A2 2 0 0 0 6 21.5h14"/></svg>',
 heart:'<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 21s-7.6-4.7-10-9.4C.4 8.3 2.4 4 6.4 4c2.3 0 4 1.3 5.6 3.2C13.6 5.3 15.3 4 17.6 4c4 0 6 4.3 4.4 7.6C19.6 16.3 12 21 12 21z"/></svg>',
 flame:'<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12.5 2c.6 3.4 5.5 5.6 5.5 11a6 6 0 0 1-12 0c0-2.6 1.4-4.6 3-5.8-.2 2 .6 3.4 1.8 3.8C10.2 7.6 10.8 4.6 12.5 2z"/></svg>',
 bolt:'<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M13.5 2 4.5 13.5h6.5L9.5 22l10-12.5h-6.6z"/></svg>',
 x:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>',
 snd:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z" fill="currentColor"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/></svg>',
 mute:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z" fill="currentColor"/><path d="M16 9.5l5 5M21 9.5l-5 5"/></svg>',
 lock:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5" y="10.5" width="14" height="10" rx="2.5" fill="currentColor"/><path d="M8 10.5V8a4 4 0 0 1 8 0v2.5"/></svg>',
 cards:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true"><rect x="3" y="7.5" width="13" height="13.5" rx="2.5" fill="currentColor" fill-opacity=".2"/><path d="M7.5 4h11A2.5 2.5 0 0 1 21 6.5V17"/></svg>'
};
export function mascot(cls){return `<svg class="${cls||''}" viewBox="0 0 40 40" aria-hidden="true"><ellipse cx="20" cy="20" rx="16.5" ry="9.5" transform="rotate(-30 20 20)" fill="var(--pca)" opacity=".18"/><ellipse cx="20" cy="20" rx="16.5" ry="9.5" transform="rotate(-30 20 20)" fill="none" stroke="var(--pca)" stroke-width="2.6"/><path d="M20 20L34.3 11.8" stroke="var(--svd)" stroke-width="3.2" stroke-linecap="round"/><path d="M20 20L24.8 28.2" stroke="var(--dr)" stroke-width="3.2" stroke-linecap="round"/><circle cx="20" cy="20" r="2.8" fill="var(--ink)"/></svg>`}

/* ---------- sound ---------- */
export const sound={muted:false};
let actx=null;
function tone(fs,dur,type,gap,vol){if(sound.muted)return;try{actx=actx||new(window.AudioContext||window.webkitAudioContext)();const t0=actx.currentTime+.01;fs.forEach((f,i)=>{const o=actx.createOscillator(),g=actx.createGain(),t=t0+i*gap;o.type=type;o.frequency.value=f;g.gain.setValueAtTime(.0001,t);g.gain.exponentialRampToValueAtTime(vol,t+.012);g.gain.exponentialRampToValueAtTime(.0001,t+dur);o.connect(g).connect(actx.destination);o.start(t);o.stop(t+dur+.03)})}catch(e){}}
export const sfx={ok:()=>tone([660,990],.12,'sine',.08,.09),bad:()=>tone([233,196],.18,'triangle',.11,.07),tap:()=>tone([520],.05,'sine',0,.035),win:()=>tone([523,659,784,1047],.18,'sine',.1,.09),flip:()=>tone([440,587],.07,'sine',.05,.04)};
