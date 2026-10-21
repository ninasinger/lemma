import { $, $$, reduced, rng, gauss, SUB, sup, minus, onDispose, sfx, highlight } from './util.js';

/* ================= math helpers ================= */
function svd2(m){const[a,b,c,d]=m,E=(a+d)/2,F=(a-d)/2,G=(c+b)/2,H=(c-b)/2,Q=Math.hypot(E,H),R=Math.hypot(F,G),a1=Math.atan2(G,F),a2=Math.atan2(H,E);return{s1:Q+R,s2:Q-R,th:(a2-a1)/2,ph:(a2+a1)/2}}
function solve(A,b){const n=b.length;A=A.map(r=>r.slice());b=b.slice();for(let i=0;i<n;i++){let p=i;for(let k=i+1;k<n;k++)if(Math.abs(A[k][i])>Math.abs(A[p][i]))p=k;[A[i],A[p]]=[A[p],A[i]];[b[i],b[p]]=[b[p],b[i]];for(let k=i+1;k<n;k++){const f=A[k][i]/A[i][i];for(let j=i;j<n;j++)A[k][j]-=f*A[i][j];b[k]-=f*b[i]}}const x=new Array(n).fill(0);for(let i=n-1;i>=0;i--){let s=b[i];for(let j=i+1;j<n;j++)s-=A[i][j]*x[j];x[i]=s/A[i][i]}return x}
function jacobi(B){const n=B.length,A=B.map(r=>r.slice()),V=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));for(let sw=0;sw<60;sw++){let off=0;for(let p=0;p<n;p++)for(let q=p+1;q<n;q++)off+=A[p][q]**2;if(off<1e-18)break;for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){if(Math.abs(A[p][q])<1e-15)continue;const th=(A[q][q]-A[p][p])/(2*A[p][q]),t=Math.sign(th||1)/(Math.abs(th)+Math.sqrt(th*th+1)),c=1/Math.sqrt(t*t+1),s=t*c;for(let k=0;k<n;k++){const x=A[k][p],y=A[k][q];A[k][p]=c*x-s*y;A[k][q]=s*x+c*y}for(let k=0;k<n;k++){const x=A[p][k],y=A[q][k];A[p][k]=c*x-s*y;A[q][k]=s*x+c*y}for(let k=0;k<n;k++){const x=V[k][p],y=V[k][q];V[k][p]=c*x-s*y;V[k][q]=s*x+c*y}}}return{vals:A.map((r,i)=>r[i]),V}}
function gridSVG(w,h,sc,cx,cy){let s='';for(let x=cx%sc;x<=w;x+=sc)s+=`<line class="g" x1="${x}" y1="0" x2="${x}" y2="${h}"/>`;for(let y=cy%sc;y<=h;y+=sc)s+=`<line class="g" x1="0" y1="${y}" x2="${w}" y2="${y}"/>`;return s+`<line class="ax" x1="0" y1="${cy}" x2="${w}" y2="${cy}"/><line class="ax" x1="${cx}" y1="0" x2="${cx}" y2="${h}"/>`}
function arrow(x1,y1,x2,y2,cls){const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy);if(L<1)return'';const ux=dx/L,uy=dy/L,hs=Math.min(10,L*.5),bx=x2-ux*hs,by=y2-uy*hs,w=hs*.6;return `<g class="arr ${cls}"><line x1="${x1}" y1="${y1}" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}"/><polygon points="${x2.toFixed(1)},${y2.toFixed(1)} ${(bx-uy*w).toFixed(1)},${(by+ux*w).toFixed(1)} ${(bx+uy*w).toFixed(1)},${(by-ux*w).toFixed(1)}"/></g>`}
const svgPt=(svg,e,W,H)=>{const r=svg.getBoundingClientRect();return[(e.clientX-r.left)/r.width*W,(e.clientY-r.top)/r.height*H]};

/* ================= widgets ================= */
export const W={};

W.pca=function(el,o){
  const R=rng(11),pts=[];
  for(let i=0;i<34;i++){const a=gauss(R)*1.55,b=gauss(R)*.5,t=.62;pts.push([a*Math.cos(t)-b*Math.sin(t),a*Math.sin(t)+b*Math.cos(t)])}
  const n=pts.length,mx=pts.reduce((s,p)=>s+p[0],0)/n,my=pts.reduce((s,p)=>s+p[1],0)/n;pts.forEach(p=>{p[0]-=mx;p[1]-=my});
  let cxx=0,cxy=0,cyy=0;pts.forEach(([x,y])=>{cxx+=x*x;cxy+=x*y;cyy+=y*y});cxx/=n;cxy/=n;cyy/=n;
  const tot=cxx+cyy,pc=((.5*Math.atan2(2*cxy,cxx-cyy))%Math.PI+Math.PI)%Math.PI;
  const vAt=t=>{const c=Math.cos(t),s=Math.sin(t);return cxx*c*c+2*cxy*c*s+cyy*s*s};
  let th=((o.start??150)%180)*Math.PI/180,show=false;
  el.innerHTML=`<div class="wbox"><svg class="wsvg grab" viewBox="0 0 320 240" role="img" aria-label="Scatter plot with a line through the center that you can rotate"></svg>
   <div class="wrow"><div class="rng"><label for="pca-ang">Angle</label><input id="pca-ang" type="range" min="0" max="179" step="1"><output></output></div>${o.reveal?'<button type="button" class="chip" id="pca-show" aria-pressed="false">Show PC1</button>':''}</div>
   <div class="meters"><div class="meter"><span>Captured variance</span><div class="track"><i></i></div><b></b></div>${o.residuals?'<div class="meter alt"><span>Leftover error</span><div class="track"><i></i></div><b></b></div>':''}</div></div>`;
  const svg=$('svg',el),inp=$('input',el),out=$('output',el),ms=$$('.meter',el),sc=30,cx=160,cy=120,X=x=>(cx+x*sc).toFixed(1),Y=y=>(cy-y*sc).toFixed(1);
  function draw(){
    const u=[Math.cos(th),Math.sin(th)];let s=gridSVG(320,240,sc,cx,cy),proj='';
    if(show)s+=`<line class="ghost" x1="${X(-6*Math.cos(pc))}" y1="${Y(-6*Math.sin(pc))}" x2="${X(6*Math.cos(pc))}" y2="${Y(6*Math.sin(pc))}"/>`;
    pts.forEach(p=>{const d=p[0]*u[0]+p[1]*u[1],q=[d*u[0],d*u[1]];if(o.residuals)s+=`<line class="resid" x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(q[0])}" y2="${Y(q[1])}"/>`;proj+=`<circle class="proj" cx="${X(q[0])}" cy="${Y(q[1])}" r="2.7"/>`});
    s+=`<line class="aline" x1="${X(-6*u[0])}" y1="${Y(-6*u[1])}" x2="${X(6*u[0])}" y2="${Y(6*u[1])}"/>`;
    pts.forEach(p=>s+=`<circle class="pt" cx="${X(p[0])}" cy="${Y(p[1])}" r="3.6"/>`);
    svg.innerHTML=s+proj+`<circle class="handle" cx="${X(3.7*u[0])}" cy="${Y(3.7*u[1])}" r="10"/>`;
    const deg=Math.round(th*180/Math.PI)%180;inp.value=deg;out.textContent=deg+'°';
    const cap=vAt(th)/tot;$('i',ms[0]).style.width=(cap*100).toFixed(1)+'%';$('b',ms[0]).textContent=Math.round(cap*100)+'%';
    if(ms[1]){$('i',ms[1]).style.width=((1-cap)*100).toFixed(1)+'%';$('b',ms[1]).textContent=Math.round((1-cap)*100)+'%'}
  }
  inp.addEventListener('input',()=>{th=inp.value*Math.PI/180;draw()});
  let drag=false;
  const setFrom=e=>{const[px,py]=svgPt(svg,e,320,240);if(Math.hypot(px-cx,py-cy)<6)return;th=((Math.atan2(-(py-cy),px-cx)%Math.PI)+Math.PI)%Math.PI;draw()};
  svg.addEventListener('pointerdown',e=>{drag=true;try{svg.setPointerCapture(e.pointerId)}catch(_){}setFrom(e)});
  svg.addEventListener('pointermove',e=>{if(drag)setFrom(e)});
  ['pointerup','pointercancel'].forEach(t=>svg.addEventListener(t,()=>drag=false));
  if(o.reveal)$('#pca-show',el).addEventListener('click',e=>{show=!show;e.currentTarget.setAttribute('aria-pressed',show);draw()});
  draw();
  return{solved(){let d=Math.abs(th-pc);d=Math.min(d,Math.PI-d);return d<4*Math.PI/180}};
};

W.svd=function(el,o){
  const P=[['Stretch + turn',[1.5,.8,.3,1]],['Shear',[1,1,0,1]],['Rotation',[.866,-.5,.5,.866]],['Rank 1',[1,.5,1,.5]]];
  let pi=o.preset??0,d,a=[0,0,0],raf=0;
  el.innerHTML=`<div class="wbox"><div class="chips" role="group" aria-label="Matrix presets">${P.map((p,i)=>`<button type="button" class="chip" data-i="${i}" aria-pressed="false">${p[0]}</button>`).join('')}</div>
   <div class="svdgrid"><svg class="wsvg" viewBox="0 0 320 240" role="img" aria-label="The unit circle being transformed by the matrix A, one SVD step at a time"></svg><div class="svdside"><div class="mat"></div><div class="sig"></div></div></div>
   <div class="steps3">${['Vᵀ rotate','Σ stretch','U rotate'].map((t,i)=>`<button type="button" class="stepbtn" data-s="${i}"><span>${i+1}</span>${t}</button>`).join('')}<button type="button" class="chip" data-reset>Reset</button></div>
   <p class="cap" aria-live="polite"></p></div>`;
  const svg=$('svg',el),sc=42,cx=160,cy=120,X=x=>cx+x*sc,Y=y=>cy-y*sc,f=v=>v.toFixed(1);
  const rot=(p,t)=>[p[0]*Math.cos(t)-p[1]*Math.sin(t),p[0]*Math.sin(t)+p[1]*Math.cos(t)];
  const T=p=>{p=rot(p,d.th*a[0]);p=[p[0]*(1+(d.s1-1)*a[1]),p[1]*(1+(d.s2-1)*a[1])];return rot(p,d.ph*a[2])};
  function setP(i){pi=i;d=svd2(P[i][1]);a=[0,0,0];cancelAnimationFrame(raf);raf=0;draw()}
  function draw(){
    const m=P[pi][1],stage=a.filter(x=>x===1).length;let s=gridSVG(320,240,sc,cx,cy),g='',c='';
    for(let k=0;k<=96;k++){const t=k/96*2*Math.PI,p=[Math.cos(t),Math.sin(t)],q=[m[0]*p[0]+m[1]*p[1],m[2]*p[0]+m[3]*p[1]],r=T(p);g+=(k?'L':'M')+f(X(q[0]))+' '+f(Y(q[1]));c+=(k?'L':'M')+f(X(r[0]))+' '+f(Y(r[1]))}
    s+=`<path class="ghostp" d="${g}"/><path class="blob" d="${c}Z"/>`;
    const v1=[Math.cos(-d.th),Math.sin(-d.th)],v2=[-Math.sin(-d.th),Math.cos(-d.th)],t1=T(v1),t2=T(v2);
    s+=arrow(cx,cy,X(t1[0]),Y(t1[1]),'a1')+arrow(cx,cy,X(t2[0]),Y(t2[1]),'a2');
    const labs=[['v₁','v₂'],['e₁','e₂'],['σ₁e₁','σ₂e₂'],['σ₁u₁','σ₂u₂']][stage];
    [[t1,labs[0],'al1'],[t2,labs[1],'al2']].forEach(([t,l,cl])=>{const L=Math.hypot(t[0],t[1]);if(L<.05)return;const k=(L+.38)/L;s+=`<text class="${cl}" x="${f(X(t[0]*k))}" y="${f(Y(t[1]*k)+4)}" text-anchor="middle">${l}</text>`});
    svg.innerHTML=s;
    $('.mat',el).innerHTML=`A =<span class="br"><span>${m[0].toFixed(2)}</span><span>${m[1].toFixed(2)}</span><span>${m[2].toFixed(2)}</span><span>${m[3].toFixed(2)}</span></span>`;
    const mx=Math.max(d.s1,1.9);$('.sig',el).innerHTML=`<div><span>σ₁</span><i style="width:${d.s1/mx*100}%"></i><span>${d.s1.toFixed(2)}</span></div><div><span>σ₂</span><i style="width:${Math.max(d.s2,0)/mx*100}%"></i><span>${Math.max(d.s2,0).toFixed(2)}</span></div>`;
    $$('.chips .chip',el).forEach((b,i)=>b.setAttribute('aria-pressed',i===pi));
    $$('.stepbtn',el).forEach((b,i)=>{b.classList.toggle('done',a[i]===1);b.disabled=!!raf||a[i]>0||(i>0&&a[i-1]<1)});
    const caps=[`The unit circle, with v₁ and v₂ marked. The dashed outline is where A will send it. Press step 1.`,`Vᵀ rotated the plane so v₁ and v₂ now sit on the axes. The circle looks the same, but every point moved.`,`Σ stretched the x-axis by σ₁ = ${d.s1.toFixed(2)} and the y-axis by σ₂ = ${Math.max(d.s2,0).toFixed(2)}.${d.s2<.01?' σ₂ is 0, so the circle collapses to a line.':''}`,`U rotated the result into place. It lands exactly on the dashed outline: three simple moves add up to A.`];
    $('.cap',el).textContent=caps[stage];
  }
  function go(i){if(reduced.matches){a[i]=1;draw();return}const t0=performance.now();const fr=now=>{const t=Math.min(1,(now-t0)/750);a[i]=t<1?(t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2):1;if(t<1){raf=requestAnimationFrame(fr)}else raf=0;draw()};raf=requestAnimationFrame(fr);draw()}
  $$('.chips .chip',el).forEach(b=>b.addEventListener('click',()=>setP(+b.dataset.i)));
  $$('.stepbtn',el).forEach(b=>b.addEventListener('click',()=>go(+b.dataset.s)));
  $('[data-reset]',el).addEventListener('click',()=>setP(pi));
  onDispose(()=>cancelAnimationFrame(raf));
  setP(pi);
  return{};
};

let IMG=null;
function imgData(){
  if(IMG)return IMG;const S=48,A=[];
  for(let y=0;y<S;y++){const row=[];for(let x=0;x<S;x++){const u=(x+.5)/S*2-1,v=(y+.5)/S*2-1;let val=.08;if(Math.hypot(u,v)<.82)val=.92;if(Math.hypot(u+.32,v+.25)<.13||Math.hypot(u-.32,v+.25)<.13)val=.08;const rs=Math.hypot(u,v-.02);if(rs>.44&&rs<.56&&v>.12)val=.08;row.push(val)}A.push(row)}
  const AtA=Array.from({length:S},(_,i)=>Array.from({length:S},(_,j)=>{let s=0;for(let k=0;k<S;k++)s+=A[k][i]*A[k][j];return s}));
  const{vals,V}=jacobi(AtA),idx=vals.map((v,i)=>i).sort((p,q)=>vals[q]-vals[p]);
  const vs=idx.map(i=>V.map(r=>r[i])),sv2=idx.map(i=>Math.max(vals[i],0)),AV=vs.map(v=>A.map(row=>row.reduce((s,x,j)=>s+x*v[j],0)));
  const tot=sv2.reduce((p,q)=>p+q,0),err=[1];let c=0;sv2.forEach(v=>{c+=v;err.push(Math.sqrt(Math.max(0,1-c/tot)))});
  const rank=sv2.filter(v=>v>1e-9*sv2[0]).length;
  return IMG={S,A,vs,AV,err,rank};
}
W.img=function(el,o){
  const D=imgData(),S=D.S,kmin=D.err.findIndex((e,i)=>i>0&&e<.1);let k=o.k??1;
  el.innerHTML=`<div class="wbox"><div class="imgs"><figure><canvas width="${S}" height="${S}" aria-label="Original image"></canvas><figcaption>Original (rank ${D.rank})</figcaption></figure><figure><canvas width="${S}" height="${S}" aria-label="Rank-k approximation"></canvas><figcaption class="kc"></figcaption></figure></div>
   <div class="rng"><label for="img-k">k</label><input id="img-k" type="range" min="1" max="20" step="1"><output></output></div>
   <div class="stats"><div><span>Error${o.goal?' · goal under 10%':''}</span><b class="e"></b></div><div><span>Numbers stored</span><b class="sto"></b></div></div></div>`;
  const cv=$$('canvas',el),inp=$('input',el);
  function paint(c,M){const ctx=c.getContext('2d'),im=ctx.createImageData(S,S);for(let y=0;y<S;y++)for(let x=0;x<S;x++){const v=Math.max(0,Math.min(1,M(y,x))),i=(y*S+x)*4;im.data[i]=27+v*(242-27);im.data[i+1]=35+v*(245-35);im.data[i+2]=51+v*(251-51);im.data[i+3]=255}ctx.putImageData(im,0,0)}
  paint(cv[0],(y,x)=>D.A[y][x]);
  function draw(){inp.value=k;$('output',el).textContent=k;paint(cv[1],(y,x)=>{let s=0;for(let t=0;t<k;t++)s+=D.AV[t][y]*D.vs[t][x];return s});
    $('.kc',el).textContent=`Rank ${k} approximation`;const e=D.err[k];$('.e',el).textContent=(e*100).toFixed(1)+'%';$('.e',el).classList.toggle('good',!!o.goal&&e<.1);
    $('.sto',el).textContent=`${(k*(2*S+1)).toLocaleString()} of ${(S*S).toLocaleString()}`}
  inp.addEventListener('input',()=>{k=+inp.value;draw()});draw();
  return{solved:()=>k===kmin};
};

let RD=null;
function ridgeData(){
  if(RD)return RD;const r=rng(7),f=x=>Math.sin(2.6*x)+.3*x,N=12,P=12,xs=[],ys=[];
  for(let i=0;i<N;i++){const x=-.95+1.9*i/(N-1)+(r()-.5)*.08;xs.push(x);ys.push(f(x)+gauss(r)*.22)}
  const XtX=Array.from({length:P},()=>new Array(P).fill(0)),Xty=new Array(P).fill(0);
  xs.forEach((x,i)=>{for(let j=0;j<P;j++){Xty[j]+=x**j*ys[i];for(let k=0;k<P;k++)XtX[j][k]+=x**(j+k)}});
  const pv=(w,x)=>{let s=0,p=1;for(const c of w){s+=c*p;p*=x}return s},fits=[];
  for(let i=0;i<=100;i++){const L=-8+i/10,w=solve(XtX.map((row,j)=>row.map((v,k)=>v+(j===k&&j>0?10**L:0))),Xty);let tr=0,te=0;xs.forEach((x,n)=>tr+=(pv(w,x)-ys[n])**2);tr/=N;for(let k=0;k<200;k++){const x=-1+2*k/199;te+=(pv(w,x)-f(x))**2}te/=200;fits.push({L,w,tr,te,norm:Math.hypot(...w.slice(1))})}
  return RD={f,xs,ys,pv,fits};
}
W.ridge=function(el,o){
  const D=ridgeData(),GOAL=.025;let i=o.start??0;
  el.innerHTML=`<div class="wbox"><div class="rgrid"><figure><svg class="wsvg" viewBox="0 0 320 200" role="img" aria-label="Degree-11 polynomial fit through twelve noisy points"></svg>
    <figcaption class="legend"><span><i class="lg-dot"></i>training points</span><span><i class="lg-dash"></i>true function</span><span><i></i>your fit</span></figcaption></figure>
    <div class="rside"><svg class="wsvg" viewBox="0 0 300 136" role="img" aria-label="Train and test error as lambda changes"></svg>
    <div class="stats"><div><span>Train error</span><b class="tr"></b></div><div><span>Test error</span><b class="te"></b></div><div><span>Weight size ‖w‖</span><b class="nw"></b></div></div></div></div>
    <div class="rng"><label for="ridge-l">λ</label><input id="ridge-l" type="range" min="0" max="100" step="1"><output></output></div></div>`;
  const[sv,su]=$$('svg',el),inp=$('input',el),X=x=>(16+(x+1)/2*288).toFixed(1),Y=y=>(100-y*36).toFixed(1);
  let tru='';for(let k=0;k<=120;k++){const x=-1+2*k/120;tru+=(k?'L':'M')+X(x)+' '+Y(D.f(x))}
  const UX=L=>36+(L+8)/10*256,UY=e=>8+(.5-Math.log10(Math.max(e,1e-5)))/4.5*96;
  let trp='',tep='';D.fits.forEach((q,k)=>{trp+=(k?'L':'M')+UX(q.L).toFixed(1)+' '+UY(q.tr).toFixed(1);tep+=(k?'L':'M')+UX(q.L).toFixed(1)+' '+UY(q.te).toFixed(1)});
  let ubase='';[[1,'1'],[.1,'0.1'],[.01,'0.01'],[.001,'10⁻³'],[1e-4,'10⁻⁴']].forEach(([e,l])=>{ubase+=`<line class="g" x1="36" y1="${UY(e)}" x2="292" y2="${UY(e)}"/><text x="31" y="${UY(e)+4}" text-anchor="end">${l}</text>`});
  [-8,-6,-4,-2,0,2].forEach(L=>ubase+=`<text x="${UX(L)}" y="128" text-anchor="middle">10${sup(L)}</text>`);
  if(o.goal)ubase+=`<line class="goal" x1="36" y1="${UY(GOAL)}" x2="292" y2="${UY(GOAL)}"/><text class="goalt" x="290" y="${UY(GOAL)-4}" text-anchor="end">goal</text>`;
  ubase+=`<path class="trp" d="${trp}"/><path class="tep" d="${tep}"/><text class="tt" x="44" y="${UY(D.fits[0].te)+14}">test</text><text x="44" y="${UY(D.fits[0].tr)-6}">train</text>`;
  function draw(){
    const q=D.fits[i];let s=`<defs><clipPath id="rclip"><rect x="0" y="0" width="320" height="200"/></clipPath></defs>`;
    for(let y=-2;y<=2;y++)s+=`<line class="${y?'g':'ax'}" x1="0" y1="${Y(y)}" x2="320" y2="${Y(y)}"/>`;
    let fp='';for(let k=0;k<=160;k++){const x=-1+2*k/160;fp+=(k?'L':'M')+X(x)+' '+Y(Math.max(-9,Math.min(9,D.pv(q.w,x))))}
    s+=`<path class="truef" d="${tru}"/><path class="fitp" clip-path="url(#rclip)" d="${fp}"/>`;
    D.xs.forEach((x,n)=>s+=`<circle class="pt2" cx="${X(x)}" cy="${Y(D.ys[n])}" r="4.5"/>`);
    sv.innerHTML=s;
    su.innerHTML=ubase+`<line class="mk" x1="${UX(q.L)}" y1="4" x2="${UX(q.L)}" y2="108"/><circle class="mkd" cx="${UX(q.L)}" cy="${UY(q.te)}" r="5"/>`;
    inp.value=i;$('output',el).innerHTML=`10<sup>${minus(q.L.toFixed(1))}</sup>`;
    $('.tr',el).textContent=q.tr.toFixed(4);$('.te',el).textContent=q.te.toFixed(4);$('.te',el).classList.toggle('good',!!o.goal&&q.te<GOAL);
    $('.nw',el).textContent=q.norm>=100?Math.round(q.norm).toLocaleString():q.norm.toFixed(1);
  }
  inp.addEventListener('input',()=>{i=+inp.value;draw()});draw();
  return{solved:()=>D.fits[i].te<GOAL};
};

W.contour=function(el,o){
  const al=-.17,h1=4,h2=.3,e1=[Math.cos(al),Math.sin(al)],e2=[-Math.sin(al),Math.cos(al)],ws=[2,1.4];
  const p1=ws[0]*e1[0]+ws[1]*e1[1],p2=ws[0]*e2[0]+ws[1]*e2[1];
  const sol=lam=>{const k1=h1/(h1+lam)*p1,k2=h2/(h2+lam)*p2;return[k1*e1[0]+k2*e2[0],k1*e1[1]+k2*e2[1]]};
  const loss=w=>{const d=[w[0]-ws[0],w[1]-ws[1]],q1=d[0]*e1[0]+d[1]*e1[1],q2=d[0]*e2[0]+d[1]*e2[1];return h1*q1*q1+h2*q2*q2};
  const sc=56,ox=72,oy=180,X=x=>(ox+x*sc).toFixed(1),Y=y=>(oy-y*sc).toFixed(1);
  const ell=cv=>{let d='';for(let k=0;k<=96;k++){const t=k/96*2*Math.PI,a=Math.sqrt(cv/h1)*Math.cos(t),b=Math.sqrt(cv/h2)*Math.sin(t);d+=(k?'L':'M')+X(ws[0]+a*e1[0]+b*e2[0])+' '+Y(ws[1]+a*e1[1]+b*e2[1])}return d+'Z'};
  let rp='';for(let k=0;k<=120;k++){const w=sol(10**(-3+k*.05));rp+=(k?'L':'M')+X(w[0])+' '+Y(w[1])}
  let base=`<defs><clipPath id="cclip"><rect x="0" y="0" width="320" height="240"/></clipPath></defs>`+gridSVG(320,240,sc,ox,oy)+'<g clip-path="url(#cclip)">';
  [.08,.3,.7,1.3,2.1].forEach(c=>base+=`<path class="lvl" d="${ell(c)}"/>`);
  base+=`<line class="eax" x1="${X(ws[0]-1.1*e1[0])}" y1="${Y(ws[1]-1.1*e1[1])}" x2="${X(ws[0]+1.1*e1[0])}" y2="${Y(ws[1]+1.1*e1[1])}"/><line class="eax" x1="${X(ws[0]-1.5*e2[0])}" y1="${Y(ws[1]-1.5*e2[1])}" x2="${X(ws[0]+1.5*e2[0])}" y2="${Y(ws[1]+1.5*e2[1])}"/>`;
  base+=`<text x="${X(ws[0]+1.18*e1[0])}" y="${+Y(ws[1]+1.18*e1[1])+4}">steep</text><text x="${+X(ws[0]+1.5*e2[0])+6}" y="${+Y(ws[1]+1.5*e2[1])+10}">flat</text>`;
  let li=o.start??0;
  el.innerHTML=`<div class="wbox"><svg class="wsvg" viewBox="0 0 320 240" role="img" aria-label="Loss contours, penalty circle and the ridge solution path"></svg>
    <div class="rng"><label for="ct-l">λ</label><input id="ct-l" type="range" min="0" max="70" step="1"><output></output></div>
    <div class="stats"><div><span>Weights w</span><b class="wv"></b></div><div><span>Steep direction kept</span><b class="k1"></b></div><div><span>Flat direction kept</span><b class="k2"></b></div></div></div>`;
  const svg=$('svg',el),inp=$('input',el);
  function draw(){
    const L=-2+li*.05,lam=10**L,w=sol(lam),r=Math.hypot(w[0],w[1]);
    let s=base+`<circle class="pen" cx="${X(0)}" cy="${Y(0)}" r="${(r*sc).toFixed(1)}"/><path class="lvlnow" d="${ell(Math.max(loss(w),1e-4))}"/><path class="rpath" d="${rp}"/></g>`;
    s+=`<path class="wstar" d="M${+X(ws[0])-5} ${+Y(ws[1])-5}l10 10M${+X(ws[0])+5} ${+Y(ws[1])-5}l-10 10"/><text class="ink" x="${+X(ws[0])+8}" y="${+Y(ws[1])-8}">w* (no penalty)</text>`;
    s+=`<circle class="org" cx="${X(0)}" cy="${Y(0)}" r="3.5"/><text class="ink" x="${+X(0)-12}" y="${+Y(0)+16}">0</text><circle class="wnow" cx="${X(w[0])}" cy="${Y(w[1])}" r="7"/>`;
    svg.innerHTML=s;inp.value=li;$('output',el).innerHTML=`10<sup>${minus(L.toFixed(2))}</sup>`;
    $('.wv',el).textContent=`(${w[0].toFixed(2)}, ${w[1].toFixed(2)})`;$('.k1',el).textContent=Math.round(h1/(h1+lam)*100)+'%';$('.k2',el).textContent=Math.round(h2/(h2+lam)*100)+'%';
  }
  inp.addEventListener('input',()=>{li=+inp.value;draw()});draw();
  return{};
};

W.decay=function(el,o){
  const init=[.15,1.1,-.9,.7,-1.2],eta=.1,target=1.2;let w,t,R,timer=0,on=!!o.on,lam=.5;
  el.innerHTML=`<div class="wbox"><svg class="wsvg" viewBox="0 0 320 170" role="img" aria-label="Bar chart of five weights during training"></svg>
    <div class="wlabels">${init.map((_,i)=>`<div><span>w${SUB[i+1]}</span><b></b><em>${i?'noise':'useful'}</em></div>`).join('')}</div>
    <div class="eq small" aria-live="polite"></div>
    <div class="wrow"><label class="switch"><input type="checkbox" id="wd-on">Weight decay</label><div class="rng"><label for="wd-lam">λ</label><input id="wd-lam" type="range" min="0.1" max="1" step="0.05"><output></output></div></div>
    <div class="wrow"><button type="button" class="chip" data-a="step">Step</button><button type="button" class="chip" data-a="run">Run</button><button type="button" class="chip" data-a="reset">Reset</button><span class="tcount"></span></div></div>`;
  const svg=$('svg',el),sw=$('#wd-on',el),sl=$('#wd-lam',el),runb=$('[data-a=run]',el),Y=v=>85-v*48;
  sw.checked=on;sl.value=lam;
  function stop(){clearInterval(timer);timer=0;runb.textContent='Run'}
  function step(){t++;w=w.map((v,i)=>{const g=i?(R()-.5)*.06:(v-target);return v-eta*(g+(on?lam:0)*v)})}
  function draw(){
    let s=`<line class="ax" x1="4" y1="85" x2="316" y2="85"/><line class="goal" x1="8" y1="${Y(target)}" x2="56" y2="${Y(target)}"/><text class="goalt" x="32" y="${Y(target)-5}" text-anchor="middle">target</text>`;
    w.forEach((v,i)=>{const x=14+i*64,y0=85,y1=Y(Math.max(-1.6,Math.min(1.6,v)));s+=`<rect class="wb${i?'':' use'}" x="${x}" y="${Math.min(y0,y1).toFixed(1)}" width="36" height="${Math.max(1.5,Math.abs(y1-y0)).toFixed(1)}" rx="5"/>`});
    svg.innerHTML=s;$$('.wlabels b',el).forEach((b,i)=>b.textContent=minus(w[i].toFixed(3)));
    $('.eq',el).textContent=on?`each step: w ← ${(1-eta*lam).toFixed(3)}·w − ${eta}·∇L`:`each step: w ← w − ${eta}·∇L (no decay)`;
    $('output',el).textContent=lam.toFixed(2);$('.tcount',el).textContent='step '+t;
  }
  function reset(){stop();w=init.slice();t=0;R=rng(3);draw()}
  sw.addEventListener('change',()=>{on=sw.checked;draw()});
  sl.addEventListener('input',()=>{lam=+sl.value;draw()});
  $('[data-a=step]',el).addEventListener('click',()=>{step();draw()});
  runb.addEventListener('click',()=>{if(timer)return stop();runb.textContent='Pause';timer=setInterval(()=>{step();draw();if(t>=800)stop()},reduced.matches?120:50)});
  $('[data-a=reset]',el).addEventListener('click',reset);
  onDispose(stop);reset();
  return{solved:()=>on&&w.slice(1).every(v=>Math.abs(v)<.05)};
};

W.drop=function(el,o){
  const L=[3,5,5,2],xs=[36,122,208,288];let p=o.p??.5,mode='train',mask;
  const pos=L.map((n,li)=>Array.from({length:n},(_,i)=>[xs[li],14+(i+.5)*(186/n)]));
  el.innerHTML=`<div class="wbox"><div class="wrow"><div class="chips" role="group" aria-label="Mode"><button type="button" class="chip" data-m="train">Train</button><button type="button" class="chip" data-m="test">Test</button></div><button type="button" class="chip" data-a="mask">New mask</button></div>
    <svg class="wsvg" viewBox="0 0 320 222" role="img" aria-label="Small neural network with some hidden neurons dropped"></svg>
    <div class="rng"><label for="dr-p">p</label><input id="dr-p" type="range" min="0" max="0.9" step="0.05"><output></output></div>
    <p class="cap" aria-live="polite"></p></div>`;
  const svg=$('svg',el),inp=$('input',el);inp.value=p;
  const remask=()=>{mask=L.map((n,li)=>Array.from({length:n},()=>li===0||li===3||Math.random()>=p))};
  function draw(){
    const scale=1/(1-p),on=(li,i)=>mode==='test'||mask[li][i];let s='';
    for(let li=0;li<3;li++)pos[li].forEach((a,i)=>pos[li+1].forEach((b,j)=>{s+=`<line class="edge${on(li,i)&&on(li+1,j)?'':' off'}" x1="${a[0]}" y1="${a[1].toFixed(1)}" x2="${b[0]}" y2="${b[1].toFixed(1)}"/>`}));
    pos.forEach((col,li)=>col.forEach(([x,y],i)=>{const hid=li===1||li===2;
      if(!hid)s+=`<circle class="nio" cx="${x}" cy="${y}" r="11"/>`;
      else if(on(li,i)){s+=`<circle class="nkeep" cx="${x}" cy="${y}" r="15"/>`;if(mode==='train'&&p>0)s+=`<text class="nsc" x="${x}" y="${y+3.5}" text-anchor="middle">×${scale<9.95?scale.toFixed(1):Math.round(scale)}</text>`}
      else s+=`<circle class="ndrop" cx="${x}" cy="${y}" r="15"/><path class="nx" d="M${x-5} ${y-5}l10 10M${x+5} ${y-5}l-10 10"/>`}));
    ['input','hidden 1','hidden 2','output'].forEach((t,i)=>s+=`<text x="${xs[i]}" y="217" text-anchor="middle">${t}</text>`);
    svg.innerHTML=s;$('output',el).textContent=p.toFixed(2);
    const kept=mask[1].filter(Boolean).length+mask[2].filter(Boolean).length;
    $('.cap',el).textContent=mode==='test'?'Test time: all 10 hidden neurons are on and nothing is scaled. The full network stands in for the average of every thinned network.':(p===0?'p = 0, so nothing is dropped. Slide p up.':`Training step: ${kept} of 10 hidden neurons survived. Each survivor’s output is multiplied by 1/(1 − ${p.toFixed(2)}) = ${scale.toFixed(2)}.`);
    $$('[data-m]',el).forEach(b=>b.setAttribute('aria-pressed',b.dataset.m===mode));$('[data-a=mask]',el).disabled=mode==='test';
  }
  inp.addEventListener('input',()=>{p=+inp.value;remask();draw()});
  $$('[data-m]',el).forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.m;draw()}));
  $('[data-a=mask]',el).addEventListener('click',()=>{remask();draw();sfx.tap()});
  remask();draw();
  return{solved:()=>Math.abs(p-.75)<1e-9};
};

W.scree=function(el){
  const v=[5,2,1.2,.5,.2,.1],tot=v.reduce((a,b)=>a+b,0);let k=1;
  el.innerHTML=`<div class="wbox"><div class="scree" role="group" aria-label="Choose how many components to keep">${v.map((x,i)=>`<button type="button" class="sbar" data-i="${i}" aria-label="Keep ${i+1} component${i?'s':''}"><em>${x.toFixed(1)}</em><i style="height:${Math.max(6,x/5*130)}px"></i><span>PC${i+1}</span></button>`).join('')}</div>
    <div class="meter"><span>Variance kept</span><div class="track"><i></i><u style="left:90%"></u></div><b></b></div><p class="cap" aria-live="polite"></p></div>`;
  function draw(){const c=v.slice(0,k).reduce((a,b)=>a+b,0)/tot;$$('.sbar',el).forEach((b,i)=>{b.classList.toggle('on',i<k);b.setAttribute('aria-pressed',i<k)});$('.track i',el).style.width=(c*100)+'%';$('.meter b',el).textContent=(c*100).toFixed(1)+'%';$('.cap',el).textContent=`Keeping k = ${k} component${k>1?'s':''}: ${(c*100).toFixed(1)}% of the variance with ${k} of 6 numbers per row. The black tick marks 90%.`}
  $$('.sbar',el).forEach(b=>b.addEventListener('click',()=>{k=+b.dataset.i+1;sfx.tap();draw()}));draw();
  return{solved:()=>k===3};
};

/* ---------- matrix foundations ---------- */
const fmtNum=v=>minus(String(Math.round(v*100)/100));
// A factor in a written-out product: negatives get parentheses, so 0·(−2) not 0·−2.
const fac=v=>v<0?`(${fmtNum(v)})`:fmtNum(v);

// Two draggable arrows; shows a·b two ways and the projection of b onto a.
W.dot=function(el,o){
  const a=[...(o.a||[3,1])],b=[...(o.b||[1,2])];let drag=null;
  const sc=30,cx=160,cy=125,X=x=>cx+x*sc,Y=y=>cy-y*sc;
  el.innerHTML=`<div class="wbox"><svg class="wsvg grab" viewBox="0 0 320 250" role="img" aria-label="Two arrows, a and b, that you can drag"></svg><div class="dotread" aria-live="polite"></div><p class="cap"></p></div>`;
  const svg=$('svg',el);
  function draw(){
    const dot=a[0]*b[0]+a[1]*b[1],la=Math.hypot(a[0],a[1]),lb=Math.hypot(b[0],b[1]);
    let s=gridSVG(320,250,sc,cx,cy);
    if(la>0){const k=dot/(la*la),p=[a[0]*k,a[1]*k];
      s+=`<line class="projdrop" x1="${X(b[0])}" y1="${Y(b[1])}" x2="${X(p[0])}" y2="${Y(p[1])}"/><line class="projseg${dot<0?' neg':''}" x1="${cx}" y1="${cy}" x2="${X(p[0])}" y2="${Y(p[1])}"/>`}
    s+=arrow(cx,cy,X(a[0]),Y(a[1]),'a1')+arrow(cx,cy,X(b[0]),Y(b[1]),'a2');
    const lab=(v,t,cl)=>{const L=Math.hypot(v[0],v[1])||1,k=(L+.55)/L;return `<text class="${cl}" x="${X(v[0]*k)}" y="${Y(v[1]*k)+4}" text-anchor="middle">${t}</text>`};
    s+=`<circle class="tip t1" cx="${X(a[0])}" cy="${Y(a[1])}" r="9"/><circle class="tip t2" cx="${X(b[0])}" cy="${Y(b[1])}" r="9"/>`+lab(a,'a','al1')+lab(b,'b','al2');
    svg.innerHTML=s;
    const cos=la&&lb?dot/(la*lb):0,ang=la&&lb?Math.acos(Math.max(-1,Math.min(1,cos)))*180/Math.PI:0,F=fmtNum;
    $('.dotread',el).innerHTML=`<div class="drow"><span class="m va">a = [${F(a[0])}, ${F(a[1])}]</span><span class="m vb">b = [${F(b[0])}, ${F(b[1])}]</span></div>
      <p class="eqline">a·b = ${fac(a[0])}·${fac(b[0])} + ${fac(a[1])}·${fac(b[1])} = <b>${F(dot)}</b></p>
      <p class="eqline">‖a‖ ‖b‖ cos θ = ${la.toFixed(2)} × ${lb.toFixed(2)} × ${minus(cos.toFixed(2))} = <b>${minus((la*lb*cos).toFixed(2))}</b></p>`;
    $('.cap',el).textContent=Math.abs(dot)<1e-9?`θ = 90°. Zero: the arrows are perpendicular, so b has no component along a.`:dot>0?`θ = ${Math.round(ang)}°. Positive: b points partly along a. The thick segment is b’s shadow on a.`:`θ = ${Math.round(ang)}°. Negative: b points partly against a.`;
  }
  const setFrom=e=>{const[px,py]=svgPt(svg,e,320,250);const v=[Math.max(-5,Math.min(5,Math.round((px-cx)/sc*2)/2)),Math.max(-4,Math.min(4,Math.round((cy-py)/sc*2)/2))];drag[0]=v[0];drag[1]=v[1];draw()};
  svg.addEventListener('pointerdown',e=>{const[px,py]=svgPt(svg,e,320,250),da=Math.hypot(px-X(a[0]),py-Y(a[1])),db=Math.hypot(px-X(b[0]),py-Y(b[1]));drag=da<=db?a:b;try{svg.setPointerCapture(e.pointerId)}catch(_){}setFrom(e)});
  svg.addEventListener('pointermove',e=>{if(drag)setFrom(e)});
  ['pointerup','pointercancel'].forEach(t=>svg.addEventListener(t,()=>drag=null));
  draw();
  return{solved:()=>Math.abs(a[0]*b[0]+a[1]*b[1])<1e-9&&Math.hypot(...a)>0&&Math.hypot(...b)>0};
};

// C = A @ B with every entry tappable: shows which row and column produce it, term by term.
W.matmul=function(el,o){
  const A=o.A,B=o.B,m=A.length,n=A[0].length,p=B[0].length,N=o.names||['A','B','C'],F=fmtNum;
  const C=A.map(r=>B[0].map((_,j)=>r.reduce((s,v,k)=>s+v*B[k][j],0)));
  let[si,sj]=o.sel||[0,0],view='row';
  const vec=p===1&&!o.noColumnView;
  const grid=(M,{rows,cols,cell,btn})=>{
    const nc=M[0].length;let h=`<div class="mgrid${rows?' hasrl':''}" style="grid-template-columns:${rows?'auto ':''}repeat(${nc},minmax(2.4em,auto))">`;
    if(cols){if(rows)h+='<span class="ml"></span>';cols.forEach(c=>h+=`<span class="ml mlc">${c}</span>`)}
    M.forEach((r,i)=>{if(rows)h+=`<span class="ml mlr">${rows[i]}</span>`;r.forEach((v,j)=>{const c=cell(i,j);h+=btn?`<button type="button" class="mc ${c}" data-i="${i}" data-j="${j}" aria-label="${N[2]}[${i}, ${j}] = ${F(v)}">${F(v)}</button>`:`<span class="mc ${c}">${F(v)}</span>`})});
    return h+'</div>';
  };
  el.innerHTML=`<div class="wbox"><div class="mm"></div>${vec?'<div class="chips" role="group" aria-label="How to read it"><button type="button" class="chip" data-v="row">Row view</button><button type="button" class="chip" data-v="col">Column view</button></div>':''}<div class="mmx" aria-live="polite"></div></div>`;
  function draw(){
    const kc=k=>`k${k%4}`,col=view==='col';
    const aCell=(i,k)=>col?`on ${kc(k)}`:i===si?`on ${kc(k)}`:'';
    const bCell=(k,j)=>col?`on ${kc(k)}`:j===sj?`on ${kc(k)}`:'';
    const cCell=(i,j)=>col||(i===si&&j===sj)?'sel':'';
    const fig=(name,M,opts,klass)=>`<figure class="${klass}"><figcaption><b>${name}</b> <span>shape (${M.length}, ${M[0].length})</span></figcaption>${grid(M,opts)}</figure>`;
    $('.mm',el).innerHTML=fig(N[0],A,{rows:o.aRows,cols:o.k,cell:aCell},'mmA')+'<span class="mmop" aria-hidden="true">@</span>'+fig(N[1],B,{rows:o.k,cols:o.bCols,cell:bCell},'mmB')+'<span class="mmop" aria-hidden="true">=</span>'+fig(N[2],C,{rows:o.aRows,cols:o.bCols,cell:cCell,btn:true},'mmC');
    $$('.mmC .mc',el).forEach(b=>b.addEventListener('click',()=>{si=+b.dataset.i;sj=+b.dataset.j;view='row';sfx.tap();draw()}));
    $$('[data-v]',el).forEach(b=>b.setAttribute('aria-pressed',b.dataset.v===view));
    let x;
    if(col){
      const terms=[...Array(n).keys()].map(k=>`<span class="${kc(k)}">${fac(B[k][0])}·[${A.map(r=>F(r[k])).join(', ')}]</span>`).join(' + ');
      x=`<p class="mmf"><b>${N[2]}</b> is a mix of the columns of ${N[0]}, weighted by the entries of ${N[1]}:</p><p class="mmf">${terms} = <b>[${C.map(r=>F(r[0])).join(', ')}]</b></p>`;
    }else{
      const terms=[...Array(n).keys()].map(k=>`<span class="${kc(k)}">${fac(A[si][k])}·${fac(B[k][sj])}</span>`).join(' + ');
      x=`<p class="mmf"><b>${N[2]}[${si}, ${sj}]</b> = row ${si} of ${N[0]} · column ${sj} of ${N[1]} = Σ<sub>k</sub> ${N[0]}[${si}, k]·${N[1]}[k, ${sj}]</p><p class="mmf">= ${terms} = <b>${F(C[si][sj])}</b></p>`;
      if(o.explain)x+=`<p class="mmnote">${o.explain(si,sj,C)}</p>`;
      else x+=`<p class="mmnote">Tap any entry of ${N[2]}. The inner index k runs along row ${si} of ${N[0]} and down column ${sj} of ${N[1]} together; matching colors are multiplied.</p>`;
    }
    $('.mmx',el).innerHTML=x;
  }
  $$('[data-v]',el).forEach(b=>b.addEventListener('click',()=>{view=b.dataset.v;sfx.tap();draw()}));
  draw();
  return{solved:()=>!o.target||(si===o.target[0]&&sj===o.target[1]&&view==='row')};
};

// X.sum / X.mean / X.max with axis=None, 0 or 1: shows which way the array collapses.
W.axis=function(el,o){
  const X=o.X||[[1,2,3,4],[5,6,7,8],[9,10,11,12]],m=X.length,n=X[0].length,F=fmtNum;
  let op=o.op||'sum',axis=o.axis===undefined?null:o.axis,keep=false;
  const AX=[[null,'None'],[0,'0'],[1,'1']];
  el.innerHTML=`<div class="wbox"><div class="wrow axctl"><div class="chips" role="group" aria-label="Operation">${['sum','mean','max'].map(x=>`<button type="button" class="chip" data-op="${x}">.${x}()</button>`).join('')}</div><div class="chips" role="group" aria-label="Axis">${AX.map(([v,t])=>`<button type="button" class="chip" data-ax="${t}">axis=${t}</button>`).join('')}</div><label class="switch"><input type="checkbox" id="ax-keep">keepdims</label></div>
    <div class="axview" aria-live="polite"></div><pre class="py axcode"></pre><p class="cap"></p></div>`;
  const agg=v=>op==='sum'?v.reduce((s,x)=>s+x,0):op==='mean'?v.reduce((s,x)=>s+x,0)/v.length:Math.max(...v);
  function draw(){
    const res=axis===0?X[0].map((_,j)=>agg(X.map(r=>r[j]))):axis===1?X.map(r=>agg(r)):[agg(X.flat())];
    const shape=axis===null?(keep?'(1, 1)':'()'):axis===0?(keep?`(1, ${n})`:`(${n},)`):(keep?`(${m}, 1)`:`(${m},)`);
    const cols=n+(axis===1?2:0)+1;let g=`<div class="axgrid" style="grid-template-columns:auto repeat(${cols-1},minmax(2.3em,auto))"><span class="axh">axis 0 ↓ / axis 1 →</span>`;
    for(let j=0;j<n;j++)g+=`<span class="axi">${j}</span>`;if(axis===1)g+='<span></span><span class="axi">result</span>';
    X.forEach((r,i)=>{g+=`<span class="axi">${i}</span>`;r.forEach((v,j)=>{const grp=axis===0?j:axis===1?i:0;g+=`<span class="mc g${grp%4}">${F(v)}</span>`});if(axis===1)g+=`<span class="axarr">→</span><span class="mc res g${i%4}">${F(res[i])}</span>`});
    if(axis===0){g+='<span></span>'+X[0].map(()=>'<span class="axarr">↓</span>').join('')+'<span class="axi">result</span>'+res.map((v,j)=>`<span class="mc res g${j%4}">${F(v)}</span>`).join('')}
    g+='</div>';
    if(axis===null)g+=`<p class="axall">All ${m*n} numbers → <span class="mc res g0">${F(res[0])}</span></p>`;
    $('.axview',el).innerHTML=g;
    const arg=axis===null?(keep?'keepdims=True':''):`axis=${axis}${keep?', keepdims=True':''}`;
    const val=axis===null&&!keep?F(res[0]):axis===null?`array([[${F(res[0])}]])`:axis===0?(keep?`array([[${res.map(F).join(', ')}]])`:`array([${res.map(F).join(', ')}])`):(keep?`array([${res.map(v=>`[${F(v)}]`).join(', ')}])`:`array([${res.map(F).join(', ')}])`);
    const code=$('.axcode',el);code.removeAttribute('data-hl');code.textContent=`X.shape            # (${m}, ${n})\nX.${op}(${arg})\n# ${val.replace(/−/g,'-')}\n# shape ${shape}`;highlight(el);
    $('.cap',el).textContent=axis===null?`No axis: all ${m*n} numbers collapse into one value${keep?', kept as a 1×1 array':''}.`:axis===0?`axis=0 runs down the rows. ${op} collapses the ${m} rows, leaving one value per column: shape ${shape}.`:`axis=1 runs across the columns. ${op} collapses the ${n} columns, leaving one value per row: shape ${shape}.`;
    $$('[data-op]',el).forEach(b=>b.setAttribute('aria-pressed',b.dataset.op===op));
    $$('[data-ax]',el).forEach(b=>b.setAttribute('aria-pressed',b.dataset.ax===(axis===null?'None':String(axis))));
  }
  $$('[data-op]',el).forEach(b=>b.addEventListener('click',()=>{op=b.dataset.op;sfx.tap();draw()}));
  $$('[data-ax]',el).forEach(b=>b.addEventListener('click',()=>{axis=b.dataset.ax==='None'?null:+b.dataset.ax;sfx.tap();draw()}));
  $('#ax-keep',el).addEventListener('change',e=>{keep=e.target.checked;draw()});
  draw();
  return{solved:()=>!o.goal||(op===o.goal.op&&axis===o.goal.axis)};
};
