import { $, $$, reduced, shuffle, esc, rich, highlight, onDispose, disposeAll, sameShape, ICON, mascot, sound, sfx } from './util.js';
import { W } from './widgets.js';
import { schedule, preview, formatInterval, dayStart, DAY, MINUTE } from './srs.js';
import { UNITS } from './content/index.js';
import { vary } from './vary.js';

const NEW_CARDS_PER_DAY = 20;
const LEARN_AHEAD = 20 * MINUTE;

/* ================= progress (lessons, XP, streak) ================= */
// The server is the source of truth. The page keeps a working copy in `st` and sends
// every change back; the server merges it, so a stale tab can't erase finished lessons.
const blankProgress=()=>({done:{},xp:0,streak:0,last:null,muted:false});
let st=blankProgress(),me=null,unsynced=null;
const applyPrefs=()=>{sound.muted=!!st.muted};
class SignedOut extends Error{}
async function api(method,path,body){
  let r;
  try{r=await fetch(path,{method,credentials:'same-origin',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)})}
  catch(e){throw new Error('Couldn’t reach the server. Check your connection and try again.')}
  if(r.status===401&&path!=='/api/login'){onSignedOut(true);throw new SignedOut()}
  const data=r.status===204?null:await r.json().catch(()=>null);
  if(!r.ok)throw new Error((data&&data.error)||`The server answered ${r.status}.`);
  return data;
}
let saveTimer=0,failures=0,dirty=false,inFlight=false;
function save(){dirty=true;setSync('saving');clearTimeout(saveTimer);saveTimer=setTimeout(flush,250)}
async function flush(){
  if(!dirty||!me)return;
  if(inFlight){clearTimeout(saveTimer);saveTimer=setTimeout(flush,250);return}
  dirty=false;inFlight=true;
  try{
    const merged=await api('PUT','/api/progress',st);
    failures=0;
    // If something changed while this save was in flight, send again; otherwise adopt the merged copy.
    if(dirty){clearTimeout(saveTimer);saveTimer=setTimeout(flush,250)}
    else{st=merged;applyPrefs();setSync('saved');if(!$('#path').hidden)renderStats()}
  }catch(e){
    if(e instanceof SignedOut)return;
    dirty=true;failures++;setSync('error');
    if(failures===1)toast('Couldn’t save your progress. Retrying in the background.');
    clearTimeout(saveTimer);saveTimer=setTimeout(flush,Math.min(30000,1000*2**failures));
  }finally{inFlight=false}
}
function setSync(state){const m=$('#sync-msg');if(!m)return;m.classList.toggle('err',state==='error');m.textContent=state==='error'?'Not saved yet: can’t reach the server. Retrying.':state==='saving'?'Saving…':'All progress saved to your account.'}
function toast(msg){const t=$('#toast');t.textContent=msg;t.hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>{t.hidden=true},4500)}
const dayStr=t=>{const d=new Date(t);return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0')};
function liveStreak(){if(!st.last)return 0;return(st.last===dayStr(Date.now())||st.last===dayStr(Date.now()-864e5))?st.streak:0}
function bumpStreak(){const t=dayStr(Date.now()),y=dayStr(Date.now()-864e5);if(st.last===t)return;st.streak=st.last===y?st.streak+1:1;st.last=t}

/* ================= review cards (spaced repetition) ================= */
// Card states live on the server. Each grade is also written to an outbox in localStorage
// first, so a review done offline (or in a tab that closes) is sent the next time it can be.
let rv={},outbox={},rvTimer=0,rvFails=0,rvInFlight=false;
const outboxKey=()=>`lemma-review-outbox:${me}`;
function loadOutbox(){try{return JSON.parse(localStorage.getItem(outboxKey()))||{}}catch(e){return{}}}
function saveOutbox(){try{if(Object.keys(outbox).length)localStorage.setItem(outboxKey(),JSON.stringify(outbox));else localStorage.removeItem(outboxKey())}catch(e){}}
function queueReview(id,state){outbox[id]=state;saveOutbox();clearTimeout(rvTimer);rvTimer=setTimeout(flushReviews,400)}
async function flushReviews(){
  if(!me||!Object.keys(outbox).length)return;
  if(rvInFlight){clearTimeout(rvTimer);rvTimer=setTimeout(flushReviews,400);return}
  const batch={...outbox};rvInFlight=true;
  try{
    await api('PUT','/api/reviews',{cards:batch});
    for(const id of Object.keys(batch))if(outbox[id]===batch[id])delete outbox[id];
    saveOutbox();rvFails=0;setSync('saved');
    if(Object.keys(outbox).length){clearTimeout(rvTimer);rvTimer=setTimeout(flushReviews,400)}
  }catch(e){
    if(e instanceof SignedOut)return;
    rvFails++;setSync('error');
    if(rvFails===1)toast('Reviews are saved on this device and will sync when the server is reachable.');
    clearTimeout(rvTimer);rvTimer=setTimeout(flushReviews,Math.min(30000,1000*2**rvFails));
  }finally{rvInFlight=false}
}
function unlockedCards(){const out=[];UNITS.forEach(u=>u.lessons.forEach(l=>{if(st.done[l.id])(l.cards||[]).forEach(c=>out.push({...c,u,l}))}));return out}
function reviewCounts(now=Date.now()){
  const cards=unlockedCards(),today=dayStart(now);
  let learn=0,review=0,fresh=0,tomorrow=0,nextDue=Infinity;
  const introducedToday=Object.values(rv).filter(s=>s.intro>=today).length;
  for(const c of cards){
    const s=rv[c.id];
    if(!s){fresh++;continue}
    if(s.due<=now){if(s.phase==='review')review++;else learn++}
    else{nextDue=Math.min(nextDue,s.due);if(s.due>=today+DAY&&s.due<today+2*DAY)tomorrow++}
  }
  const freshToday=Math.min(fresh,Math.max(0,NEW_CARDS_PER_DAY-introducedToday));
  return{learn,review,fresh:freshToday,freshLeft:fresh,tomorrow,nextDue,unlocked:cards.length,due:learn+review+freshToday};
}
function relTime(t,now=Date.now()){
  const ms=t-now;if(!Number.isFinite(ms))return'';
  if(ms<MINUTE)return'in under a minute';
  if(ms<3600e3)return`in ${Math.round(ms/MINUTE)} min`;
  if(t<dayStart(now)+DAY)return`in ${Math.round(ms/3600e3)} h`;
  const days=Math.round((dayStart(t)-dayStart(now))/DAY);
  return days<=1?'tomorrow':`in ${days} days`;
}

/* ================= path view ================= */
const REVIEW_UNIT={id:'review',title:'Mixed review',sub:'Ten questions drawn from every unit, shuffled.',c:'gold',review:true,lessons:[{id:'review',title:'Mixed review',steps:null}]};
const ALL=[];UNITS.forEach(u=>u.lessons.forEach(l=>ALL.push({u,l})));
function buildMixedQuiz(){const pool=[];UNITS.forEach(u=>u.lessons.forEach(l=>l.steps.forEach(s=>{if(['mc','tf','num','shape'].includes(s.t))pool.push(s)})));return shuffle(pool).slice(0,10)}
const views=['login','path','lesson','review','done'];
function show(id){views.forEach(k=>$('#'+k).hidden=k!==id);$('#top').hidden=id!=='path';window.scrollTo(0,0)}
// A lesson passes only with every question right the first time. A unit opens once every lesson
// in the units before it has passed; the mixed review at the end opens once the whole course has.
const unitPassed=u=>u.lessons.every(l=>st.done[l.id]);
const unitOpen=ui=>UNITS.slice(0,ui).every(unitPassed);
const nextUp=()=>ALL.find(x=>!st.done[x.l.id])||(!st.done.review?{u:REVIEW_UNIT,l:REVIEW_UNIT.lessons[0]}:null);
function renderStats(){
  const s=liveStreak();
  $('#st-streak').innerHTML=ICON.flame+`<b>${s}</b>`;$('#st-streak').classList.toggle('off',!s);$('#st-streak').title=`${s} day streak`;
  $('#st-xp').innerHTML=ICON.bolt+`<b>${st.xp}</b>`;
  $('#mute').innerHTML=st.muted?ICON.mute:ICON.snd;$('#mute').setAttribute('aria-pressed',st.muted);$('#mute').setAttribute('aria-label',st.muted?'Turn sound on':'Mute sound effects');
  renderReviewPanel();
}
function renderReviewPanel(){
  const c=reviewCounts(),top=$('#rv-top'),go=$('#rp-go');
  top.innerHTML=ICON.cards+(c.due?`<span class="badge">${c.due>99?'99+':c.due}</span>`:'');
  top.setAttribute('aria-label',c.due?`Review ${c.due} cards`:'Reviews: all caught up');
  $('.rp-ic').innerHTML=ICON.cards;
  const panel=$('#revpanel');panel.dataset.state=!c.unlocked?'empty':c.due?'due':'clear';
  if(!c.unlocked){
    $('#rp-title').textContent='Flashcards unlock as you finish lessons';
    $('#rp-sub').textContent='Each lesson adds cards that come back right before you would forget them.';
    go.hidden=true;
  }else if(c.due){
    $('#rp-title').textContent=`${c.due} card${c.due===1?'':'s'} to review`;
    $('#rp-sub').innerHTML=`<span class="cnt cn">${c.fresh} new</span><span class="cnt cl">${c.learn} learning</span><span class="cnt cr">${c.review} review</span>`;
    go.hidden=false;
  }else{
    $('#rp-title').textContent='All caught up';
    const parts=[];if(Number.isFinite(c.nextDue))parts.push(`Next card ${relTime(c.nextDue)}`);if(c.tomorrow)parts.push(`${c.tomorrow} due tomorrow`);if(c.freshLeft)parts.push(`${c.freshLeft} new cards wait for tomorrow`);
    $('#rp-sub').textContent=parts.join(' · ')||`${c.unlocked} cards in your deck`;
    go.hidden=true;
  }
}
function renderPath(){
  renderStats();
  const nx=nextUp(),n=ALL.filter(x=>st.done[x.l.id]).length;
  $('#ib-text').textContent=nx?`${n} of ${ALL.length} lessons done · next: ${nx.l.title}`:`All ${ALL.length} lessons and the review done. Replay anything.`;
  $('#ib-bar').style.width=(n/ALL.length*100)+'%';
  $('#ib-go').textContent=nx?(n?'Continue':'Start'):'Review again';
  $('#ib-go').onclick=()=>{const t=nx||{u:REVIEW_UNIT,l:REVIEW_UNIT.lessons[0]};startLesson(t.u,t.l)};
  const box=$('#units');box.innerHTML='';let gi=0;
  [...UNITS,REVIEW_UNIT].forEach((u,ui)=>{
    const open=unitOpen(ui),blocker=UNITS.slice(0,ui).find(x=>!unitPassed(x));
    const sec=document.createElement('section');sec.className='unit'+(open?'':' locked');sec.style.cssText=`--uc:var(--${u.c});--ucd:var(--${u.c}-d)`;
    sec.innerHTML=`<div class="ubanner"><div><p class="eyebrow">${u.review?'Finale':'Unit '+(ui+1)}${open?'':' · Locked'}</p><h2>${u.title}</h2><p>${u.sub}</p></div>${u.guide?`<button type="button" class="gbtn">${ICON.book}<span>Guidebook</span></button>`:''}</div><ol class="nodes"></ol>`;
    if(u.guide)$('.gbtn',sec).addEventListener('click',()=>openGuide(u));
    const ol=$('.nodes',sec);
    u.lessons.forEach(l=>{const done=!!st.done[l.id],cur=!!nx&&nx.l===l,off=[0,1,0,-1][gi%4]*58;gi++;
      const li=document.createElement('li');li.className='node'+(done?' done':'')+(cur?' cur':'')+(open?'':' locked');li.style.setProperty('--x',off+'px');
      li.innerHTML=`${cur?'<span class="bubble">Start</span>':''}<button type="button" class="nbtn" aria-label="${esc(l.title)}${done?', passed':open?'':', locked'}">${!open?ICON.lock:done&&!u.review?ICON.check:(u.review?ICON.trophy:(cur?ICON.play:ICON.star))}</button><span class="nlabel">${l.title}</span>`;
      $('button',li).addEventListener('click',()=>open?startLesson(u,l):toast(`Pass every lesson in ${blocker.title} to unlock ${u.review?'the mixed review':u.title}.`));ol.appendChild(li)});
    box.appendChild(sec);
  });
}
function openGuide(u){const g=$('#guide');g.style.cssText=`--uc:var(--${u.c});--ucd:var(--${u.c}-d)`;$('#guide-t').textContent=`${u.title} guidebook`;$('#guide-b').innerHTML=rich(u.guide);highlight($('#guide-b'));g.hidden=false;$('#guide-x').focus()}
function closeGuide(){$('#guide').hidden=true}

/* ================= lesson engine ================= */
let LS=null;
const PRAISE=['Nice!','Exactly right','You got it','Correct!','Spot on'];
function setMain(t,en){const b=$('#mainbtn');b.textContent=t;b.disabled=!en}
function setBar(state,o={}){const cb=$('#checkbar');cb.dataset.state=state;if(state==='idle'){$('#fb').hidden=true;return}$('#fb').hidden=false;$('.fb-ic',cb).innerHTML=state==='ok'?ICON.check:ICON.x;$('.fb-t',cb).textContent=o.t;$('.fb-x',cb).innerHTML=rich(o.x||'');highlight($('.fb-x',cb))}
function updateTop(){const pct=Math.round(LS.done/LS.total*100);$('#pbar').style.width=pct+'%';$('#lesson .pbar').setAttribute('aria-valuenow',pct);$('#hearts').innerHTML=ICON.heart+`<b>${LS.hearts}</b>`;$('#hearts').setAttribute('aria-label',`${LS.hearts} hearts left`)}
function resetBar(id){const pb=$(id);pb.style.transition='none';pb.style.width='0%';void pb.offsetWidth;pb.style.transition=''}
function startLesson(u,l){
  disposeAll();closeGuide();
  const steps=l.steps||buildMixedQuiz();
  LS={u,l,q:steps.slice(),i:0,hearts:5,total:steps.length,done:0,qs:steps.filter(s=>s.t!=='learn').length,wrong:new Set(),phase:'answer',ctrl:null,cur:null};
  $('#lesson').style.cssText=`--uc:var(--${u.c});--ucd:var(--${u.c}-d)`;
  resetBar('#pbar');show('lesson');renderStep();
}
const head=s=>`<h2 class="qh">${rich(s.q??s.title)}</h2>${s.code?`<pre class="py">${esc(s.code)}</pre>`:''}`;
const RENDER={
  learn(w,s){w.insertAdjacentHTML('beforeend',`<h2 class="qh">${rich(s.title)}</h2><div class="prose">${rich(s.body)}</div>`);if(s.w){const b=document.createElement('div');w.appendChild(b);W[s.w](b,s.wo||{})}setMain('Continue',true);return{}},
  task(w,s){w.insertAdjacentHTML('beforeend',`<h2 class="qh">${rich(s.title)}</h2><div class="prose">${rich(s.body)}</div>`);const b=document.createElement('div');w.appendChild(b);const h=W[s.w](b,s.wo||{});setMain('Check',true);return{check:()=>({ok:h.solved()})}},
  mc(w,s){
    const tf=s.t==='tf',opts=tf?[{t:'True',c:s.a===true},{t:'False',c:s.a===false}]:shuffle(s.o.map((t,i)=>({t,c:i===0})));
    w.insertAdjacentHTML('beforeend',`${head(s)}<div class="opts${tf?' tf':''}" role="radiogroup" aria-label="Answer choices"></div>`);
    const box=$('.opts',w);let sel=-1;
    opts.forEach((o,i)=>{const b=document.createElement('button');b.type='button';b.className='opt';b.setAttribute('role','radio');b.setAttribute('aria-checked','false');b.innerHTML=`<kbd>${i+1}</kbd><span>${rich(o.t)}</span>`;b.addEventListener('click',()=>pick(i));box.appendChild(b)});
    function pick(i){if(LS.phase!=='answer'||i>=opts.length)return;sel=i;$$('.opt',box).forEach((b,j)=>{b.classList.toggle('sel',j===i);b.setAttribute('aria-checked',j===i)});setMain('Check',true);sfx.tap()}
    setMain('Check',false);
    return{pick,check(){const ok=opts[sel].c;$$('.opt',box).forEach((b,j)=>{b.classList.remove('sel');if(opts[j].c)b.classList.add('right');else if(j===sel)b.classList.add('wrong');b.disabled=true});return{ok,ans:opts.find(o=>o.c).t}}};
  },
  num(w,s){
    w.insertAdjacentHTML('beforeend',`${head(s)}<div class="numrow"><button type="button" class="signbtn" aria-label="Switch the sign of your answer">±</button><input id="num-in" class="numin" type="text" inputmode="decimal" autocomplete="off" spellcheck="false" placeholder="Your answer" aria-label="Your answer">${s.suf?`<span class="suf">${s.suf}</span>`:''}</div>`);
    const inp=$('input',w),sign=$('.signbtn',w);inp.addEventListener('input',()=>setMain('Check',inp.value.trim()!==''));focusSoon(inp);setMain('Check',false);
    // Phone decimal keypads have no minus key, so ± flips the sign of whatever is typed.
    sign.addEventListener('click',()=>{if(LS.phase!=='answer')return;const v=inp.value.trim();inp.value=/^[-−]/.test(v)?v.replace(/^[-−]\s*/,''):'−'+v;inp.dispatchEvent(new Event('input'));inp.focus({preventScroll:true})});
    return{check(){sign.disabled=true;const v=parseFloat(inp.value.replace(/[,%\s]/g,'').replace('−','-'));const ok=Number.isFinite(v)&&Math.abs(v-s.a)<=(s.tol||0)+1e-9;inp.disabled=true;inp.classList.add(ok?'right':'wrong');return{ok,ans:s.a.toLocaleString('en-US').replace('-','−')+(s.suf||'')}}};
  },
  shape(w,s){
    w.insertAdjacentHTML('beforeend',`${head(s)}<div class="numrow"><input id="shape-in" class="numin shapein" type="text" autocomplete="off" autocapitalize="none" autocorrect="off" spellcheck="false" placeholder="(n, d)" aria-label="Shape, written like (n, d)"></div><p class="hint">Write it like <code>(500, 3)</code> or <code>(3,)</code>. Use <code>()</code> for a single number${s.err?' and <code>error</code> if the line fails':''}.</p>`);
    const inp=$('input',w);inp.addEventListener('input',()=>setMain('Check',inp.value.trim()!==''));focusSoon(inp);setMain('Check',false);
    return{check(){const ok=sameShape(inp.value,s.a);inp.disabled=true;inp.classList.add(ok?'right':'wrong');return{ok,ans:s.a}}};
  },
  match(w,s){
    w.insertAdjacentHTML('beforeend',`<h2 class="qh">${rich(s.q)}</h2><div class="match"></div>`);
    const grid=$('.match',w),L=shuffle(s.p.map((p,i)=>({t:p[0],i}))),Rt=shuffle(s.p.map((p,i)=>({t:p[1],i})));let sel=null,left=s.p.length;
    L.forEach((o,r)=>[o,Rt[r]].forEach((q,side)=>{const b=document.createElement('button');b.type='button';b.className='mitem';b.innerHTML=rich(q.t);b.dataset.i=q.i;b.dataset.side=side;b.addEventListener('click',()=>tap(b));grid.appendChild(b)}));
    function tap(b){
      if(LS.phase!=='answer'||b.classList.contains('ok'))return;
      if(!sel||sel.dataset.side===b.dataset.side){const was=sel;if(sel)sel.classList.remove('sel');sel=was===b?null:b;if(sel){sel.classList.add('sel');sfx.tap()}return}
      if(sel.dataset.i===b.dataset.i){[sel,b].forEach(x=>{x.classList.remove('sel');x.classList.add('ok');x.disabled=true});sel=null;left--;if(left)sfx.tap();else setTimeout(()=>reveal(true,s,{}),220)}
      else{const a=sel;sel=null;[a,b].forEach(x=>{x.classList.remove('sel');x.classList.add('bad')});LS.wrong.add(LS.q[LS.i]);sfx.bad();setTimeout(()=>[a,b].forEach(x=>x.classList.remove('bad')),450)}
    }
    setMain('Check',false);return{};
  },
  // Open-ended interview practice: answer first (typed or out loud), then compare with a model
  // answer and tick the rubric points you actually covered. No hearts at stake.
  explain(w,s){
    w.insertAdjacentHTML('beforeend',`${head(s)}${answerBox(s.secs)}<div class="xmodel" hidden><div class="fdiv"><span>Model answer</span></div><div class="prose">${rich(s.model)}</div>${rubricBox(s.rubric)}</div>`);
    const stop=startTimer($('.xtimer',w),s.secs);
    setMain('Show model answer',true);
    return{explain:true,
      reveal(){stop();$('.xmodel',w).hidden=false;highlight(w);sfx.flip();$('.xmodel',w).scrollIntoView({block:'start',behavior:reduced.matches?'auto':'smooth'})},
      score(){const b=$$('.rubric input',w);return b.filter(x=>x.checked).length/b.length}};
  }
};
RENDER.tf=RENDER.mc;
const KIND={recall:'Recall',derive:'Derive',diagnose:'Diagnose',defend:'Defend',code:'Code it'};
const answerBox=secs=>`<div class="xans"><label class="sr" for="x-ans">Your answer</label><textarea id="x-ans" rows="5" placeholder="Answer in writing or out loud, then compare with the model answer."></textarea><p class="xtimer" aria-hidden="true"></p></div>`;
const rubricBox=items=>`<div class="rubric"><p class="eyebrow">Tick what your answer covered</p>${items.map(r=>`<label class="rb"><input type="checkbox"><span>${rich(r)}</span></label>`).join('')}</div>`;
function startTimer(el,secs=90){
  const t0=Date.now(),tick=()=>{const s=Math.floor((Date.now()-t0)/1000);el.textContent=`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')} · aim for about ${secs} s`;el.classList.toggle('over',s>secs)};
  tick();const id=setInterval(tick,1000);const stop=()=>clearInterval(id);onDispose(stop);return stop;
}
function focusSoon(inp){if(matchMedia('(pointer: coarse)').matches)return;setTimeout(()=>{try{inp.focus({preventScroll:true})}catch(e){}},60)}
const KICK={learn:'New idea',mc:'Pick one',tf:'True or false?',num:'Work it out',shape:'Predict the shape',match:'Tap the matching pairs',task:'Try it yourself'};
const kicker=s=>s.t==='explain'?`${KIND[s.kind]} · say it or write it`:KICK[s.t];
function renderStep(){
  disposeAll();
  // vary() draws fresh numbers each time, so a missed question comes back as a new instance.
  const s=LS.cur=vary(LS.q[LS.i]),stg=$('#stage');stg.innerHTML='';LS.phase='answer';LS.ctrl=null;setBar('idle');updateTop();
  const w=document.createElement('div');w.className='step';w.innerHTML=`<p class="kicker">${LS.i>=LS.total?'Missed earlier · ':''}${kicker(s)}</p>`;stg.appendChild(w);
  LS.ctrl=RENDER[s.t](w,s);highlight(w);window.scrollTo(0,0);
}
function reveal(ok,s,r){
  LS.phase='reveal';
  if(ok){sfx.ok();LS.done++;setBar('ok',{t:PRAISE[Math.floor(Math.random()*PRAISE.length)],x:s.ex||''})}
  else{sfx.bad();setBar('bad',{t:'Not quite',x:(r.ans!=null?`<p>Answer: <b>${r.ans}</b></p>`:'')+(s.ex?`<p>${s.ex}</p>`:'')+missNote()});$('#hearts').classList.remove('hit');void $('#hearts').offsetWidth;$('#hearts').classList.add('hit')}
  setMain('Continue',true);updateTop();$('#mainbtn').focus({preventScroll:true});
}
const missNote=()=>'<p class="miss-note">It comes back at the end with new numbers. Get it right then and the lesson still passes.</p>';
function onMain(){
  const b=$('#mainbtn');if(b.disabled||!LS)return;const t=LS.q[LS.i],s=LS.cur;
  if(s.t==='explain'){
    if(LS.phase==='answer'){LS.ctrl.reveal();LS.phase='rubric';setMain('Continue',true);return}
    // Under 60% of the rubric is a miss: the prompt comes back at the end of the run, like any other.
    const covered=LS.ctrl.score();if(covered<0.6){LS.wrong.add(t);LS.q.push(t);sfx.tap();toast('This one comes back at the end of the lesson.');return next()}
    LS.done++;sfx.ok();return next();
  }
  if(LS.phase==='answer'){
    if(s.t==='learn'){LS.done++;return next()}
    if(!LS.ctrl.check)return;
    const r=LS.ctrl.check();
    if(r.ok)return reveal(true,s,r);
    LS.wrong.add(t);
    if(s.t==='task'){sfx.bad();LS.phase='retry';setBar('retry',{t:'Not yet',x:s.hint});setMain('Try again',true);return}
    LS.hearts--;LS.q.push(t);reveal(false,s,r);
  }else if(LS.phase==='retry'){LS.phase='answer';setBar('idle');setMain('Check',true)}
  else{if(LS.hearts<=0)return fail();next()}
}
function next(){LS.i++;if(LS.i>=LS.q.length)return finish();renderStep()}
function doneScreen({title,sub,xp,accLabel,acc,cards}){
  $('#done').classList.remove('fail');$('#done-mascot').innerHTML=mascot('mascot');
  $('#done-t').textContent=title;$('#done-sub').textContent=sub;$('#done-tiles').hidden=false;
  $('#d-xp').textContent='+'+xp;$('#d-acc-l').textContent=accLabel;$('#d-acc').textContent=acc;$('#d-st').textContent=st.streak+(st.streak===1?' day':' days');
  $('#done-cards').hidden=!cards;$('#done-cards').innerHTML=cards||'';
  $('#done-alt').hidden=true;$('#done-go').textContent='Continue';$('#done-go').onclick=()=>{LS=null;RS=null;show('path');renderPath()};
  show('done');sfx.win();confetti();$('#done-go').focus({preventScroll:true});
}
// Reaching the end means every question has been answered right: a miss comes back, with new
// numbers, until it is. That passes the lesson. Only the questions missed are ever asked again.
function finish(){
  disposeAll();const l=LS.l,missed=LS.wrong.size,perfect=!missed,xp=(l.id==='review'?20:10)+(perfect?5:0),firstTime=!st.done[l.id];
  st.done[l.id]=true;st.xp+=xp;bumpStreak();save();
  const acc=LS.qs?Math.round(100*(LS.qs-missed)/LS.qs):100,n=(l.cards||[]).length;
  doneScreen({title:perfect?'Flawless lesson!':'Lesson passed!',sub:perfect?`${l.title} · ${LS.u.title}`:`You redid the ${missed} question${missed===1?'':'s'} you missed. ${l.title} · ${LS.u.title}`,xp,accLabel:perfect?'Accuracy':'First try',acc:acc+'%',
    cards:firstTime&&n?`${ICON.cards}<span><b>${n} flashcards</b> added to your reviews. They come back right before you’d forget them.</span>`:''});
}
function fail(){
  disposeAll();const u=LS.u,l=LS.l;
  $('#done').classList.add('fail');$('#done-mascot').innerHTML=`<span style="color:var(--heart);display:block;width:96px;height:96px">${ICON.heart}</span>`;
  $('#done-t').textContent='Out of hearts';$('#done-sub').textContent='Every wrong answer comes back until you get it, so a second run goes faster. The guidebook has the key facts.';
  $('#done-tiles').hidden=true;$('#done-cards').hidden=true;$('#done-alt').hidden=false;$('#done-alt').onclick=()=>{LS=null;show('path');renderPath()};
  $('#done-go').textContent='Try again';$('#done-go').onclick=()=>startLesson(u,l);
  show('done');
}
function confetti(){
  if(reduced.matches)return;const c=document.createElement('canvas');c.className='confetti';document.body.appendChild(c);
  const ctx=c.getContext('2d'),dpr=window.devicePixelRatio||1,Wd=c.width=innerWidth*dpr,Hd=c.height=innerHeight*dpr,cs=getComputedStyle(document.documentElement);
  const cols=['--pca','--svd','--l2','--wd','--dr','--gold'].map(v=>cs.getPropertyValue(v).trim()||'#888');
  const ps=Array.from({length:150},()=>({x:Wd/2+(Math.random()-.5)*Wd*.25,y:Hd*.38,vx:(Math.random()-.5)*16*dpr,vy:(-Math.random()*15-5)*dpr,r:Math.random()*6,vr:(Math.random()-.5)*.35,w:(6+Math.random()*6)*dpr,h:(3+Math.random()*4)*dpr,c:cols[Math.floor(Math.random()*cols.length)]}));
  const t0=performance.now();(function fr(t){ctx.clearRect(0,0,Wd,Hd);ps.forEach(p=>{p.vy+=.38*dpr;p.vx*=.99;p.x+=p.vx;p.y+=p.vy;p.r+=p.vr;ctx.save();ctx.translate(p.x,p.y);ctx.rotate(p.r);ctx.fillStyle=p.c;ctx.fillRect(-p.w/2,-p.h/2,p.w,p.h);ctx.restore()});if(t-t0<2800)requestAnimationFrame(fr);else c.remove()})(t0);
}

/* ================= review session ================= */
let RS=null;
const kindOf=card=>{const s=rv[card.id];return!s?'new':s.phase==='review'?'review':'learn'};
function buildQueue(now){
  const counts=reviewCounts(now),learn=[],review=[],fresh=[];
  for(const card of unlockedCards()){const s=rv[card.id];if(!s)fresh.push(card);else if(s.due<=now)(s.phase==='review'?review:learn).push(card)}
  const byDue=(a,b)=>rv[a.id].due-rv[b.id].due;learn.sort(byDue);review.sort(byDue);
  // New cards are mixed in among reviews so a session doesn't end on a block of unfamiliar cards.
  const news=fresh.slice(0,counts.fresh),mixed=[];let ni=0;
  review.forEach((r,i)=>{mixed.push(r);if(ni<news.length&&i%3===2)mixed.push(news[ni++])});
  while(ni<news.length)mixed.push(news[ni++]);
  return[...learn,...mixed];
}
function startReview(){
  const queue=buildQueue(Date.now());
  if(!queue.length){toast('Nothing to review right now.');renderReviewPanel();return}
  RS={queue,reviewed:0,again:0,seen:new Set(),shown:false};
  resetBar('#rv-bar');show('review');renderCard();
}
function renderRvCounts(){
  const n={new:0,learn:0,review:0};RS.queue.forEach(c=>n[kindOf(c)]++);const cur=kindOf(RS.queue[0]);
  $('#rv-counts').innerHTML=`<b class="cn${cur==='new'?' cur':''}" title="New">${n.new}</b><b class="cl${cur==='learn'?' cur':''}" title="Learning">${n.learn}</b><b class="cr${cur==='review'?' cur':''}" title="Review">${n.review}</b>`;
  $('#rv-counts').setAttribute('aria-label',`${n.new} new, ${n.learn} learning, ${n.review} to review`);
}
function renderCard(){
  disposeAll();
  const card=vary(RS.queue[0]);RS.shown=false;
  $('#review').style.cssText=`--uc:var(--${card.u.c});--ucd:var(--${card.u.c}-d)`;
  $('#rv-kicker').textContent=card.kind?`${KIND[card.kind]} · ${card.u.title}`:`${card.u.title} · ${card.l.title}`;
  $('#rv-front').innerHTML=rich(card.front);$('#rv-back').innerHTML=rich(card.back);
  // Cards with a rubric are open-ended: answer first, then tick what you covered.
  const deep=$('#rv-deep'),rub=$('#rv-rubric');deep.hidden=!card.rubric;rub.hidden=true;
  if(card.rubric){
    deep.innerHTML=answerBox(card.secs);startTimer($('.xtimer',deep),card.secs);
    rub.innerHTML=rubricBox(card.rubric);
    $$('input',rub).forEach(x=>x.addEventListener('change',suggestGrade));
  }
  $$('#rv-grades .grade').forEach(b=>b.classList.remove('suggest'));
  $('#rv-back').hidden=true;$('#rv-hr').hidden=true;highlight($('#rv-card'));
  $('#rv-show').hidden=false;$('#rv-grades').hidden=true;
  const pct=Math.round(RS.reviewed/(RS.reviewed+RS.queue.length)*100);$('#rv-bar').style.width=pct+'%';$('#review .pbar').setAttribute('aria-valuenow',pct);
  renderRvCounts();window.scrollTo(0,0);$('#rv-show').focus({preventScroll:true});
}
// Rubric coverage suggests a grade: under half is Again, under 80% Hard, otherwise Good.
function suggestGrade(){
  const boxes=$$('#rv-rubric input'),covered=boxes.filter(x=>x.checked).length/boxes.length;
  const g=covered<0.5?1:covered<0.8?2:3;
  $$('#rv-grades .grade').forEach(b=>b.classList.toggle('suggest',+b.dataset.g===g));
}
function showAnswer(){
  if(!RS||RS.shown)return;RS.shown=true;disposeAll();
  $('#rv-back').hidden=false;$('#rv-hr').hidden=false;$('#rv-show').hidden=true;$('#rv-grades').hidden=false;
  if(RS.queue[0].rubric){$('#rv-rubric').hidden=false;suggestGrade()}
  const p=preview(rv[RS.queue[0].id]||null);
  $$('#rv-grades .grade').forEach((b,i)=>{$('span',b).textContent=formatInterval(p[i]);b.setAttribute('aria-label',`${$('b',b).textContent}: next in ${formatInterval(p[i])}`)});
  sfx.flip();$('#rv-grades .g3').focus({preventScroll:true});
  $('#rv-hr').scrollIntoView({block:'start',behavior:reduced.matches?'auto':'smooth'});
}
function grade(g){
  if(!RS||!RS.shown)return;
  const card=RS.queue.shift(),now=Date.now(),next=schedule(rv[card.id]||null,g,now);
  rv[card.id]=next;queueReview(card.id,next);
  RS.reviewed++;RS.seen.add(card.id);if(g===1)RS.again++;
  if(next.phase!=='review'&&next.due-now<=LEARN_AHEAD)RS.queue.push(card);
  if(g===1)sfx.bad();else sfx.tap();
  if(RS.queue.length)renderCard();else finishReview();
}
function finishReview(){
  const cards=RS.seen.size,recall=Math.round(100*(RS.reviewed-RS.again)/RS.reviewed),c=reviewCounts();
  st.xp+=cards;bumpStreak();save();
  const nextTxt=Number.isFinite(c.nextDue)?`Next card ${relTime(c.nextDue)}.`:'';
  doneScreen({title:'Reviews done!',sub:`${cards} card${cards===1?'':'s'} reviewed. ${nextTxt}`,xp:cards,accLabel:'Recalled',acc:recall+'%',cards:''});
}

/* ================= wiring ================= */
$('#logo-s').innerHTML=mascot();$('#logo-l').innerHTML=mascot();
$('#quit').innerHTML=ICON.x;$('#guide-x').innerHTML=ICON.x;$('#rv-quit').innerHTML=ICON.x;
$('#quit').addEventListener('click',()=>{disposeAll();LS=null;show('path');renderPath()});
$('#rv-quit').addEventListener('click',()=>{RS=null;show('path');renderPath()});
$('#mainbtn').addEventListener('click',onMain);
$('#rv-show').addEventListener('click',showAnswer);
$$('#rv-grades .grade').forEach(b=>b.addEventListener('click',()=>grade(+b.dataset.g)));
$('#rp-go').addEventListener('click',startReview);
$('#rv-top').addEventListener('click',()=>{if(reviewCounts().due)startReview();else{$('#revpanel').scrollIntoView({behavior:reduced.matches?'auto':'smooth',block:'center'});toast('All caught up. New cards arrive as you finish lessons.')}});
$('#guide-x').addEventListener('click',closeGuide);
$('#guide').addEventListener('click',e=>{if(e.target.id==='guide')closeGuide()});
$('#mute').addEventListener('click',()=>{st.muted=!st.muted;applyPrefs();save();renderStats();sfx.tap()});
let resetArm=0;
$('#reset').addEventListener('click',e=>{const b=e.currentTarget;if(!resetArm){resetArm=setTimeout(()=>{resetArm=0;b.textContent='Reset progress';b.classList.remove('warn')},3500);b.textContent='Tap again to erase lessons and reviews';b.classList.add('warn');return}clearTimeout(resetArm);resetArm=0;b.textContent='Reset progress';b.classList.remove('warn');resetProgress()});
document.addEventListener('keydown',e=>{
  if(!$('#guide').hidden){if(e.key==='Escape')closeGuide();return}
  if(!$('#review').hidden&&RS){
    if((e.key===' '||e.key==='Enter')&&!RS.shown){if(e.target.tagName==='BUTTON'&&e.target!==$('#rv-show'))return;e.preventDefault();showAnswer();return}
    if(/^[1-4]$/.test(e.key)&&RS.shown){e.preventDefault();grade(+e.key)}
    return;
  }
  if(e.key!=='Enter'&&!/^[1-4]$/.test(e.key))return;
  if(!$('#done').hidden){if(e.key==='Enter'&&e.target.tagName!=='BUTTON'){e.preventDefault();$('#done-go').click()}return}
  if($('#lesson').hidden||!LS)return;
  if(e.key==='Enter'){if(e.target.tagName==='BUTTON')return;e.preventDefault();onMain()}
  else if(LS.ctrl&&LS.ctrl.pick&&e.target.tagName!=='INPUT')LS.ctrl.pick(+e.key-1);
});
setInterval(()=>{if(me&&!$('#path').hidden)renderReviewPanel()},30000);
async function resetProgress(){
  clearTimeout(saveTimer);dirty=false;clearTimeout(rvTimer);outbox={};saveOutbox();
  try{st=await api('DELETE','/api/progress');rv={};applyPrefs();renderPath();setSync('saved')}
  catch(e){if(!(e instanceof SignedOut))toast(e.message)}
}

/* ================= sign in ================= */
function showLogin(msg){
  disposeAll();LS=null;RS=null;closeGuide();show('login');
  const err=$('#login-err');err.hidden=!msg;err.textContent=msg||'';
  focusSoon($('#login-user'));
}
function onSignedOut(expired){
  // Keep anything not yet saved so it can be sent once the same person signs back in.
  if(dirty&&me)unsynced={user:me,progress:st};
  clearTimeout(saveTimer);clearTimeout(rvTimer);dirty=false;me=null;
  showLogin(expired?'Your session ended. Sign in again to keep going.':'');
}
async function enter(username){
  me=username;$('#who').textContent=me;
  if(unsynced&&unsynced.user===me){await api('PUT','/api/progress',unsynced.progress).catch(()=>{});unsynced=null}
  const[p,r]=await Promise.all([api('GET','/api/progress'),api('GET','/api/reviews')]);
  st=p;rv=r.cards||{};applyPrefs();
  // Anything graded offline or before a tab closed wins if it is newer than what the server has.
  outbox=loadOutbox();
  for(const[id,s]of Object.entries(outbox))if(!rv[id]||s.last>=rv[id].last)rv[id]=s;
  show('path');renderPath();setSync('saved');flushReviews();
}
$('#login-form').addEventListener('submit',async e=>{
  e.preventDefault();
  const u=$('#login-user').value.trim(),p=$('#login-pass').value,btn=$('#login-go'),err=$('#login-err');
  if(!u||!p){err.textContent='Enter your username and password.';err.hidden=false;return}
  btn.disabled=true;btn.textContent='Signing in…';err.hidden=true;
  try{const r=await api('POST','/api/login',{username:u,password:p});$('#login-pass').value='';await enter(r.username)}
  catch(e2){err.textContent=e2.message;err.hidden=false}
  finally{btn.disabled=false;btn.textContent='Sign in'}
});
$('#signout').addEventListener('click',async()=>{
  clearTimeout(saveTimer);await flush();await flushReviews();
  await api('POST','/api/logout').catch(()=>{});
  me=null;st=blankProgress();rv={};outbox={};showLogin();
});
window.addEventListener('pagehide',()=>{
  const send=(path,body)=>fetch(path,{method:'PUT',keepalive:true,credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).catch(()=>{});
  if(dirty&&me)send('/api/progress',st);
  if(me&&Object.keys(outbox).length)send('/api/reviews',{cards:outbox});
});
if('serviceWorker'in navigator&&(location.protocol==='https:'||['localhost','127.0.0.1'].includes(location.hostname))){
  navigator.serviceWorker.register('/sw.js').catch(()=>{});
}
(async()=>{
  let r=null;
  try{r=await fetch('/api/me',{credentials:'same-origin'})}catch(e){}
  if(r&&r.ok){try{await enter((await r.json()).username);return}catch(e){}}
  showLogin(r?'':'Couldn’t reach the server. Check your connection, then reload.');
})();
