import pw from '/home/claude/.npm-global/lib/node_modules/playwright/index.js'; const {chromium}=pw;
import fs from 'fs';
const b = await chromium.launch({executablePath:'/opt/pw-browsers/chromium'});
const p = await b.newPage({viewport:{width:1360,height:900}});
const errs=[]; p.on('pageerror',e=>errs.push(e.message));
await p.goto('file://'+process.cwd()+'/EkMaap_demo.html');
const out={batch:{},knn:null,report:null};
const SEEDS=20;
for (const scn of ['mixed','touching']) {
  const agg={images:0,truth:0,matched:0,missed:0,extra:0,absErrSum:0,errSum:0,sized:0,maxErr:0,condOK:0,labelOK:0,review:0,cardFound:0,ms:[]};
  for (let s=1;s<=SEEDS;s++){
    const r = await p.evaluate(async ([scn,s])=>{const c=await EK.loadSynthetic(scn,s);
      const errs=c.rows.filter(r=>r.o&&r.o.diameter_mm!=null).map(r=>r.o.diameter_mm-r.g.d_mm);
      return {truth:c.truth,matched:c.matched,missed:c.missed,extra:c.extra,errs,
        condOK:c.rows.filter(r=>r.o&&r.o.cond===r.g.cond).length,labelOK:c.rows.filter(r=>r.o&&r.o.label===r.g.label).length,
        review:c.review,card:c.cardFound,ms:EK.state.result.ms,
        misses:c.rows.filter(r=>!r.o||r.o.cond!==r.g.cond||r.o.label!==r.g.label).map(r=>({t:r.g.cond,tl:r.g.label,d:r.g.d_mm,got:r.o?r.o.label:'MISS',why:r.o?r.o.reasons:[]}))};},[scn,s]);
    agg.images++; agg.truth+=r.truth; agg.matched+=r.matched; agg.missed+=r.missed; agg.extra+=r.extra;
    for(const e of r.errs){agg.absErrSum+=Math.abs(e);agg.errSum+=e;agg.sized++;agg.maxErr=Math.max(agg.maxErr,Math.abs(e));}
    agg.condOK+=r.condOK; agg.labelOK+=r.labelOK; agg.review+=r.review; agg.cardFound+=r.card?1:0; agg.ms.push(r.ms);
    if(r.misses.length) console.log(scn,s,JSON.stringify(r.misses));
  }
  agg.sizeMAE=agg.absErrSum/agg.sized; agg.sizeBias=agg.errSum/agg.sized; agg.medianMs=agg.ms.sort((a,b)=>a-b)[agg.ms.length>>1]; delete agg.ms;
  out.batch[scn]=agg; console.log(scn, JSON.stringify(agg));
}
// kNN: label onions from seeds 1..10 (mixed) with truth, then leave-one-photo-out evaluation
out.knn = await p.evaluate(async ()=>{ EK.state.examples=[]; for(let s=1;s<=10;s++){await EK.loadSynthetic('mixed',s); EK.teachFromTruth();}
  const r=EK.evaluate(); return {n:r.n,tested:r.tested,scheme:r.scheme,accuracy:r.accuracy,heuristicAgreement:r.heuristicAgreement,per:r.per,synthetic:r.synthetic};});
console.log('knn', JSON.stringify(out.knn));
// With kNN active, grade unseen seeds 21..30
out.knnUnseen = await p.evaluate(async ()=>{let ok=0,n=0,rev=0; for(let s=21;s<=30;s++){const c=await EK.loadSynthetic('mixed',s); for(const r of c.rows){if(!r.o)continue;n++; if(r.o.cond===r.g.cond)ok++; if(r.o.label==='Q')rev++;}} return {n,condOK:ok,review:rev,mode:EK.state.result.onions[0].source};});
console.log('knnUnseen', JSON.stringify(out.knnUnseen));
// report + verify + tamper
out.report = await p.evaluate(async ()=>{ EK.state.examples=[]; await EK.loadSynthetic('mixed',3);
  document.getElementById('lotId').value='TEST-001'; document.getElementById('lotCentre').value='Demo centre';
  const rec = await EK.renderReport(); const txt=JSON.stringify(rec);
  const img = EK.state.imageBytes;
  const good = await EK.verify(txt, img);
  const t = JSON.parse(txt); t.summary.pct['Grade A'] += 10; const bad = await EK.verify(JSON.stringify(t), img);
  const imgT = new Uint8Array(img.slice(0)); imgT[imgT.length-10]^=1; const badImg = await EK.verify(txt, imgT.buffer);
  return {hash:rec.record_sha256, good:[good.recordOK,good.imageOK], tamperedNumber:[bad.recordOK,bad.imageOK], tamperedImage:[badImg.recordOK,badImg.imageOK]};});
console.log('report', JSON.stringify(out.report));
out.errors=errs;
fs.writeFileSync('test_results.json', JSON.stringify(out,null,1));
console.log('errors',errs);
await b.close();
