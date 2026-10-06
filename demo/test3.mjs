import { chromium } from 'playwright';
import fs from 'fs';
const b = await chromium.launch();
const p = await b.newPage();
await p.goto('file://'+process.cwd()+'/EkMaap_demo.html');
// size error by condition (mixed, 20 seeds)
const byCond = await p.evaluate(async ()=>{const acc={}; for(let s=1;s<=20;s++){const c=await EK.loadSynthetic('mixed',s); for(const r of c.rows){ if(!r.o) continue; const e=r.o.diameter_mm-r.g.d_mm; (acc[r.g.cond]=acc[r.g.cond]||[]).push(e);} }
  const o={}; for(const [k,v] of Object.entries(acc)) o[k]={n:v.length, mae:v.reduce((s,e)=>s+Math.abs(e),0)/v.length, bias:v.reduce((s,e)=>s+e,0)/v.length, max:Math.max(...v.map(Math.abs))}; return o;});
console.log('size by condition', JSON.stringify(byCond));
// degradations
const conds = [['none',''],['blur 2px','blur(2px)'],['dim light (70%)','brightness(0.7)'],['warm light','sepia(0.35)'],['bright (130%)','brightness(1.3)']];
const stress = {};
for (const [name, filter] of conds) {
  const r = await p.evaluate(async ([filter])=>{ let agg={truth:0,matched:0,extra:0,absErr:0,sized:0,condOK:0,labelOK:0,review:0,card:0};
    for (let s=1;s<=10;s++){ const syn = EK.makeSynthetic('mixed', s);
      const c2=document.createElement('canvas'); c2.width=syn.canvas.width; c2.height=syn.canvas.height; const x=c2.getContext('2d'); x.filter=filter||'none'; x.drawImage(syn.canvas,0,0);
      const blob = await new Promise(r=>c2.toBlob(r,'image/jpeg',0.7)); const f=new File([blob],'t.jpg',{type:'image/jpeg'});
      document.getElementById('inFile'); await (async()=>{ const buf=await f.arrayBuffer(); const bmp=await createImageBitmap(blob); })();
      // use the app's own loader
      const dt = new DataTransfer(); dt.items.add(f); const inp=document.getElementById('inFile'); inp.files=dt.files; inp.dispatchEvent(new Event('change'));
      await new Promise(r=>setTimeout(r,50)); for(let k=0;k<100 && (!EK.state.result || EK.state.fileName!=='t.jpg' || EK.state.lastCheck===EK.state.result);k++) await new Promise(r=>setTimeout(r,50));
      EK.state.lastCheck = EK.state.result; EK.state.synthetic = syn; const c = EK.compareGT();
      agg.truth+=c.truth; agg.matched+=c.matched; agg.extra+=c.extra; agg.condOK+=c.rows.filter(r=>r.o&&r.o.cond===r.g.cond).length; agg.labelOK+=c.rows.filter(r=>r.o&&r.o.label===r.g.label).length; agg.review+=c.review; agg.card+=c.cardFound?1:0;
      for(const r of c.rows) if(r.o&&r.o.diameter_mm!=null){agg.absErr+=Math.abs(r.o.diameter_mm-r.g.d_mm); agg.sized++;}
    } agg.sizeMAE=agg.sized?agg.absErr/agg.sized:null; return agg; }, [filter]);
  stress[name]=r; console.log(name, JSON.stringify(r));
}
// no size card
const nocard = await p.evaluate(async ()=>{ const syn=EK.makeSynthetic('mixed',1); const x=syn.canvas.getContext('2d'); x.fillStyle='rgb(240,238,232)'; x.fillRect(40,40,300,300);
  const blob=await new Promise(r=>syn.canvas.toBlob(r,'image/png')); const f=new File([blob],'nocard.png',{type:'image/png'});
  const dt=new DataTransfer(); dt.items.add(f); const inp=document.getElementById('inFile'); inp.files=dt.files; inp.dispatchEvent(new Event('change'));
  for(let k=0;k<100 && EK.state.fileName!=='nocard.png';k++) await new Promise(r=>setTimeout(r,50)); await new Promise(r=>setTimeout(r,300));
  const s=EK.state.result.summary; return {scale:EK.state.result.scaleSource, count:s.count, review:s.Review, gradeA:s['Grade A']};});
console.log('no card', JSON.stringify(nocard));
const prev = JSON.parse(fs.readFileSync('test_results.json'));
prev.sizeByCondition = byCond; prev.stress = stress; prev.noCard = nocard;
fs.writeFileSync('test_results.json', JSON.stringify(prev,null,1));
await b.close();
