import { chromium } from 'playwright';
const b = await chromium.launch(); const p = await b.newPage(); const errs=[]; p.on('pageerror',e=>errs.push(e.message));
await p.goto('file://'+process.cwd()+'/EkMaap_demo.html'); await p.waitForTimeout(1000);
const r = await p.evaluate(async()=>{ let t=0,m=0,ok=0; for(const sc of ['mixed','touching']) for(let s=1;s<=5;s++){ const c=await EK.loadSynthetic(sc,s); t+=c.truth; m+=c.matched; ok+=c.rows.filter(x=>x.o&&x.o.label===x.g.label).length; }
  document.getElementById('lotId').value='T1'; const rec=await EK.renderReport(); const txt=JSON.stringify(rec);
  const good=await EK.verify(txt, EK.state.imageBytes); const t2=JSON.parse(txt); t2.summary.pct['Grade A']+=5; const bad=await EK.verify(JSON.stringify(t2), EK.state.imageBytes);
  return {t,m,ok,good:good.recordOK&&good.imageOK,tamperCaught:!bad.recordOK}; });
console.log(r, 'errors', errs); await b.close();
