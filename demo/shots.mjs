import { chromium } from 'playwright';
const b = await chromium.launch();
const errs=[];
for (const [name,vp,lg] of [['desk-en',{width:1366,height:900},'en'],['phone-hi',{width:390,height:844},'hi'],['phone-ta',{width:360,height:780},'ta'],['desk-mr',{width:1366,height:900},'mr'],['phone-te',{width:390,height:844},'te'],['phone-gu',{width:390,height:844},'gu'],['phone-kn',{width:360,height:780},'kn']]) {
  const p = await b.newPage({viewport:vp, deviceScaleFactor:1.5});
  p.on('pageerror',e=>errs.push(name+': '+e.message)); p.on('console',m=>{if(m.type()==='error'&&!/ERR_|net::/.test(m.text()))errs.push(name+': '+m.text())});
  await p.goto('file://'+process.cwd()+'/EkMaap_demo.html'); await p.waitForTimeout(1200);
  await p.evaluate(l=>EK.setLang(l), lg); await p.waitForTimeout(300);
  const ov=[]; const chk=async t=>ov.push(t+':'+await p.evaluate(()=>document.documentElement.scrollWidth-innerWidth));
  await p.screenshot({path:`s_${name}_home.png`, fullPage:true}); await chk('home');
  if (name.startsWith('phone')) { await p.click('#menuBtn'); await p.waitForTimeout(150); await p.screenshot({path:`s_${name}_menu.png`}); await p.click('#nav button[data-t="grade"]'); }
  else await p.click('#nav button[data-t="grade"]');
  await p.waitForTimeout(200);
  await p.screenshot({path:`s_${name}_grade.png`, fullPage:true}); await chk('grade');
  await p.click('#onionList [data-id="2"]'); await p.waitForTimeout(300);
  await p.screenshot({path:`s_${name}_sheet.png`}); await chk('sheet');
  await p.click('#sheetBody [data-teach="rotten"]'); await p.waitForTimeout(200); await p.click('#sheetClose');
  await p.click('#btnNext'); await p.fill('#lotId','LSG-0142'); await p.fill('#lotCentre','Lasalgaon APMC'); await p.fill('#lotOfficer','R. Patil'); await p.click('#btnReport'); await p.waitForTimeout(600);
  await p.screenshot({path:`s_${name}_report.png`, fullPage:true}); await chk('report');
  await p.evaluate(async()=>{ for(let s=3;s<=5;s++){await EK.loadSynthetic('mixed',s); EK.teachFromTruth();} await EK.loadSynthetic('mixed',2); EK.regrade(); EK.show('settings'); });
  await p.click('#btnEval'); await p.waitForTimeout(200);
  await p.screenshot({path:`s_${name}_settings.png`, fullPage:true}); await chk('settings');
  await p.evaluate(()=>EK.show('help')); await chk('help');
  await p.click('[data-fs="21"]'); await p.evaluate(()=>EK.show('grade')); await p.waitForTimeout(200); await chk('grade-A+');
  if(name==='phone-hi') await p.screenshot({path:`s_${name}_grade_big.png`});
  console.log(name, ov.join(' '));
  await p.close();
}
console.log('errors',errs); await b.close();
