"""Self-contained HTML inspection artifact; no server or dependencies required."""

import json
from decimal import Decimal


def _display_decimal(value):
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def render_report(rows, registry, summary):
    data = {"rows": rows, "entities": registry["entities"], "summary": summary}
    payload = json.dumps(data, ensure_ascii=False, default=_display_decimal).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entity recognition and linking — inspection</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;color:#17212b;background:#f7f8fa;margin:24px}
h1{font-size:25px;margin:0 0 8px}.note{color:#536171;max-width:1000px}.stats{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}
.stat{background:white;border:1px solid #dbe1e8;border-radius:8px;padding:12px 18px}.stat strong{font-size:22px;display:block}
.controls{display:flex;gap:12px;flex-wrap:wrap;position:sticky;top:0;background:#f7f8fa;padding:12px 0;z-index:1}
input,select,button{font:inherit;padding:8px;border:1px solid #bdc8d3;border-radius:5px;background:white}input{min-width:280px;flex:1}
table{width:100%;border-collapse:collapse;table-layout:fixed;background:white}th,td{border:1px solid #dbe1e8;padding:12px;vertical-align:top;text-align:left;overflow-wrap:anywhere}
th{background:#edf1f5}th:first-child{width:95px}th:last-child{width:190px}.original,.normalized{white-space:pre-wrap}.normalized{font-family:ui-monospace,monospace}
mark{background:#d9f0e6;color:#104c34;padding:1px 2px}.ambiguous{background:#ffe5c0;color:#704106}.muted{color:#697787;font-size:13px}
mark.value{background:#dceaff;color:#193f77}
#load{display:block;margin:20px auto}details{margin:20px 0}code{background:#eaf0f6;padding:2px 5px}
</style>
<h1>Entity recognition and linking</h1>
<p class="note">Only lexicon names and aliases are linked. Numeric IDs identify entities; they carry no numerical meaning.
Green highlights show linked spans. Orange spans have multiple candidates and remain unchanged.
Text outside annotated spans stays unchanged. Pronouns and names outside the lexicon remain unresolved.</p>
<p id="value-note" class="note" hidden>Blue spans are typed values. Their tokens point to structured annotations; value IDs are local to each row and kind. Qualifiers describe wording, not world truth. Ranges and unresolved numbers remain visible for review.</p>
<div id="stats" class="stats"></div>
<div class="controls"><input id="search" aria-label="Search" placeholder="Search original text, normalized text, row ID or entity name…">
<select id="status" aria-label="Mention status"><option value="all">All rows</option><option value="linked">Has linked mentions</option><option value="none">No linked mentions</option><option value="typed">Has typed values</option><option value="no_annotations">No linked entity or typed value</option><option value="suppressed">Suppressed alias candidates</option><option value="ambiguous">Ambiguous mentions</option></select>
<select id="entity" aria-label="Entity"><option value="">All entities</option></select>
<select id="kind" aria-label="Value type" hidden><option value="">All value types</option><option value="money">Money</option><option value="quantity">Quantity</option><option value="percent">Percent</option><option value="date">Date</option><option value="duration">Duration</option><option value="version">Version</option><option value="unresolved">Unresolved values</option></select></div>
<p id="count" class="muted"></p>
<table><thead><tr><th>Row / date</th><th>Original sentence</th><th>Normalized sentence</th><th>Entity links</th></tr></thead><tbody id="rows"></tbody></table>
<button id="load" type="button">Show 200 more rows</button>
<details><summary>How to interpret this report</summary><p>This is a deterministic preprocessing audit, not a truth classifier or a measured NER accuracy score.
The report covers the complete practice dataset. A row with no linked mention may use a pronoun, an unlisted short name, or no named entity at all.</p>
<p>Offsets in the CSV are zero-based Python character offsets, with the end excluded. The registry preserves IDs when the lexicon is reordered or expanded.</p></details>
<script type="application/json" id="data">""" + payload + """</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), byId=new Map(data.entities.map(e=>[e.id,e]));
const status=document.getElementById('status'),entity=document.getElementById('entity'),search=document.getElementById('search');
const summary=data.summary;
const kind=document.getElementById('kind');if(summary.value_mentions!==undefined){kind.hidden=false;document.getElementById('value-note').hidden=false;document.querySelector('h1').textContent='Entity and typed-value inspection';}
for(const [value,label] of [[summary.rows,'rows'],[summary.rows_with_links,'rows with links'],[summary.linked_mentions,'linked mentions'],[summary.entities_seen,'entities seen'],[summary.rows_without_links,'rows without links']]){
const box=document.createElement('div');box.className='stat';const strong=document.createElement('strong');strong.textContent=value.toLocaleString();box.append(strong,document.createTextNode(label));document.getElementById('stats').append(box);}
if(summary.value_mentions!==undefined){const box=document.createElement('div');box.className='stat';const strong=document.createElement('strong');strong.textContent=summary.value_mentions.toLocaleString();box.append(strong,document.createTextNode('typed values'));document.getElementById('stats').append(box);}
for(const e of data.entities.filter(e=>e.active)){const opt=document.createElement('option');opt.value=e.id;opt.textContent=e.id+' · '+e.name;entity.append(opt);}
let limit=200,filtered=[];
function cell(tr,text,css){const td=document.createElement('td');if(css)td.className=css;if(text!==null)td.textContent=text;tr.append(td);return td;}
function drawOriginal(td,row){let pos=0;const chars=Array.from(row.text),spans=[...row.mentions.map(m=>({...m,ambiguous:false})),...row.ambiguous_mentions.map(m=>({...m,ambiguous:true})),...(row.values||[]).map(m=>({...m,isValue:true}))].sort((a,b)=>a.start-b.start);
for(const m of spans){td.append(document.createTextNode(chars.slice(pos,m.start).join('')));const mark=document.createElement('mark');if(m.ambiguous)mark.className='ambiguous';if(m.isValue)mark.className='value';mark.textContent=chars.slice(m.start,m.end).join('');mark.title=m.isValue?JSON.stringify(m):m.ambiguous?'Candidates: '+m.candidate_entity_ids.join(', '):m.entity_id+' · '+m.canonical_name;td.append(mark);pos=m.end;}td.append(document.createTextNode(chars.slice(pos).join('')));}
function render(){const body=document.getElementById('rows');body.replaceChildren();for(const row of filtered.slice(0,limit)){
const tr=document.createElement('tr');cell(tr,row.id+' · '+row.date,'muted');drawOriginal(cell(tr,null,'original'),row);cell(tr,row.normalized_text,'normalized');
const links=cell(tr,null);for(const m of row.mentions){const div=document.createElement('div');div.textContent=m.entity_id+' · '+m.canonical_name;div.title='Original ['+m.start+', '+m.end+'), normalized ['+m.normalized_start+', '+m.normalized_end+')';links.append(div);}
for(const m of row.ambiguous_mentions){const div=document.createElement('div');div.className='ambiguous';div.textContent=m.surface+' → ? '+m.candidate_entity_ids.join(', ');links.append(div);}for(const m of row.suppressed_mentions||[]){const div=document.createElement('div');div.className='muted';div.textContent='Not linked: '+m.surface+' · '+m.reason;div.title='Candidate IDs: '+m.candidate_entity_ids.join(', ');links.append(div);}if(!links.childNodes.length)links.textContent='No lexicon link';body.append(tr);}
for(let i=0;i<Math.min(limit,filtered.length);i++){const row=filtered[i],links=body.children[i].lastElementChild;for(const value of row.values||[]){const div=document.createElement('div');div.textContent=value.token+' · '+String(value.value)+(value.unit?' '+value.unit:'')+(value.currency?' '+value.currency:'')+(value.per?'/'+value.per:'')+(value.op!=='eq'?' ['+value.op+']':'')+(value.approximate?' [approx]':'')+(value.negated_prefix?' [negation nearby]':'');div.title=JSON.stringify(value);links.append(div);}for(const value of row.unresolved_values||[]){const div=document.createElement('div');div.className='muted';div.textContent='Unresolved: '+value.surface+' · '+value.reason;links.append(div);}}
document.getElementById('count').textContent='Showing '+Math.min(limit,filtered.length)+' of '+filtered.length+' matching rows';document.getElementById('load').hidden=limit>=filtered.length;}
function filter(){const q=search.value.toLocaleLowerCase();filtered=data.rows.filter(row=>{
if(status.value==='linked'&&!row.mentions.length)return false;if(status.value==='none'&&row.mentions.length)return false;if(status.value==='ambiguous'&&!row.ambiguous_mentions.length)return false;
if(status.value==='typed'&&!(row.values||[]).length)return false;if(status.value==='no_annotations'&&(row.mentions.length||(row.values||[]).length))return false;
if(status.value==='suppressed'&&!(row.suppressed_mentions||[]).length)return false;
if(entity.value&&!row.mentions.some(m=>String(m.entity_id)===entity.value)&&!row.ambiguous_mentions.some(m=>m.candidate_entity_ids.includes(Number(entity.value))))return false;
if(kind.value==='unresolved'&&!(row.unresolved_values||[]).length)return false;if(kind.value&&kind.value!=='unresolved'&&!(row.values||[]).some(v=>v.kind===kind.value))return false;
return !q||[row.id,row.text,row.normalized_text,...row.mentions.map(m=>m.canonical_name),JSON.stringify(row.values||[])].join(' ').toLocaleLowerCase().includes(q);});limit=200;render();}
search.addEventListener('input',filter);status.addEventListener('change',filter);entity.addEventListener('change',filter);kind.addEventListener('change',filter);document.getElementById('load').addEventListener('click',()=>{limit+=200;render();});filter();
</script></html>"""
