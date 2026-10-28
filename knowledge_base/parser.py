"""Conservative, training-template based numeric assertion parser."""
from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation


_NUM = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)(?!\.\d)")
_OP = [(re.compile(r"\b(at least|no less than)\b", re.I), "ge"),
       (re.compile(r"\b(up to|at most|no more than)\b", re.I), "le"),
       (re.compile(r"\b(over|more than|above|greater than)\b", re.I), "gt"),
       (re.compile(r"\b(under|less than|below)\b", re.I), "lt")]
_CURRENCIES = {"€": "EUR", "$": "USD", "£": "GBP", "eur": "EUR", "usd": "USD", "gbp": "GBP"}
_UNSAFE = re.compile(r"\b(?:not|never|formerly|previously|used to|would|will|planned|planning|expected|expect|target|goal|next|last (?:year|quarter|month|week|day)|in \d{4}|on \w+ \d{1,2}|since|until|before|after|can't|cannot|doesn't|don't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b", re.I)


def _unsafe_context(text):
    if _UNSAFE.search(text):
        return True
    # Explicit calendar-month references are temporal scope even if no year is given.
    return bool(re.search(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", text, re.I))


def _norm_value(v):
    if isinstance(v, dict):
        d = dict(v)
        if "value" in d and isinstance(d["value"], (int, float, str)):
            try:
                d["value"] = str(Decimal(str(d["value"])).normalize())
            except InvalidOperation:
                pass
        return d
    return v


def _subjects(lexicon):
    src = lexicon.get("subjects", []) if isinstance(lexicon, dict) else []
    result = []
    for item in src:
        if isinstance(item, dict) and item.get("name"):
            for alias in [item["name"], *item.get("aliases", [])]:
                if alias:
                    result.append((str(alias), str(item["name"])))
    return sorted(result, key=lambda x: (-len(x[0]), x[0].casefold()))


def _matches(text, subjects):
    found=[]
    occupied=[]
    for alias, canonical in subjects:
        pat=re.compile(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",re.I)
        for m in pat.finditer(text):
            if not any(m.start()<b and m.end()>a for a,b in occupied):
                occupied.append((m.start(),m.end())); found.append((m.start(),m.end(),canonical,alias))
    return sorted(found)


def _number_spans(text):
    return list(_NUM.finditer(text))


def _value_for(text, m):
    raw=m.group().replace(",", "")
    try: number=str(Decimal(raw).normalize())
    except InvalidOperation: return None
    left=text[max(0,m.start()-24):m.start()]
    right=text[m.end():m.end()+32]
    op="eq"
    for rx, val in _OP:
        hit=rx.search(left)
        if hit and not re.search(r"\w",left[hit.end():]): op=val
    if re.match(r"\s*%",right):
        v={"kind":"percent","value":number}
    else:
        cur=None
        for sym, code in _CURRENCIES.items():
            if left.endswith(sym) or re.match(r"\s*"+re.escape(sym),right,re.I): cur=code; break
        if cur:
            period=None
            pm=re.match(r"\s*(?:(?:a|per)\s+)?(day|week|month|year)\b|\s*/\s*(day|week|month|year)\b",right,re.I)
            if pm: period=(pm.group(1) or pm.group(2)).lower()
            v={"kind":"money","value":number,"currency":cur}
            if period: v["per"]=period
        else:
            unitm=re.match(r"\s+([A-Za-z]+)\b",right)
            unit=unitm.group(1).lower() if unitm else ""
            if unit in ("day","days","week","weeks","month","months","year","years","hour","hours","minute","minutes"):
                unit={"days":"day","weeks":"week","months":"month","years":"year","hours":"hour","minutes":"minute"}.get(unit,unit)
                unit={"hour":"h","minute":"min"}.get(unit,unit)
                v={"kind":"duration","value":number,"unit":unit}
            elif unit:
                unit=unit[:-1] if unit.endswith("s") else unit
                v={"kind":"quantity","value":number,"unit":unit}
            else: return None
    if op!="eq": v["op"]=op
    return v


def _template(text, alias, num):
    out=text[:num.start()] + "<VALUE>" + text[num.end():]
    pat=re.compile(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",re.I)
    return re.sub(pat,"<SUBJECT>",out,count=1).casefold()


class AssertionParser:
    def __init__(self, training_rows: list[dict], lexicon: dict):
        self.subjects=_subjects(lexicon)
        self.allowed={s["name"]:set(s.get("attributes",[])) for s in lexicon.get("subjects",[]) if isinstance(s,dict) and s.get("name")}
        self.templates=defaultdict(set)
        self.conflicted_templates=set()
        self.stats={"training_rows":len(training_rows),"eligible_rows":0,"template_count":0}
        for row in training_rows:
            text=str(row.get("text", "")); form=row.get("form")
            if isinstance(form,str):
                import json
                try: form=json.loads(form)
                except (ValueError,TypeError): continue
            atoms=form if isinstance(form,list) else [form]
            if len(atoms)!=1 or not isinstance(atoms[0],dict): continue
            atom=atoms[0]; subj=atom.get("subject"); attr=atom.get("attribute"); val=atom.get("value")
            if subj not in self.allowed or attr not in self.allowed[subj] or not isinstance(val,dict) or val.get("kind") not in {"money","quantity","duration","percent"}: continue
            if _unsafe_context(text): continue
            if re.search(r"\b(?:and|but|while|although|whereas)\b",text,re.I): continue
            matches=_matches(text,self.subjects); explicit=[m for m in matches if m[2]==subj]
            nums=_number_spans(text)
            if len(explicit)!=1 or len(nums)!=1: continue
            parsed=_value_for(text,nums[0])
            if not parsed or _norm_value(parsed)!=_norm_value(val): continue
            signature=_template(text,explicit[0][3],nums[0])
            self.templates[signature].add((subj,attr))
            attrs_by_subject=defaultdict(set)
            for s,a in self.templates[signature]:
                attrs_by_subject[s].add(a)
            if any(len(attrs)>1 for attrs in attrs_by_subject.values()):
                self.conflicted_templates.add(signature)
            self.stats["eligible_rows"]+=1
        self.stats["template_count"]=len(self.templates)

    def parse(self, text: str) -> dict:
        text=str(text or "")
        unsupported={"atoms":[],"complete":False,"reason":"unsupported","method":"lexicon_numeric_template_v1"}
        if _unsafe_context(text) or re.search(r"\b(?:and|but|while|although|whereas)\b",text,re.I):
            return {**unsupported,"reason":"unsupported"}
        matches=_matches(text,self.subjects); nums=_number_spans(text)
        if len(nums)>1 or len(matches)!=1 or len(nums)!=1:
            return {**unsupported,"reason":"ambiguous_parse" if len(nums)>1 or len(matches)>1 else "unsupported"}
        parsed=_value_for(text,nums[0])
        if parsed is None: return unsupported
        # Match template after substituting each candidate alias; inference allows only exact structural wording.
        signatures=set()
        for start,end,canonical,alias in matches:
            signatures.add((canonical,_template(text,alias,nums[0])))
        interpretations=set()
        for canonical,sig in signatures:
            pairs=self.templates.get(sig,())
            if sig in self.conflicted_templates and any(s==canonical for s,_ in pairs):
                return {**unsupported,"reason":"ambiguous_parse"}
            interpretations.update((s,a) for s,a in pairs if s==canonical)
        if len(interpretations)!=1: return {**unsupported,"reason":"ambiguous_parse" if interpretations else "unsupported"}
        subject,attribute=next(iter(interpretations))
        if attribute not in self.allowed.get(subject,set()): return unsupported
        return {"atoms":[{"subject":subject,"attribute":attribute,"value":parsed}],"complete":True,"reason":"parsed","method":"lexicon_numeric_template_v1"}

    def summary(self):
        return {**self.stats,"templates_by_interpretation":sum(len(v) for v in self.templates.values()),"method":"lexicon_numeric_template_v1"}
