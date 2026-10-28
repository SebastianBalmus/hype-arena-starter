"""Frozen-split real-only versus ancestry-filtered augmented training diagnostic."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.run_synthetic_transfer import sha
from evaluation.run_text_experiments import read_csv, write_csv, write_json, HASHES
from evaluation.scoring import score_rows
from experiments.daria_feature_variants import CONFIGS
from experiments.daria_feature_sweep import fit_predict, audit_cache
from baseline import make_model, predict_rows


def main():
    dest = ROOT/'results/evaluations/augmented_training'/sys.argv[1]
    dest.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for name, expected in HASHES.items():
        assert sha(ROOT/name) == expected, name
    original = read_csv(ROOT/'data/practice.csv')
    data = ROOT/'data/augmentation-20261028-2000'
    synthetic = read_csv(data/'practice_additional_2000.csv')
    provenance = {r['id']: r for r in map(json.loads, (data/'provenance.jsonl').read_text().splitlines())}
    manifest = json.loads((ROOT/'evaluation/splits-v1.json').read_text())
    config = next(c for c in CONFIGS if c['id']=='expanded_none')
    params = json.loads((ROOT/'results/comparisons/daria-features-20261028-001/method-inputs/stratified/gb-params.json').read_text())
    source = dest/'source'
    commit = subprocess.check_output(['git','rev-parse','5943f61'],cwd=ROOT,text=True).strip()
    for f in subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'hooly'],cwd=ROOT,text=True).splitlines():
        if f.endswith('.py') and f not in ('hooly/run.py','hooly/writer.py'):
            p=source/f
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(subprocess.check_output(['git','show',f'{commit}:{f}'],cwd=ROOT))
    run = {'benchmark_version':'ancestry-filtered-augmentation-diagnostic-v1','seed':42,'policy':'fixed_050',
           'method':'Daria expanded_none + gradient boosting','parameters':params,'upstream_commit':commit,
           'pinned_sha256':HASHES,'synthetic_sha256':sha(data/'practice_additional_2000.csv'),
           'provenance_sha256':sha(data/'provenance.jsonl'),'runner_sha256':sha(__file__),
           'selection':'Same frozen primary-track GB parameters for every track and both conditions. No tuning on these results.',
           'access_profile':'batch_unlabelled_v1 with declared synthetic augmentation',
           'direct_boundary':'Real test labels/forms absent. Synthetic rows allowed only if their source is in frozen training IDs. Calibration fit/held groups also keep source families together.',
           'eligible_for_clean_improvement_claim':False,
           'limitations':'The supplied augmentation templates/selection were authored with full-corpus access; filtering ancestors removes direct held-out-answer descendants but does not recreate untouched split-specific augmentation. Reused development data; native HMM OOF retains permitted training forms; temporal evidence is probabilistic, not proof.',
           'generation':{'status':'not_implemented'},'proofs':{'status':'not_verified'},'resources':{'status':'not_run','reason':'Native macOS; container constraints and peak RAM not measured'},
           'conditions':{}}
    write_json(dest/'run.json',run)
    all_results={}
    fields=['id','text','date','label']
    for track in ('stratified','temporal','checker'):
        split=manifest['splits'][track]
        train_ids,test_ids,fit_ids=map(set,(split['train_ids'],split['test_ids'],split['fit_ids']))
        gold=[r for r in original if r['id'] in test_ids]
        permitted=[r for r in synthetic if provenance[r['id']]['source_id'] in train_ids]
        assert all(provenance[r['id']]['source_id'] not in test_ids for r in permitted)
        all_results[track]={}
        for condition in ('real_only','real_plus_generated'):
            out=dest/track/condition
            inputs=out/'inputs'
            inputs.mkdir(parents=True)
            aug=permitted if condition=='real_plus_generated' else []
            train=[r for r in original if r['id'] in train_ids]+aug
            runtime=[{k:r[k] for k in fields} for r in aug]
            runtime += [{k:('' if k=='label' and r['id'] in test_ids else r[k]) for k in fields} for r in original]
            calibration=list(split['fit_ids'])+[r['id'] for r in aug if provenance[r['id']]['source_id'] in fit_ids]
            write_csv(inputs/'runtime.csv',runtime,fields)
            write_csv(inputs/'forms.csv',train,fields+['form'])
            write_json(inputs/'calibration-fit.json',calibration)
            cmd=[sys.executable,str(ROOT/'integrations/daria_features.py'),'--source-dir',str(source),'--input',str(inputs/'runtime.csv'),'--forms',str(inputs/'forms.csv'),'--lexicon',str(ROOT/'data/lexicon.json'),'--calibration-fit-ids',str(inputs/'calibration-fit.json'),'--output',str(out/'features.json')]
            run['conditions'][f'{track}/{condition}']={'real_train':len(train_ids),'synthetic_train':len(aug),'total_train':len(train),'test':len(gold),'training_ids':[r['id'] for r in train],'test_ids':split['test_ids'],'command':cmd}
            write_json(dest/'run.json',run)
            print(f'{track}/{condition}: training {len(train_ids)} original + {len(aug)} generated; testing {len(gold)} original',flush=True)
            with (out/'fit.log').open('w') as log:
                subprocess.run(cmd,cwd=inputs,check=True,stdout=log,stderr=subprocess.STDOUT)
            cache=json.loads((out/'features.json').read_text())
            audit_cache(cache,[r['id'] for r in train],split['test_ids'])
            pred,timing=fit_predict(config,cache,params)
            c,errors,scored=score_rows(gold,pred)
            assert errors['eligible_for_comparison'],errors
            byid={r['id']:r for r in gold}
            enriched=[dict(s,text=byid[s['id']]['text'],date=byid[s['id']]['date']) for s in scored]
            write_csv(out/'scored_predictions.csv',enriched,list(enriched[0]))
            write_csv(out/'predictions.csv',pred,list(pred[0]))
            controls={}
            for name in ('always_true','tfidf_lr','tfidf_lr_balanced'):
                model=make_model(name,seed=42)
                model.fit([r['text'] for r in train],[r['label'] for r in train])
                cp=predict_rows(model,gold,min_confidence=.5)
                for r in cp:r['decision_reason']='decided' if r['label']!='unsure' else 'low_confidence'
                controls[name]=score_rows(gold,cp)[0]
            write_json(out/'metrics.json',{'classification':c,'output_errors':errors,'controls':controls,'feature_timing':cache['timings'],'classifier_timing':timing,'eligible_for_clean_improvement_claim':False})
            all_results[track][condition]=c
            print(f'  Accuracy {c["accuracy"]:.4f}, macro F1 {c["macro_f1"]:.4f}',flush=True)
    lines=['# Original training split plus generated examples','',
           'Same Daria expanded-feature gradient boosting pipeline, primary-track frozen settings, seed 42 and fixed_050 for all runs. Real-only controls were rebuilt under identical settings and calibration policy. No model selection used these outcomes.','',
           'Only generated descendants of each frozen training pool are included. The calibration fitting pool likewise excludes descendants of its held source rows. All pending original labels/forms are hidden. This is still a development diagnostic: augmentation authoring/selection saw the full corpus, and the supplied synthetic batch was filtered rather than recreated from split-only information. Results cannot establish a clean v1 improvement or be directly compared to the previous 1,897-row transfer test.','',
           '| Track | Training | Original train | Generated train | Test | Accuracy | Macro F1 |','|---|---|---:|---:|---:|---:|---:|']
    for track,conditions in all_results.items():
        for condition,c in conditions.items():
            a=run['conditions'][f'{track}/{condition}']
            lines.append(f'| {track} | {condition} | {a["real_train"]} | {a["synthetic_train"]} | {a["test"]} | {c["accuracy"]:.4f} | {c["macro_f1"]:.4f} |')
    for track,conditions in all_results.items():
        lines += ['',f'## {track}: per-class results','','| Training | Class | Precision | Recall | F1 | Support |','|---|---|---:|---:|---:|---:|']
        for condition,c in conditions.items():
            for label,m in c['per_class'].items():lines.append(f'| {condition} | {label} | {m["precision"]:.4f} | {m["recall"]:.4f} | {m["f1"]:.4f} | {m["support"]} |')
            lines += ['',f'{condition} confusion matrix (gold rows / predicted columns true, false, unsure): `{json.dumps(c["confusion_matrix"])}.`','']
        failures=[r for r in read_csv(dest/track/'real_plus_generated/scored_predictions.csv') if r['gold']!=r['label']]
        lines += ['Representative augmented-training failures (observed errors, not a causal proof audit):','']
        for r in failures[:5]:lines.append(f'- {r["id"]}: {r["gold"]} → {r["label"]}. {r["text"]}')
    lines += ['','Per-class controls, risk/coverage, reliability bins, confusion matrices, source-scoped inputs and predictions are saved per condition. Generation, independent proof verification and constrained container benchmarks were not run. Prior reference results and shared splits are unchanged.','', 'Reproduce: `.venv/bin/python evaluation/run_augmented_training.py NEW-UNIQUE-RUN-ID`.','']
    (dest/'report.md').write_text('\n'.join(lines))
    run['total_seconds']=time.perf_counter()-started
    write_json(dest/'run.json',run)
    write_json(dest/'summary.json',all_results)


if __name__=='__main__':
    main()
