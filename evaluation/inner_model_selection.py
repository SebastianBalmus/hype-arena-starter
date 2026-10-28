#!/usr/bin/env python3
"""Select one predeclared text model on frozen inner validation, then test once."""
import argparse, csv, hashlib, json, os, platform, subprocess, sys, tempfile, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import run_text_experiments as helpers
from evaluation.scoring import score_rows
from experiments.text_variants import MODELS

TRACKS=helpers.TRACKS
METHOD=ROOT/'experiments/text_variants.py'

def write_csv(path, rows, fields):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def score_predictions(gold,predpath,outdir):
    preds=helpers.numeric_predictions(helpers.read_csv(predpath))
    metrics,errors,scored=score_rows(gold,preds)
    if outdir is not None:
        helpers.write_csv(outdir/'scored_predictions.csv',scored,
          ['id','gold','label','confidence','decision_reason','correct','output_valid','error_codes'])
    if not errors['eligible_for_comparison']: raise RuntimeError(f'Invalid output: {errors}')
    return metrics,errors,scored

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run-id',required=True); a=ap.parse_args()
    if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.run_id): ap.error('invalid run ID')
    for f,h in helpers.HASHES.items():
        actual=helpers.digest(ROOT/f)
        if actual!=h: ap.error(f'Benchmark hash mismatch for {f}: {actual}')
    rootout=ROOT/'results/evaluations/inner_selected'/a.run_id
    if rootout.exists(): ap.error(f'Output already exists: {rootout}')
    rows=helpers.read_csv(ROOT/'data/practice.csv'); manifest=json.loads((ROOT/'evaluation/splits-v1.json').read_text())
    if len(rows)!=len({r['id'] for r in rows}): ap.error('duplicate practice IDs')
    row_by={r['id']:r for r in rows}; source_files=(*helpers.SOURCE_FILES,'evaluation/inner_model_selection.py')
    source_hashes={f:helpers.digest(ROOT/f) for f in source_files}
    results={}; selections={}; timings={}; allids={}
    import numpy,scipy,sklearn,joblib,threadpoolctl
    versions={'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__}
    with tempfile.TemporaryDirectory(prefix='hype-inner-v1-') as tmp:
      tmp=Path(tmp)
      for track in TRACKS:
        s=manifest['splits'][track]; train_ids=s['train_ids']; test_ids=s['test_ids']; fit_ids=s['fit_ids']; val_ids=s['validation_ids']
        if set(fit_ids)|set(val_ids)!=set(train_ids) or set(fit_ids)&set(val_ids): raise ValueError(f'{track}: inner IDs do not partition outer train')
        # Validation labels are evaluator-only. Method sees train rows and validation rows with empty labels.
        val_input=helpers.sanitized_rows(rows,fit_ids,val_ids)
        val_path=tmp/f'{track}-validation.csv'; helpers.write_csv(val_path,val_input,['id','text','date','label'])
        val_gold=[{'id':r['id'],'label':r['label']} for r in rows if r['id'] in set(val_ids)]
        val_scores={}; val_outs={}; valtim={}
        for model in MODELS:
          pred=tmp/f'{track}-{model}-val.csv'; tpath=tmp/f'{track}-{model}-val-time.json'
          cmd=[sys.executable,str(METHOD),'--input',str(val_path),'--model',model,'--output',str(pred),'--seed','42','--timings',str(tpath)]
          env=os.environ.copy()
          for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): env[k]='2'
          start=time.perf_counter(); subprocess.run(cmd,cwd=ROOT,env=env,check=True,capture_output=True,text=True); elapsed=time.perf_counter()-start
          # Keep evaluator-only validation gold in memory, outside the input directory.
          m,e,_=score_predictions(val_gold,pred,None)
          if not e['eligible_for_comparison']: raise RuntimeError(f'{track}/{model} validation invalid: {e}')
          val_scores[model]=m['macro_f1']; val_outs[model]={'macro_f1':m['macro_f1'],'per_class':m['per_class'],'confusion_matrix':m['confusion_matrix']}
          valtim[model]={'method_process_seconds':elapsed,**json.loads(tpath.read_text())}
        selected=max(MODELS,key=lambda m:(val_scores[m],-MODELS.index(m)))
        outdir=rootout/'fixed_050'/track; outdir.mkdir(parents=True)
        selection={'benchmark_version':'arena-eval-v1','track':track,'policy':'fixed_050','seed':42,'threshold':0.5,'fit_ids':fit_ids,'validation_ids':val_ids,'validation_macro_f1_by_model':val_scores,'validation_metrics_by_model':val_outs,'selected_model':selected,'tie_break':'first model in experiments.text_variants.MODELS ordering'}
        helpers.write_json(outdir/'validation_selection.json',selection)
        # Refit selected model on all outer-train IDs; test labels/forms are omitted from method input.
        test_input=helpers.sanitized_rows(rows,train_ids,test_ids); test_path=tmp/f'{track}-test.csv'; helpers.write_csv(test_path,test_input,['id','text','date','label'])
        test_gold=[{'id':r['id'],'label':r['label']} for r in rows if r['id'] in set(test_ids)]
        pred=outdir/'predictions.csv'; tpath=tmp/f'{track}-test-time.json'
        cmd=[sys.executable,str(METHOD),'--input',str(test_path),'--model',selected,'--output',str(pred),'--seed','42','--timings',str(tpath)]
        start=time.perf_counter(); subprocess.run(cmd,cwd=ROOT,env=env,check=True,capture_output=True,text=True); elapsed=time.perf_counter()-start
        om,errors,scored=score_predictions(test_gold,pred,outdir)
        metric={'benchmark_version':'arena-eval-v1','method_id':'inner_selected','selected_model':selected,'run_id':a.run_id,'split':track,'policy':'fixed_050','access_profile':'batch_unlabelled_v1','eligible_for_comparison':errors['eligible_for_comparison'],'output_errors':errors,'classification':{'status':'ok',**om},'generation':{'status':'not_implemented','reason':'Classification-only','verified_valid_yield_at_1000':None},'proofs':{'status':'not_implemented','reason':'No proof functionality','decision_proof_rate':None,'grounded_decision_yield':None},'resources':{'status':'not_run','reason':'Constrained offline Linux amd64 benchmark not run','method_process_seconds':elapsed,'peak_ram_bytes':None,**json.loads(tpath.read_text())}}
        helpers.write_json(outdir/'metrics.json',metric)
        results[track]=metric; selections[track]=selection; timings[track]={'validation':valtim,'outer':{'method_process_seconds':elapsed,**json.loads(tpath.read_text())}}
        allids[track]={'fit_ids':fit_ids,'validation_ids':val_ids,'outer_train_ids':train_ids,'test_ids':test_ids,
                       'validation_input_sha256':helpers.digest(val_path),'test_input_sha256':helpers.digest(test_path)}
        print(f'{track:10} selected={selected:26} validation_macro_f1={val_scores[selected]:.3f} outer_macro_f1={om["macro_f1"]:.3f}',flush=True)
    run={'benchmark_version':'arena-eval-v1','method_id':'inner_selected','run_id':a.run_id,'seed':42,'splits':list(TRACKS),'policies':['fixed_050'],'pinned_sha256':helpers.HASHES,'source_commit':helpers.source_version(),'source_file_sha256':source_hashes,'dependencies':versions,'exact_reproduction_command':f'.venv/bin/python evaluation/inner_model_selection.py --run-id {a.run_id}','training_sources':'Only split fit IDs for validation selection; selected model refit on full outer train IDs for test. Text features only; IDs/dates not features; no forms or lexicon.','training_ids':{t:allids[t]['outer_train_ids'] for t in TRACKS},'fit_ids':{t:allids[t]['fit_ids'] for t in TRACKS},'validation_ids':{t:allids[t]['validation_ids'] for t in TRACKS},'test_ids':{t:allids[t]['test_ids'] for t in TRACKS},'access_profile':'batch_unlabelled_v1','confidence':{'semantics':'uncalibrated winning-class probability','calibration':'none','threshold':0.5,'policy':'fixed_050','selection':'model selected by inner validation macro F1; outer test not used'},'leakage_audit':{'status':'pass_logical_boundary','detail':'Separate method process receives sanitized CSV with validation/test labels blank and forms omitted; no OS filesystem enforcement. Selection uses validation labels only in evaluator.'},'factual_assets':'None; classifier predictions are not world evidence.','timings':timings,'hardware':{'os':platform.platform(),'machine':platform.machine(),'cpu':platform.processor(),'thread_limit':2,'cpu_limit_enforced':None,'memory_limit_bytes':None,'process_limit':None,'timing':'perf_counter'},'capabilities':{'classification':'implemented','generation':'not_implemented','proofs':'not_implemented','independent_truth_verifier':'not_run','constrained_container_benchmark':'not_run','frontend':'not_implemented'},'selection_note':'Previous experiments already used these outer development splits; inner selection reduces direct outer-test selection leakage but does not make outer test pristine.'}
    run['sanitized_input_sha256']={t:{'validation':allids[t]['validation_input_sha256'],
                                    'test':allids[t]['test_input_sha256']} for t in TRACKS}
    run['model_asset_sha256']=None
    run['model_asset_reason']='Models are fitted in memory separately for each split and are not serialized.'
    helpers.write_json(rootout/'run.json',run)
    lines=['# Inner-selected text classifier','',f'Run: `{a.run_id}`; protocol `arena-eval-v1`; policy `fixed_050`.','', 'Models were compared using frozen inner validation IDs. Macro F1 selected the model, with ties broken by the declared MODELS ordering. The selected model was refit on each full outer training pool and evaluated once on its test IDs. Test labels were not sent to the method process.','', 'The practice corpus and outer development splits have already informed prior experiments. Inner selection reduces direct test-label selection, but these outer scores are not pristine generalization estimates. The temporal track is a forward batch stress test. Confidence is uncalibrated. Generation and proofs are not implemented; constrained resources were not measured.','', '| Split | Selected model | Validation macro F1 | Outer macro F1 | False P/R | Unsure F1 | Coverage | Errors/count | Risk | Brier |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in TRACKS:
      m=results[t]['classification']; sel=selections[t]; model=sel['selected_model']; v=sel['validation_macro_f1_by_model'][model]; p=m['per_class']
      lines.append(f"| {t} | {model} | {v:.3f} | {m['macro_f1']:.3f} | {p['false']['precision']:.3f}/{p['false']['recall']:.3f} | {p['unsure']['f1']:.3f} | {m['decided_coverage']:.3f} | {m['decided_errors']}/{m['decided_count']} | {m['decided_risk'] if m['decided_risk'] is not None else 'n/a'} | {m['correctness_brier'] if m['correctness_brier'] is not None else 'n/a'} |")
    lines += ['', '## Matched controls', '', 'The controls below use the same benchmark, split and fixed_050 policy. They trained on the complete outer training pool without inner model selection, so they are useful context but not fully paired with this selected-fit procedure.', '', '| Split | Selected model | Always true | Word LR | Balanced word LR |', '|---|---:|---:|---:|---:|']
    for t in TRACKS:
      cv=[]
      for cn in helpers.CONTROLS:
        cp=ROOT/'results/evaluations'/cn/'text-variants-20261028-001'/'fixed_050'/t/'metrics.json'
        cm=json.loads(cp.read_text())['classification']['macro_f1']
        cv.append(f'{cm:.3f}')
      lines.append(f"| {t} | {results[t]['classification']['macro_f1']:.3f} | {cv[0]} | {cv[1]} | {cv[2]} |")
    lines += ['', 'Confusion matrices and reliability bins are included in each split\'s `metrics.json`. '
              'Grounded yield, verified generation yield, proof rate and constrained peak RAM are not measured.',
              '', '## Representative stratified errors', '',
              'The text classifiers do not implement parsing, evidence or temporal rules; '
              'the emitted reason is shown rather than a guessed causal explanation.', '',
              '| ID | Gold | Predicted | Confidence | Reason |', '|---|---|---|---:|---|']
    scored_path=rootout/'fixed_050'/'stratified'/'scored_predictions.csv'
    failures=[r for r in helpers.read_csv(scored_path) if r['correct']=='False']
    failures.sort(key=lambda r:(r['gold']=='true',r['id']))
    for r in failures[:5]:
      conf=float(r['confidence']) if r['confidence'] else None
      lines.append(f"| {r['id']} | {r['gold']} | {r['label']} | {conf if conf is not None else 'n/a'} | {r['decision_reason']} |")
    lines += ['', '## Reproduction', '', f'```sh\n.venv/bin/python evaluation/inner_model_selection.py --run-id NEW-UNIQUE-ID\n```','', 'Method code hashes and exact frozen ID arrays are recorded in `run.json` and `validation_selection.json`. The method receives sanitized runtime-shaped CSV in a separate process; filesystem isolation is logical, not OS-enforced.',
              '', 'Development guidance: Kassis, Agarwal, He, Patel and Brueckner (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).','']
    (rootout/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Completed all tracks; results are development comparisons, not pristine generalization estimates.')

if __name__=='__main__': main()
