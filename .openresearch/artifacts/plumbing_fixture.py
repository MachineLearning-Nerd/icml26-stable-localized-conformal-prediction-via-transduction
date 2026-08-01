"""Synthesise a complete result tree so the analysis pipeline can be exercised end to end.

Numbers here are random and mean nothing scientifically; the point is that every
code path -- shard reduction, real-shard merge, all six claim functions, page
generation and the publish gate -- runs on realistically SHAPED input.
"""
import json, os, numpy as np
R=np.random.default_rng(0)
ROOT="plumb2/repo/results"; NL=19; MODELS=2
LB=[0.0,0.0005,0.00075,0.001,0.002,0.005,0.0075,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5]

def sim_shard(lo,hi,scale,T=500):
    r=hi-lo
    def blk(shape,c): return (c+scale*R.standard_normal(shape))
    out={}
    for k in ("base","SDCP","PPI","StCP-sel","oracle","NOAL"):
        out[k]={"n_repeats":r,
                "cov_mean_per_repeat":(0.905+0.01*R.standard_normal((MODELS,r))).tolist(),
                "size_mean_per_repeat":blk((MODELS,r),2.0).tolist(),
                "cov_sum_over_repeats":(0.9*r+0.05*R.standard_normal((MODELS,T))).tolist(),
                "size_sum_over_repeats":(2.0*r+0.1*R.standard_normal((MODELS,T))).tolist()}
    out["StCP"]={"n_repeats":r,
        "cov_mean_per_repeat":(0.905+0.01*R.standard_normal((MODELS,NL,r))).tolist(),
        "size_mean_per_repeat":blk((MODELS,NL,r),1.85).tolist(),
        "cov_sum_over_repeats":(0.9*r+0.05*R.standard_normal((MODELS,NL,T))).tolist(),
        "size_sum_over_repeats":(1.85*r+0.1*R.standard_normal((MODELS,NL,T))).tolist()}
    out["selected_idx"]=[[3]*r for _ in range(MODELS)]
    return out

for setting,scale in [("logabs-n30-m500",0.30),("logabs-n100-m500",0.18),("logabs-n500-m500",0.09),
                      ("logabs-n30-m30",0.34),("logabs-n30-m100",0.32),("noshift-n30-m500",0.30)]:
    d=os.path.join(ROOT,"shards",setting); os.makedirs(d,exist_ok=True)
    for lo in range(0,50,10):
        json.dump(sim_shard(lo,lo+10,scale),open(os.path.join(d,f"{lo}_{lo+10}.json"),"w"))

METH=['','SDCP','PPI','SLCP','SLCP-sel','ORCP','NOAL']
LBL={'':'base','SDCP':'SDCP','PPI':'PPI','SLCP':'SLCP','SLCP-sel':'SLCP-sel','ORCP':'ORCP','NOAL':'NOAL'}
os.makedirs(os.path.join(ROOT,"real"),exist_ok=True)
for ds,cls in [("CRIME",False),("BIO",False),("STAR",False),("DERMA",True),("TISSUE",True)]:
    # Table 1's "Std" is the spread of per-repeat MEAN set size, so the fixture has
    # to give each method its own across-repeat dispersion. Drawing them all with
    # the same noise makes a_ref - a_oracle collapse to zero and the oracle-adjusted
    # percentage explode -- an artefact of the fixture, not of the pipeline. These
    # values track the real CRIME/GLCP column (base 0.75, oracle 0.16).
    ACROSS = {'':0.75,'SDCP':1.42,'PPI':0.78,'SLCP':0.50,'SLCP-sel':0.58,'ORCP':0.16,'NOAL':0.86}
    CENTER = {'':2.6,'SDCP':3.1,'PPI':2.7,'SLCP':2.2,'SLCP-sel':2.3,'ORCP':2.0,'NOAL':2.8}
    covs={};sizes={}
    for mi,model in enumerate(['GLCP','SCC']):
        for m in METH:
            k=NL if m=='SLCP' else 1
            covs[(model,m)]=0.905+0.008*R.standard_normal((k,50,1))+0.002*R.standard_normal((k,50,120))
            per_rep=CENTER[m]+ACROSS[m]*R.standard_normal((k,50,1))
            sizes[(model,m)]=per_rep+0.05*R.standard_normal((k,50,120))
    for si,(lo,hi) in enumerate([(0,10),(10,20),(20,30),(30,40),(40,50)]):
        agg={};pr={}
        for model in ['GLCP','SCC']:
            for m in METH:
                key=(model+' '+m).strip(); c,s=covs[(model,m)][:,lo:hi],sizes[(model,m)][:,lo:hi]
                mar,size=c.mean(axis=(1,2)),s.mean(axis=(1,2)); mstd,sstd=c.mean(-1).std(-1),s.mean(-1).std(-1)
                loc=np.zeros(c.shape[0]) if cls else np.zeros((c.shape[0],3))
                agg[key]=([mar.tolist(),mstd.tolist(),size.tolist(),sstd.tolist(),loc.tolist()] if m=='SLCP'
                          else [float(mar[0]),float(mstd[0]),float(size[0]),float(sstd[0]),
                                float(loc[0]) if cls else loc[0].tolist()])
        for m in METH:
            c=np.concatenate([covs[('GLCP',m)],covs[('SCC',m)]],0)[:,lo:hi]
            s=np.concatenate([sizes[('GLCP',m)],sizes[('SCC',m)]],0)[:,lo:hi]
            pr[LBL[m]]={'cov_mean_per_repeat':c.mean(-1).tolist(),'size_mean_per_repeat':s.mean(-1).tolist()}
        json.dump({'kind':'real_shard','status':'OK','dataset':ds,'shard':[lo,hi],'n_reported':30,
                   'n_over_m':'30/500','aggregates':agg,'per_repeat':pr},
                  open(os.path.join(ROOT,"real",f"{ds}-s{si}.json"),"w"))

os.makedirs(os.path.join(ROOT,"checks"),exist_ok=True)
perrep=[]
for rep in range(5):
    raw=[{"lambda":float(l),"discrepancy":0.002+0.001*i,"delta_sq_norm":60.0/(1+400*l) if l>0 else 60.0,
          "objective":0.0} for i,l in enumerate(LB)]
    for q in raw: q["objective"]=q["discrepancy"]+q["lambda"]*q["delta_sq_norm"]
    perrep.append({"repeat":rep,"n_calibration":30,"m_unlabeled":500,"raw_path":raw,
        "enforced_delta_sq_norm_by_lambda":[q["delta_sq_norm"] for q in raw],
        "beta_level_q_by_lambda":[0.93]*NL,"beta_level_note":"probability level",
        "unlabeled_sample_intervention":{"lambda":0.03,
            "target_unlabeled":{"discrepancy":0.0031,"delta_sq_norm":0.09},
            "source_unlabeled_sham":{"discrepancy":0.0034,"delta_sq_norm":0.085},
            "target_resample_null":{"discrepancy":0.00312,"delta_sq_norm":0.0895},
            "treatment_shift_discrepancy":0.10,"null_shift_discrepancy":0.006,
            "treatment_shift_delta_sq":0.055,"null_shift_delta_sq":0.005}})
rd=np.array([[q["delta_sq_norm"] for q in p["raw_path"]] for p in perrep])
rc=np.array([[q["discrepancy"] for q in p["raw_path"]] for p in perrep])
json.dump({"kind":"invariants","dtype":"logabs","n":30,"m":500,"reps":5,"lambda_grid":LB,"seconds":900,
  "provenance":{"F_hat_S_given_X_trained_on":{"agent":"predAgent","distribution":"SOURCE","n_rows":2000,
      "uses_target_labels":False},"predictor_trained_on":{"agent":"trAgent","distribution":"SOURCE","n_rows":2000},
      "note":"Built before any target data exists."},
  "per_repeat":perrep,
  "objective_identity":{"form":"L = discrepancy + lambda*||theta-theta_hat||^2",
      "source_anchor":"Main/Tuner.py:202","holds_for_every_lambda_and_repeat":True},
  "regularisation_path_unenforced":{"mean_delta_sq_at_min_lambda":float(rd[:,0].mean()),
      "mean_delta_sq_at_max_lambda":float(rd[:,-1].mean()),
      "shrinkage_ratio":float(rd[:,-1].mean()/rd[:,0].mean()),
      "fraction_non_increasing_delta_steps":float(np.mean(np.diff(rd,axis=1)<=1e-12)),
      "fraction_non_decreasing_discrepancy_steps":float(np.mean(np.diff(rc,axis=1)>=-1e-12)),
      "delta_shrinks_overall_in_every_repeat":True},
  "regularisation_path_enforced_by_check_order":{"mean_delta_sq_at_min_lambda":float(rd[:,0].mean()),
      "mean_delta_sq_at_max_lambda":float(rd[:,-1].mean()),
      "fraction_non_increasing_delta_steps":1.0,"note":"context only"},
  "unlabeled_target_intervention":{"description":"matched null","statistic_note":"discrepancy is primary",
      "mean_treatment_shift_discrepancy":0.10,"mean_null_shift_discrepancy":0.006,
      "treatment_exceeds_null_discrepancy_per_repeat":[True]*5,
      "treatment_exceeds_null_in_every_repeat":True,"treatment_exceeds_null_in_majority":True,
      "mean_treatment_shift_delta_sq":0.055,"mean_null_shift_delta_sq":0.005}},
  open(os.path.join(ROOT,"checks","invariants.json"),"w"))

BAND=[0.88,0.9+0.02+1/31]

def _arm(name, exch, mean):
    per=[mean]*20
    return {"arm":name,"exchangeable":exch,
        "calibration_distribution":"target" if exch else "source (mu_s, gamma_s=1.2, me_s=d/3)",
        "repeats":len(per),"coverage_per_repeat":per,"coverage_mean":mean,
        "coverage_stderr":0.004,"coverage_ci95":[mean-0.008,mean+0.008],
        "selected_lambda_idx":[2]*len(per),"band":BAND,
        "inside_band":bool(BAND[0]<=mean<BAND[1])}

# CONTROL_MEAN_NONEXCH is the knob the plumbing fixture turns: pushing the
# non-exchangeable arm back inside the band must make C6 BLOCKED, not FALSIFIED.
NONEXCH=float(os.environ.get("PLUMB_NONEXCH_COVERAGE",0.812))
arms=[_arm("exchangeable",True,0.903),_arm("non_exchangeable",False,NONEXCH)]
json.dump({"kind":"control_exchangeability","dtype":"logabs","n":30,"m":500,"reps":20,
  "band":{"lo":BAND[0],"hi":BAND[1],
      "formula":"[1-a-a_tol, 1-a+a_tol+1/(n+1)) at a=0.1, a_tol=0.02, n=30"},
  "seconds":0.0,"arms":arms,
  "control_is_informative":arms[0]["inside_band"] and not arms[1]["inside_band"],
  "interpretation":"synthetic plumbing fixture"},
  open(os.path.join(ROOT,"checks","control_exchangeability.json"),"w"))
print("synthetic result tree written")
