from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


def weighted_stats(structure, attr):
    vals=[]; weights=[]
    for el, amount in structure.composition.get_el_amt_dict().items():
        obj=next(e for e in structure.composition.elements if e.symbol==el)
        value=getattr(obj,attr,None)
        try: value=float(value)
        except (TypeError,ValueError): value=np.nan
        vals.append(value); weights.append(amount)
    a=np.asarray(vals,float); w=np.asarray(weights,float); mask=np.isfinite(a)
    if not mask.any(): return [np.nan]*4
    a=a[mask]; w=w[mask]/w[mask].sum(); mean=float(np.sum(a*w))
    return [mean,float(np.nanmin(a)),float(np.nanmax(a)),float(np.sqrt(np.sum(w*(a-mean)**2)))]


def features(sdict):
    s=Structure.from_dict(sdict); lat=s.lattice.matrix; av,bv,cv=lat
    normal=np.cross(av,bv); area=float(np.linalg.norm(normal)); normal=normal/max(area,1e-12)
    proj=s.cart_coords@normal; thickness=float(proj.max()-proj.min()) if len(proj) else 0.0
    dm=s.distance_matrix.copy(); dm[dm<1e-8]=np.nan
    nearest=np.nanmin(dm,axis=1)
    row=[len(s),len(s.composition.elements),area,area/len(s),len(s)/area,thickness,s.lattice.a,s.lattice.b,s.lattice.a/max(s.lattice.b,1e-12),float(np.nanmin(dm)),float(np.nanmean(nearest)),float(np.nanstd(nearest))]
    for attr in ('Z','atomic_mass','X','atomic_radius'):
        row.extend(weighted_stats(s,attr))
    return row

FEATURE_NAMES=['n_sites','n_elements','in_plane_area','area_per_atom','areal_density','layer_thickness','a','b','a_over_b','min_distance','mean_nearest','std_nearest']+[f'{a}_{s}' for a in ('Z','atomic_mass','X','atomic_radius') for s in ('mean','min','max','std')]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data'); ap.add_argument('--output-dir',default='baseline_pymatgen_rf'); args=ap.parse_args()
    d=Path(args.data_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    train=pd.read_json(d/'train.json.gz',orient='records',lines=True,compression='infer'); val=pd.read_json(d/'validation.json.gz',orient='records',lines=True,compression='infer'); test=pd.read_json(d/'test_public.json.gz',orient='records',lines=True,compression='infer')
    Xtr=np.asarray([features(x) for x in train.structure]); Xv=np.asarray([features(x) for x in val.structure]); Xt=np.asarray([features(x) for x in test.structure])
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',ExtraTreesRegressor(n_estimators=400,min_samples_leaf=2,max_features=0.8,random_state=42,n_jobs=-1))])
    model.fit(Xtr,train.exfoliation_energy_mev_atom); vp=model.predict(Xv); tp=model.predict(Xt)
    metrics={'mae_mev_atom':float(mean_absolute_error(val.exfoliation_energy_mev_atom,vp)),'rmse_mev_atom':float(mean_squared_error(val.exfoliation_energy_mev_atom,vp)**0.5),'spearman':float(spearmanr(val.exfoliation_energy_mev_atom,vp).statistic)}
    pd.DataFrame({'sample_id':test.sample_id.astype(str),'predicted_exfoliation_energy_mev_atom':tp}).to_csv(out/'predictions.csv',index=False)
    joblib.dump(model,out/'model.joblib'); (out/'feature_names.json').write_text(json.dumps(FEATURE_NAMES,indent=2)); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
