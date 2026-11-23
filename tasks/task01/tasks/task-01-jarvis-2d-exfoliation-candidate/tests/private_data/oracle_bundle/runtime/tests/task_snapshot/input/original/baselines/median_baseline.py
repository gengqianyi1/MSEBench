from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",default="data"); ap.add_argument("--output-dir",default="baseline_median"); args=ap.parse_args()
    d=Path(args.data_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    train=pd.read_json(d/'train.json.gz',orient='records',lines=True,compression='infer')
    val=pd.read_json(d/'validation.json.gz',orient='records',lines=True,compression='infer')
    test=pd.read_json(d/'test_public.json.gz',orient='records',lines=True,compression='infer')
    med=float(train.exfoliation_energy_mev_atom.median())
    vp=np.full(len(val),med)
    metrics={'median':med,'mae_mev_atom':float(mean_absolute_error(val.exfoliation_energy_mev_atom,vp)),'rmse_mev_atom':float(mean_squared_error(val.exfoliation_energy_mev_atom,vp)**0.5),'spearman':0.0}
    pd.DataFrame({'sample_id':test.sample_id.astype(str),'predicted_exfoliation_energy_mev_atom':med}).to_csv(out/'predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
