import pandas as pd
from src.prediction_validator import validate_predictions

def test_validator_accepts_complete_predictions(tmp_path):
    test=tmp_path/'test.json.gz'; pred=tmp_path/'pred.csv'
    pd.DataFrame({'sample_id':['a','b'],'structure':[{},{}]}).to_json(test,orient='records',lines=True,compression='gzip')
    pd.DataFrame({'sample_id':['a','b'],'predicted_exfoliation_energy_mev_atom':[1.0,2.0]}).to_csv(pred,index=False)
    assert validate_predictions(pred,test)['passed']
