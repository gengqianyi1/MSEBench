from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from grader import grade

class LithiumFixtures(unittest.TestCase):
    def evaluate(self, smiles, extra=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sub=root/'submission'; private=root/'private'; sub.mkdir(); private.mkdir()
            payload={'smiles':smiles}; payload.update(extra or {})
            (sub/'candidate.json').write_text(json.dumps(payload),encoding='utf-8')
            (private/'reserved_smiles.txt').write_text('CC(OC)(OC)C(OC)OC\n',encoding='utf-8')
            return grade(SimpleNamespace(submission_dir=sub,private_data_dir=private,run_dir=root,task_dir=root))
    def test_invalid_smiles(self): self.assertEqual(self.evaluate('not-smiles')['score'],0)
    def test_historical_full_score_is_reserved(self): self.assertIn('C6_leakage_control',self.evaluate('CC(OC)(OC)C(OC)OC')['failure_codes'])
    def test_three_oxygen_neighbour_filter(self): self.assertIn('C5',self.evaluate('COC(OC)(OC)OC')['failure_codes'])
    def test_known_valid_candidate(self): self.assertEqual(self.evaluate('COCCOC(=O)C(F)OC')['score'],100)
    def test_candidate_reported_properties_are_ignored(self):
        result=self.evaluate('COCCOC(=O)C(F)OC',{'logP':999,'score':0,'HOMO_eV':999})
        self.assertEqual(result['score'],100)
        self.assertLessEqual(result['evidence']['properties']['logP'],0.9)
    def test_non_string_payload_is_rejected(self): self.assertIn('INVALID_CANDIDATE',self.evaluate({'not':'smiles'})['failure_codes'])

if __name__=='__main__': unittest.main()
