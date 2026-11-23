from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from grader import grade

class PybammFixtures(unittest.TestCase):
    def run_case(self,candidate,nested=True):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sub=root/'submission'; private=root/'private'; sub.mkdir(); private.mkdir()
            payload={'candidate':candidate} if nested else candidate
            (sub/'candidate.json').write_text(json.dumps(payload),encoding='utf-8')
            return grade(SimpleNamespace(submission_dir=sub,private_data_dir=private,run_dir=root,task_dir=root))
    def test_missing_fields(self): self.assertIn('INVALID_CANDIDATE',self.run_case({'cc_c_rate':2})['failure_codes'])
    def test_out_of_range_does_not_start_simulation(self):
        c={'cc_c_rate':9,'cc_until_voltage_V':4.1,'cv_voltage_V':4.1,'cv_until_current_C':0.1,'negative_electrode_thickness_um':75,'positive_electrode_thickness_um':75,'electrode_porosity':0.34,'particle_radius_um':5.5}
        result=self.run_case(c); self.assertEqual(result['subscores']['valid_submission_and_protocol'],0); self.assertIsNone(result['evidence']['simulation_error'])
    def test_flat_and_nested_envelopes_are_equivalent(self):
        c={'cc_c_rate':9,'cc_until_voltage_V':4.1,'cv_voltage_V':4.1,'cv_until_current_C':0.1,'negative_electrode_thickness_um':75,'positive_electrode_thickness_um':75,'electrode_porosity':0.34,'particle_radius_um':5.5}
        nested=self.run_case(c,True); flat=self.run_case(c,False)
        self.assertNotIn('INVALID_CANDIDATE',flat['failure_codes'])
        self.assertEqual(nested['score'],flat['score'])
        self.assertEqual(nested['evidence']['candidate'],flat['evidence']['candidate'])

if __name__=='__main__': unittest.main()
