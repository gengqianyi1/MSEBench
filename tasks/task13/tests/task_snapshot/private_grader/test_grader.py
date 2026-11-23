from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from grader import grade

class RouteFixtures(unittest.TestCase):
    def run_case(self,route):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sub=root/'submission'; private=root/'private'; sub.mkdir(); private.mkdir()
            (sub/'route.json').write_text(json.dumps({'route':route}),encoding='utf-8')
            return grade(SimpleNamespace(submission_dir=sub,private_data_dir=private,run_dir=root,task_dir=root))
    def test_empty_route(self): self.assertIn('parsed',self.run_case([])['failure_codes'])
    def test_unknown_reaction(self):
        result=self.run_case([{'reaction':'magic','reactants':['CCO'],'product':'CCO'},{'reaction':'magic','reactants':['CCO'],'product':'CCO'}]); self.assertIn('all_reactions_in_library',result['failure_codes'])
    def reference_route(self):
        return [
            {'reaction':'suzuki_coupling','reactants':['COC(=O)c1cc(Br)cc(Br)c1','Cc1ccc(B(O)O)cc1'],'product':'COC(=O)c1cc(Br)cc(-c2ccc(C)cc2)c1'},
            {'reaction':'suzuki_coupling','reactants':['COC(=O)c1cc(Br)cc(-c2ccc(C)cc2)c1','OB(O)c1ccccc1'],'product':'COC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1'},
            {'reaction':'ester_hydrolysis','reactants':['COC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1'],'product':'Cc1ccc(-c2cc(C(=O)O)cc(-c3ccccc3)c2)cc1'},
            {'reaction':'fischer_esterification','reactants':['Cc1ccc(-c2cc(C(=O)O)cc(-c3ccccc3)c2)cc1','OCCBr'],'product':'Cc1ccc(-c2cc(C(=O)OCCBr)cc(-c3ccccc3)c2)cc1'},
            {'reaction':'sn2_n_alkylation','reactants':['NCc1ccccc1','Cc1ccc(-c2cc(C(=O)OCCBr)cc(-c3ccccc3)c2)cc1'],'product':'Cc1ccc(-c2cc(C(=O)OCCNCc3ccccc3)cc(-c3ccccc3)c2)cc1'},
            {'reaction':'n_acylation','reactants':['Cc1ccc(-c2cc(C(=O)OCCNCc3ccccc3)cc(-c3ccccc3)c2)cc1','CC(=O)Cl'],'product':'CC(=O)N(CCOC(=O)c1cc(-c2ccccc2)cc(-c2ccc(C)cc2)c1)Cc1ccccc1'},
        ]
    def test_reference_route_scores_100(self): self.assertEqual(self.run_case(self.reference_route())['score'],100)
    def test_reactant_order_is_equivalent(self):
        route=self.reference_route(); route[0]['reactants'].reverse()
        self.assertEqual(self.run_case(route)['score'],100)
    def test_reaction_name_normalization(self):
        route=self.reference_route(); route[0]['reaction']='Suzuki Coupling'; route[5]['reaction']='N-Acylation'
        self.assertEqual(self.run_case(route)['score'],100)
    def test_six_step_boundary_is_accepted(self): self.assertNotIn('n_steps_within_limit',self.run_case(self.reference_route())['failure_codes'])

if __name__=='__main__': unittest.main()
