import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parser import ParseError, parse_candidate
from xtb_runner import parse_dipro_hole_coupling_ev


class CandidateParserTests(unittest.TestCase):
    def test_accepts_required_candidate_shape(self):
        parsed = parse_candidate(
            '{"candidate":{"candidate_id":"c1","parent_candidate_id":null,'
            '"proposal_mode":"direct_smiles","smiles":"c1ccccc1",'
            '"structural_modification":"baseline",'
            '"modification_reasoning":"parser smoke test"}}'
        )
        self.assertEqual(parsed["candidate_id"], "c1")

    def test_rejects_missing_required_fields(self):
        with self.assertRaises(ParseError):
            parse_candidate('{"candidate": {"smiles": "c1ccccc1"}}')


class DiproHoleCouplingParserTests(unittest.TestCase):
    def test_selects_hole_summary_not_later_charge_or_ct_values(self):
        output = """
 total |J(AB,eff)| for hole transport (occ. MOs) : 0.160 eV
 total |J(AB,eff)| for charge transport (unocc. MOs) : 0.132 eV
 charge transfer coupling : 0.093 eV
"""
        self.assertAlmostEqual(parse_dipro_hole_coupling_ev(output), 0.160)

    def test_rejects_command_echo_and_dipro_threshold(self):
        output = """
 program call : xtb dimer.xyz --gfn 2 --dipro 0.1 --chrg 0
 energy threshold for DIPRO: 0.100 eV
 normal termination of xtb
"""
        self.assertIsNone(parse_dipro_hole_coupling_ev(output))

    def test_rejects_electron_only_output(self):
        output = "total |J(AB,eff)| for electron transport (unocc. MOs) : 0.132 eV"
        self.assertIsNone(parse_dipro_hole_coupling_ev(output))

    def test_converts_mev_and_accepts_labelled_fixture(self):
        output = "hole transport (occ. MOs): 1->2 V=44 meV"
        self.assertAlmostEqual(parse_dipro_hole_coupling_ev(output), 0.044)

    def test_missing_or_unrecognised_hole_summary_fails_closed(self):
        self.assertIsNone(parse_dipro_hole_coupling_ev("DIPRO completed without summary"))


if __name__ == "__main__":
    unittest.main()
