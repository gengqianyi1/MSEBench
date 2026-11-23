# Measurement contract and applicability

Formal scorer: `rdkit_descriptor_surrogate_v2`, RDKit 2023.09.6.

The C1 value is `-6.92 + 0.055*(O_count-4) - 0.035*F_count + 0.025*(rotors-4) + 0.035*logP`. It is a benchmark rule surrogate only. No calibration dataset or uncertainty supports interpreting it as HOMO energy, oxidation stability, high-temperature fast-charge performance, interfacial stability, or total electrolyte performance.

Input is RDKit-sanitized canonical SMILES. C/H/O/F, charge, size, fragment, descriptor, SMARTS, and atom-neighbour boundaries are deterministic. Carbon atoms with three or more oxygen neighbours are rejected by direct graph inspection. Invalid structures and reserved examples score zero. xTB is not a selectable formal scoring mode.

Exact v2 boundaries: elements C/H/O/F only; formal charge 0; one fragment; no radicals; 7-16 heavy atoms; MW <250; surrogate in (-7.20,-6.35); MW 155-210; logP [-0.2,0.9]; rotatable bonds 4-5; oxygen count 4-5; HBD 0; task SA score <2.8; fluorine count <=2; no rings/aromatic atoms; and none of epoxide, peroxide, acid halide, anhydride, or an sp3 carbon with at least three oxygen neighbours. The task SA rule is `1.2 + 0.06*heavy_atoms + 0.35*rings + 0.18*max(0, hetero_atoms-5) + 0.25*chiral_centres + 0.45*(spiro+bridgehead) + 0.10*max(0,F_count-3)`.

Reporting boundary: a score from this contract must be labelled `static_cheminformatics_compliance`. It is excluded from the Agent leaderboard/cross-task capability mean because the archived no-tool direct-answer baseline reached 100. The score only measures the frozen rule set and must not be reported as real HOMO, SEI, fast-charge, or electrolyte performance.
