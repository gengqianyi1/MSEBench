# Generic xTB constrained-scan syntax

For one constrained dihedral, a detailed input may use this form. Replace every uppercase placeholder before passing the file as `control_path` to `xtb_run` with `operation=scan`.

```text
$constrain
  force constant=FORCE_CONSTANT
  dihedral: ATOM_I, ATOM_J, ATOM_K, ATOM_L, auto
$scan
  1: START_DEG, STOP_DEG, NUMBER_OF_POINTS
$end
```

Atom indices are one-based. A decreasing start/stop range reverses direction. One job may contain one schedule line. Requested values and completed frames need not coincide, so inspect the actual geometries and energies.
