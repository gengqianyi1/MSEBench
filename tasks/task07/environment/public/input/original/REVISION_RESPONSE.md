# Revision response — QED Molecular Design v4

The Ant review findings are closed in the revised v4 tree. The benchmark now has one public and four physically isolated organizer cases, a common output schema, hash-bound replay, and a fair one-round beam baseline with the same public information and budget as participants.

The original beam score of 53.9889 was traced to a uniform aromatic-halogen shortcut. A public `halogen_count` descriptor and per-case constraints remove that shortcut without hidden formatting or score scaling. The corrected deterministic beam score is 31.3753 (macro 21.8814 plus stability 9.4939).

The formal campaign is `working/formal_runs_v4_20260805_r3`: 15 real provider runs, exact model identities, no fallback, no BLOCKED cases, and byte-identical offline evaluator replay. Earlier `formal_runs_v4_20260805*` directories remain historical failure/calibration evidence and are not release results.

Participant material contains the public case and public evaluator only. Hidden cases and the complete formal evidence are organizer-only.
