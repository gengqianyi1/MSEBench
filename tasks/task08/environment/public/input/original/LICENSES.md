# Third-party components

Benchmark-authored source is under the accompanying MIT license. Dependencies
are installed separately and are not vendored in either release archive.

| Component | Frozen version | SPDX license | Purpose |
| --- | ---: | --- | --- |
| xTB | 6.7.1 | LGPL-3.0-or-later | Native quantum-chemistry execution |
| RDKit | 2025.03.5 | BSD-3-Clause | Geometry and XYZ support |
| OpenAI Python | 2.46.0 | Apache-2.0 | OpenAI-compatible transport |
| Pydantic | 2.13.4 | MIT | Strict schemas |
| PyYAML | 6.0.3 | MIT | Configuration loading |
| python-dotenv | 1.2.2 | BSD-3-Clause | Local environment loading |
| pytest | 9.1.1 | MIT | Test runner |

xTB scientific citation: C. Bannwarth, S. Ehlert, S. Grimme, *WIREs Comput.
Mol. Sci.* 2021, 11, e1493. DOI: `10.1002/wcms.1493`.

No model weights, xTB executable, Conda environment, provider response, or API
credential is included. Native xTB text appearing in frozen case evidence is
program output, not a redistributed executable or library.
