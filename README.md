# Integrating Multi-Source Feedback in Computational Design

[![Paper](https://img.shields.io/badge/Paper-ACM%20TiiS-blue)](https://dl.acm.org/journal/tiis) 
[![DOI](https://img.shields.io/badge/DOI-10.1145/xxxxxxx-red)](https://doi.org/10.1145/xxxxxxx)
[![License](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the code implementation for our paper:

> **"Integrating Multi-Source Feedback in Computational Design"**  
> *Accepted for publication at ACM Transactions on Interactive Intelligent Systems (ACM TiiS)*

The paper is published under a **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

## Authors

- **Francisco Erivaldo Fernandes Junior** — Instituto Tecnológico de Aeronáutica, Brazil  
  [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--2301--8820-green)](https://orcid.org/0000-0003-2301-8820)

- **Thomas Langerak** — Aalto University, Finland  
  [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--2536--0208-green)](https://orcid.org/0000-0003-2536-0208)

- **Mira Keränen** — Aalto University, Finland  
  [![ORCID](https://img.shields.io/badge/ORCID-0009--0008--6234--2694-green)](https://orcid.org/0009-0008-6234-2694)

- **Danqing Shi** — Lund University, Sweden  
  [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8105--0944-green)](https://orcid.org/0000-0002-8105-0944)

- **Ardak Alipova** — Nazarbayev University, Kazakhstan  
  [![ORCID](https://img.shields.io/badge/ORCID-0009--0004--3653--9670-green)](https://orcid.org/0009-0004-3653-9670)

- **Antti Oulasvirta** — Aalto University, Finland  
  [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--2498--7837-green)](https://orcid.org/0000-0002-2498-7837)

---

## About

This repository provides the source code for the experiments and analyses presented in the paper. It includes:

- `backend/` — orchestration, evaluators, logging.
- `frontend/` — React UI to configure runs and view results.
- `technical_evaluation/` — small scripts for sanity checks.
- `backend/third_party_modules/` — vendored code we did **not** author.

## Disclaimer

**This code is provided "as-is" without any warranty or support.**

Due to our limited time and other commitments, we are unable to provide technical support, answer questions, or address issues related to this codebase. The code is shared for transparency and reproducibility of the research results presented in the paper. You are welcome to use, adapt, and modify it under the terms of the MIT license.

For any questions about the research itself, please refer to the paper.

## Running (very brief)
- Start the backend (see scripts in `backend/`).
- Start the frontend (see `frontend/package.json`).
- Results are written under `backend/run/<user_id>/...` and viewable in the UI.

## Citation

If you find this work useful, please cite:

```bibtex
@article{fernandes2026integrating,
  title={Integrating Multi-Source Feedback in Computational Design},
  author={Fernandes Junior, Francisco Erivaldo and Langerak, Thomas and Keränen, Mira and Shi, Danqing and Alipova, Ardak and Oulasvirta, Antti},
  journal={ACM Transactions on Interactive Intelligent Systems},
  year={2026},
  publisher={ACM}
}
```
