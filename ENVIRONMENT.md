# Environment and reproduction boundary

## Registered execution

- Fixed command: `bash run.sh`
- Node dispatch: [`config/node.json`](config/node.json)
- Locked environment: [`uv.lock`](uv.lock)
- Upstream implementation: `OswinMin/StCP@1d8df7614d49eada881426742688ba75fec631b9`
- Compute record: [`results/compute_provenance.json`](results/compute_provenance.json)

The pipeline uses committed node configuration rather than behavioural
environment variables. The source-trained conditional CDF and predictor are
required to be built before target labels are generated for C1.

## Recorded environments

The evidence payload records Python `3.12.11`, NumPy `2.5.1`, SciPy `1.18.0`,
and Torch `2.13.0`. Local runs record Darwin arm64; hosted runs record Linux
x86-64 with an 8-core cgroup quota. Thread pools were pinned to the available
quota where the provenance record includes that field.

## Completion boundary

This documentation pass does not launch the full experiment pipeline. The
committed evidence is the reproducibility snapshot already referenced by
`results/analysis.json`; `results/EVAL.md` exits with code `1` because C3 is
blocked by missing rate evidence. BIO/CRIME/DERMA real-data coverage is
complete in the integrated analysis, STAR is partial, TISSUE is incomplete,
and the merged LogAbs analysis does not contain the full `n=500` setting.

