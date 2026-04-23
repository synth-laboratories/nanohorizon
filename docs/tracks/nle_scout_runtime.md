# NLE Scout Runtime Track

- task: NetHack/NLE via `nle`
- environment family: `nle`
- canonical reward: `scout_score`
- runtime shim: `nanohorizon.nle_core.http_shim`
- local launcher: `scripts/run_nle_shim.sh`
- smoke config: `configs/nle_scout_smoke.yaml`

This first track validates the runtime, action catalog, scout reward, text rendering, and terminal image rendering. It does not define a training baseline.
