# Recent trajectory proxy eval

- This comparison uses the direct rollout path with a fake Craftax runner because the real `craftax` package is not installed in this checkout.
- The only code difference between baseline and candidate is the new recent-trajectory block in `src/nanohorizon/craftax_core/rollout.py`.
- The fake policy returns a higher-value craft action only when the prompt contains `Recent trajectory:`.
