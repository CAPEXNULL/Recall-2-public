# Python API

Run `python examples/applicability_demo.py` from the repository root for a complete executable example using the production core and semantic validator. Recommendation and candidate providers are explicitly offline fixtures. The three assertions exercise CONFIRMED/TRANSFER, VIOLATED/REJECT, and UNKNOWN/REJECT.

For real provider calls, use `NebiusClient()` from `model_client` as the recommendation model and `Recall(NebiusClient(), semantic_registry=registry)` to opt into the semantic adapter. Configure `NEBIUS_API_KEY` and `NEBIUS_MODEL_ID` first. The registry declares entities, roles and typed relations; see `recall/canonical_registry_config.json`. Hermes extraction is configured in `recall/semantic_runtime_config.json` and uses the existing Nebius transport.

Save successful outcomes with `preconditions` and `structured_action`; typed preconditions declare subject, relation, operator and expected native type/value. Parameterized actions declare each parameter's name, role and mandatory flag. A role reference can be expressed as `{"role_ref": "declared_role_id"}` in the subject. Missing or ambiguous mandatory role bindings reject transfer. The demonstration uses a fixed action with no role parameters.

Explicit features control retrieval only; they do not confirm facts or bypass applicability. Missing current values cannot create false facts in the semantic adapter. The default legacy extractor has a different scope; use the explicit adapter when evaluating this guarantee.
