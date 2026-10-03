# Deterministic current-value validation

The explicitly enabled semantic adapter uses the model only to extract typed candidates. No grounding or evidence-verifier model calls decide source support. `source_assertions.py` attests candidates using typed source records or a closed declarative grammar. `_resolve` and the existing evaluator decide CONFIRMED / VIOLATED / UNKNOWN.

Missing/null values, modal propositions, unsupported syntax, mismatched subjects/values and unavailable source evidence do not create facts. Explicit current negative states may supply false. Model assertion_mode is advisory and cannot authorize or veto a source-confirmed state.

The grammar accepts explicit field/value declarations, measurements, timestamps, complete collections, directed relations and role assignments. Field IDs and optional source_labels bind labels. Compound Boolean IDs admit noun/predicate shorthand. Declared entity kinds and generic type nouns may prefix subjects. Count descriptors must occur in the registry meaning of a count field. This is a conservative English declarative grammar, not an unrestricted natural-language interpreter. Unrecognised wording returns UNKNOWN. Typed records support native values independent of prose language; nonactual/unreported records cannot attest values.

Numeric identities use exact decimal representations: 90, 90.0 and 90.00 have one identity. Units use the existing canonical registry conversion. No new rounding/tolerance is introduced and raw facts remain intact.

Inspect hypotheses[*].validation_authority, current_value_state, current_value_observations, source_observations and state_validation_authority. Unsupported candidates retain failure reasons. Conflicting facts remain separate assertions with CONFLICT and evaluate UNKNOWN.

There is one semantic model request per extraction. Existing recommendation fallback remains separate. Default Recall without semantic_registry retains its existing extractor. Runtime/source changes require a new fingerprint and relevant rechecks. Historical results retain their original scope.
