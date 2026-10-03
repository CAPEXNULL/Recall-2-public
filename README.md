# Recall 2

Outcome-backed action memory with conservative applicability checks. A successful past action transfers only when all mandatory Structured Preconditions are CONFIRMED and required action roles bind unambiguously. VIOLATED and UNKNOWN both produce REJECT and a fresh model recommendation. Similarity alone never authorizes transfer.

## Setup and run

Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r recall/requirements.txt
$env:NEBIUS_API_KEY = "<your key>"
$env:NEBIUS_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
$env:RECALL_SESSION_SECRET = "<random private session secret>"
cd recall
python app.py
```

Open http://127.0.0.1:5000. Supply credentials through environment variables; `.env.example` is a template and is not automatically loaded. The selected model must be enabled in your Nebius account. Live requests incur provider charges.

For hosting, from `recall/`: `waitress-serve --host=0.0.0.0 --port=8080 --call app:create_app`. Use one process, HTTPS termination and `RECALL_HTTPS=1`. Browser memory is isolated by session and volatile; restarting clears it.

## NVIDIA and Nebius integration

`recall/model_client.py` makes real HTTPS inference requests to Nebius Token Factory at `https://api.tokenfactory.nebius.com/v1`, using the configured NVIDIA Nemotron model for fresh recommendations. The development held-out run used `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`. No model weights are bundled; the public source contains the complete integration.

The optional semantic adapter uses `NousResearch/Hermes-4-405B` through the same Nebius transport to extract candidates. It is not an NVIDIA model. Candidate state validation and CONFIRMED / VIOLATED / UNKNOWN resolution run in code, without a verifier-model request.

## Applicability API and verification

The browser UI demonstrates NVIDIA recommendations and saving outcomes. It does not author mandatory preconditions, so saving a browser outcome alone does not demonstrate TRANSFER. Use the Python API for declared preconditions and structured actions: `Recall.decide(text, facts=None, features=None)` and `Recall.save_outcome(..., preconditions=..., structured_action=...)`. See [API usage](docs/API_USAGE.md) and [deterministic source rules](docs/CURRENT_VALUE_GROUNDING_USAGE.md).

From the repository root, run the reproducible, credential-free checks:

```powershell
python -m unittest discover -s recall/tests -p "test_source_rules.py" -v
python -m unittest discover -s recall/tests -p "test_explicit_features.py" -v
python examples/applicability_demo.py
```

The source-rule tests use an explicitly scripted candidate extractor and the real production validation code; they make no network calls and do not establish live model quality. The API demonstration checks TRANSFER, explicit false -> VIOLATED/REJECT, and missing current value -> UNKNOWN/REJECT.

## Scope and development evidence

The current deterministic semantic stage passed the existing 32 development scenarios, scoring and offline replay. This is not a new untouched generalization benchmark. The source validator supports a bounded declarative grammar; unsupported language, forecasts, missing values and conflicts fail closed to UNKNOWN. The default UI uses the legacy extractor; the new adapter requires an explicit semantic registry. Recommendations are not executed automatically.

[Hackathon updates](HACKATHON_UPDATES.md) describe the substantive changes recorded during the contest period. Internal author materials, provider response archives and historical experiments are not runtime dependencies and are excluded from the clean public package. Functional source, configs, assets and the judging checks are included.

## License and submission

MIT; see [LICENSE](LICENSE). A public repository, deployed working demo, video and Devpost submission still need their actual published URLs. Preparing this source package alone does not complete the competition submission.
