# Vendored: GEMBA (GPT Estimation Metric Based Assessment)

- Source: https://github.com/MicrosoftTranslator/GEMBA
- Vendored: 2026-09-08, from `main` (`.git` history stripped per project convention).
- License: **CC-BY-SA-4.0** (see `LICENSE.md` in this directory) — separate from this project's
  own license, kept in place for attribution/share-alike compliance. This subtree is used
  unmodified except as noted below.

Used for the real GEMBA-MQM implementation (`gemba.utils.get_gemba_scores`), called from
`Internal_AST_Evaluation_Colab.ipynb`/`Internal_AST_Evaluation_Kaggle.ipynb`'s mục 3.5, replacing
this project's earlier custom/unvalidated LLM-judge prompt with the original authors' own
implementation. Requires `OPENAI_API_KEY` (set by the calling notebook, not stored here) and
`pip install openai pandas termcolor ipdb absl-py diskcache tqdm` (see `requirements.txt`).

**Not vendored** (present in the upstream repo, dropped here as unused/unneeded bloat, not
required by `get_gemba_scores`):
- `mt-metrics-eval-v2/` (~79MB of WMT22 benchmark eval data, unrelated to this project's use case)
- `gemba.egg-info/` (regenerable packaging build artifact)

**Local patches (deviate from upstream)**, both in `gemba/gpt_api.py`'s `GptApi.call_api()`,
both needed to use newer OpenAI reasoning-style models (gpt-5.x family) that the upstream repo
predates:
- Sent `max_tokens` (old param name) -- rejected with `Unsupported parameter: 'max_tokens'...
  Use 'max_completion_tokens' instead`. Changed to always send `max_completion_tokens`.
- Always sent `temperature` (GEMBA starts every call at `temperature=0`) -- rejected with
  `Unsupported value: 'temperature' does not support 0.0 with this model. Only the default (1)
  value is supported`. Changed to omit `temperature` entirely when it's 0, letting the model use
  its own default; still sent for GEMBA's retry-escalated non-zero temperatures.

**Local patch (deviates from upstream, real scoring bug)**: `gemba/gemba_mqm_utils.py`'s
`parse_mqm_answer()` only recognized "no error" via the exact substrings `"no-error"`/`"no error"`.
Modern models phrase this many other ways ("no other major errors", "no critical errors
identified", "no minor errors detected beyond the above") -- none matched, so these clearly
POSITIVE sentences fell through to the generic append-to-`errors[error_level]` branch and got
counted as a real error (usually `critical`, since they commonly follow a `Critical:` header),
capping the score at the -25 floor regardless of actual translation quality. Verified this was
happening on real GEMBA-MQM runs in this project (near-uniform -25.0 scores across visibly good
and bad translations alike). Fixed by requiring a line to start with one of the real MQM category
names (`accuracy`, `fluency`, `locale convention`, `style`, `terminology`, `non-translation`,
`other`, after stripping leading bullet chars) before counting it as an error at all -- matches
the exact format the prompt's own few-shot examples use for genuine error lines.
