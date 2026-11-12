## Important Instructions
- **code readability** is paramount, do not compromise it for the sake of brevity or complexity

## General Instructions
- use Context7 when you are not sure or need official implementation details; if Context7 is unavailable, use the closest official upstream documentation source
- for anything touching LIBERO, robosuite, MuJoCo, or LeRobot, read the vendored sources under `third_party/` (and the notes in `notes/`) before guessing at an API
- keep the human-facing plan in `docs/ROADMAP.md`; treat it as a design document, not code

## Project Structure
```text
nano-vla/
├── pyproject.toml
├── main.py
├── datasets/          # LIBERO HDF5 data; ignored by Git
├── docs/              # ROADMAP and scratch docs; ignored by Git
├── notes/             # Working notes; ignored by Git
├── notebooks/         # Local notebooks; ignored by Git
├── outputs/           # Generated artifacts; ignored by Git
├── scripts/           # download_libero, replay_in_mujoco, replay_in_libero_env, infer_smolvla, plot_smolvla_gripper
├── tests/             # stdlib unittest suites (no simulator, torch, or network required)
└── third_party/       # Vendored LIBERO and LeRobot (editable installs); ignored by Git
```

- Keep new modules small and readable; add subpackages only when a module genuinely stops being cohesive.
- Keep tests in `tests/`, standalone utilities in `scripts/`, and generated artifacts out of Git.

## Environment and uv
- use `uv` instead of `pip`; use `uv add` to add dependencies and `uv run` for scripts, tests, and CLIs instead of calling `python` directly
- Python is pinned to 3.12; `torch`/`torchvision` come from the `pytorch-cu126` index and CUDA is required for the model-facing scripts
- `libero` and `lerobot` are editable path dependencies under `third_party/`; run `uv sync` after changing either
- `robosuite`, `mujoco`, `numpy`, and `numba` are pinned to a mutually compatible set; do not bump them casually, and keep `opencv-python-headless` as the only OpenCV distribution (see the `robosuite` dependency metadata comment)
- `third_party/` and `datasets/` are gitignored but must still be installed/synced locally before replay or training scripts will run

## Testing
- run the suite with `uv run python -m unittest discover -s tests -v` (all 22 tests run in well under a second)
- tests must stay dependency-light: use stdlib `unittest`, fake or mock the simulator, and never open an interactive MuJoCo viewer
- `scripts/replay_in_mujoco.py` isolates LIBERO config through `LIBERO_CONFIG_PATH`; preserve that behavior and its tests when editing the script

## LeRobot as an oracle, not scaffolding
- the target implementation does not import `lerobot`; use it to produce golden tensors and a known-good success rate, then test our own code against them
- when comparing against LeRobot, pin the exact commit and record the comparison in a test or note
- prefer the smaller LeRobot-free building blocks described in `docs/ROADMAP.md` over re-using its dataset/policy/processor plumbing

## Pi Sub-agents
- When the `subagent` tool is available, delegate autonomously when it improves efficiency; explicit user requests are not required.
- Available user-level agents: `scout` for codebase exploration, `planner` for implementation plans, `worker` for implementation and tests, and `reviewer` for read-only code review.
- Run independent exploration tasks in parallel. Use a planner for complex changes and a reviewer after substantial code changes; handle simple tasks directly.
- Give each agent a self-contained task with relevant context, exact file paths, constraints, and expected output. Sub-agents do not inherit the conversation.
- Sub-agents share the working directory: avoid concurrent edits to the same files, preserve existing changes, and verify delegated work before reporting completion.
- If you are already executing a delegated sub-agent task, complete it directly rather than recursively delegating.

## Python Best Practices
- do not use `dataclass` decorator
- do not use Ruff for linting or formatting
- use keyword-only arguments for functions/methods with multiple parameters: put a bare `*` after `self` (or after positional-only args), then name every remaining parameter so callers must pass them by keyword (e.g. `def append_step(self, *, obs: ..., action: ...) -> None`)
- keep a function's return annotation on the `def` line when the whole signature fits one line (e.g. `def make_dagger_round_seeds(*, seed: int, rounds: int) -> list[int]:`)
- when a signature is broken across lines, put the closing `)`, `-> ReturnType`, and `:` on the **same line as the last parameter** — not on a separate line (e.g. `    epoch: int,) -> None:` with a trailing comma, or `    epoch: int) -> None:` without one); this keeps the full header collapsible as one block in the editor

## GitHub README images
- GitHub can continue showing an older image when a README image is overwritten at the same URL. When changing a displayed PNG or GIF, give the updated image a new descriptive filename and update its README reference and generating script together; remove the obsolete image if it is no longer used.
- Before reporting an image update as live, verify the README reference and image blob on the pushed branch (which may not be the repository's default branch). Include the explicit branch URL when pointing someone to the result.

## Git Commits
- use conventional commits: `<type>: <short summary>` — one subject line only
- common types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `ci`, `build`, `perf`, `revert`
- do **not** add a commit body unless the user explicitly asks for one
- summary after `type:` must start with a lowercase letter, have no trailing period, and stay concise (about 6–12 words)
- use plain types only — no scopes (`feat(scope): ...` is wrong)
- prefer intent/outcome over implementation detail; avoid vague subjects like `update`, `changes`, `misc`
- before committing, check recent style with `git log --oneline -10` and match it
- if the message is wrong, fix it **before** pushing; never push then amend/force-push unless the user explicitly asks to rewrite published history

Examples:
- `feat: add live LIBERO action replay script`
- `fix: restore initial state before replaying actions`
- `docs: explain SmolVLA observation-to-action pipeline`

## Sections to ignore
Do not treat these as code changes; ignore them during development and review:
- `docs/` — manually curated documentation; not a code change
- `notes/` — working notes; not a code change
- `notebooks/` — local experiments; not a code change
- `ISSUES.md` — personal issue tracker; not a code change
- `datasets/` and `third_party/` — vendored/downloaded assets; not a code change
- `outputs/` — generated artifacts; not a code change
