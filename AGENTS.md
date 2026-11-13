# knn-lab

AI Lab coursework: K-Nearest Neighbours on Iris (scikit-learn built-in) and a generated fruit dataset. Shows how K and input scaling change the predictions. Streamlit demo in `app.py`.

## Layout

- `knn_lab/` this project: `spec.py` (dataset SPECs and models), `data.py` (loading and generating the data), `knn.py` (decision maps, neighbours, live K accuracy)
- `pxlab/` vendored copy of the shared engine (see `pxlab/VERSION`). Don't edit it here; it is replaced on the next sync.
- `main.py` runs the full study; `app.py` the demo; `app_overview.md` the demo's Overview text
- `outputs/` figures and CSV tables, `models/` saved pipelines and train/test splits (both committed for the hosted app)
- `docs/` methodology, results, deploy

## Commands

- Full study: `uv run python main.py`
- Demo: `uv run streamlit run app.py`
- Lint: `uv run ruff check . && uv run ruff format --check .`

## Conventions

- Beginner-student style on purpose: plain functions, docstrings on every function, comments explaining the AI concepts.
- `random_state=42` everywhere. Scaling is fitted inside the pipeline on training data only.
- After changing the study, re-run `main.py` and refresh `docs/results.md`, the README results and `app_overview.md`.
