# Continuity

## State

v1.0.0: full study on Iris and fruit, Streamlit demo with Overview, Classify, Effect of K and Results tabs.

## Decisions

- D001 Fruit data is generated (`knn_lab/data.py`, FRUIT_PROFILES) because no clearly licensed public fruit table exists. Apples and oranges overlap on purpose.
- D002 The shared engine is vendored in `pxlab/` so the repo stands alone.
- D003 The best model is picked by cross-validation score, never by test score.

## Next

Nothing open.
