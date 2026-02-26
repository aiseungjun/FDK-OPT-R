# Cardio Data Layout

Place CoronaryDominance data under this directory.

## Expected structure
- Root should include `anonymous_syntax_holdout/`.
- Dominance folders: `Left_Dominance/`, `Right_Dominance/`.
- Each study folder contains `LCA/` and/or `RCA/` with `.npz` files.
- Each `.npz` must include key `pixel_array` (shape `[T,H,W]` or `[H,W]`).

## Directory example
```
cardio_data/
  anonymous_syntax_holdout/
    Left_Dominance/
      <study_id>/
        LCA/
          frame_0001.npz
        RCA/
          frame_0002.npz
    Right_Dominance/
      <study_id>/
        LCA/
        RCA/
```
