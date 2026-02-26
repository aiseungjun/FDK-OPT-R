# WEISS Data Layout

Place WEISS data files in this directory with the exact names below.

## Required files
- `T1T2.hdf5` (dataset key: `train_img`)
- `T3-T6.hdf5` (dataset key: `test`)

## Optional (used for pretrain mode)
- `Phantom.hdf5` (dataset key: `test`)

## Directory example
```
weiss/
  T1T2.hdf5
  T3-T6.hdf5
  Phantom.hdf5
```
