# JHU Data Layout

Place JHU cadaver fluoroscopy data under this directory.

## Expected structure
- Patient folders must be named `cadaver_*`.
- Frames are discovered from image files under each `preview/` directory.
- Supported extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

## Directory example
```
jhu/
  cadaver_17-1882/
    fluoro/
      <session_or_view>/
        preview/
          0001.png
          0002.png
          ...
```
