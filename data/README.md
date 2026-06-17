# Data

- **`raw/`** — put your input padel match videos here (`.mp4`, `.mov`, ...).
- **`processed/`** — annotated output videos and exported analytics land here.

Both folders' contents are **gitignored** — only the folder structure is tracked.

## Getting a clip

Use your own recording, or a clip you have the rights to share publicly (this repo
is meant for LinkedIn — avoid copyrighted broadcast footage). A 10–30 second rally
is perfect for a first test.

A small download helper is available (you supply the URL and confirm you have rights):

```bash
python scripts/download_sample.py "<video-url>"
```
