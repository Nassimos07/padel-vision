# Padel Vision Remotion Renderer

This is the V2 rendering layer. Python generates tracking data, masks, player
positions, and frame images. Remotion turns that data into a polished FHD video
with React/SVG/CSS.

## Install

```bash
cd renderer
npm install
```

## Export Real Tracking Data

Run the Python exporter from the repository root. It writes frame images, masks,
stable player IDs, positions, and heatmap snapshots into Remotion's public folder.

```bash
python scripts/export_remotion_data.py data/raw/padel_clip.mp4 --seconds 5 --stride 2
```

The default output is:

```text
renderer/public/render/
├── frames.json
├── frames/frame_000001.jpg
└── masks/frame_000001_p1.png
```

The renderer automatically uses `renderer/public/render/frames.json` when it
exists. If no export exists yet, it falls back to the built-in demo scene.

## Preview

```bash
npm run preview
```

## Render FHD Video

```bash
npm run render
```

For faster iteration, render the shorter preview composition:

```bash
npm run render:preview
```

Generated exports are ignored by git, so you can iterate without committing
large frame dumps.

## Data Contract

Each frame should eventually provide:

```json
{
  "frame": 120,
  "time": 4.0,
  "sourceWidth": 1463,
  "sourceHeight": 812,
  "image": "render/frames/frame_000120.jpg",
  "players": [
    {
      "id": "P1",
      "team": "A",
      "color": "#ff3d8b",
      "bbox": [430, 640, 500, 792],
      "feet": [468, 792],
      "head": [455, 648],
      "speed": 18.4,
      "state": "SPRINT",
      "move": [-0.55, -1],
      "mask": "render/masks/frame_000120_p1.png"
    }
  ],
  "heatmap": {
    "nx": 12,
    "ny": 8,
    "values": [[0, 0.1, 0.8]]
  }
}
```
