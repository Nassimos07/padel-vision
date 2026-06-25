import React from 'react';
import type {FramePayload, HeatmapFrame} from '../types';
import type {Projector} from '../lib/projection';

type Props = {
  payload: FramePayload;
  projector: Projector;
  alpha: number;
};

const palette = [
  [235, 255, 245],
  [120, 232, 218],
  [45, 166, 224],
  [70, 78, 190],
  [64, 38, 126],
  [13, 14, 32],
];

const mix = (a: number[], b: number[], t: number) => {
  return a.map((v, index) => Math.round(v * (1 - t) + b[index] * t));
};

const colorFor = (value: number) => {
  const scaled = Math.max(0, Math.min(1, value));
  const pos = scaled * (palette.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(palette.length - 1, lo + 1);
  const color = mix(palette[lo], palette[hi], pos - lo);
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
};

const flattenHeatmap = (heatmap?: HeatmapFrame) => {
  if (!heatmap) {
    return [];
  }
  const cells = [];
  for (let y = 0; y < heatmap.ny; y++) {
    for (let x = 0; x < heatmap.nx; x++) {
      cells.push({x, y, value: heatmap.values[y]?.[x] ?? 0});
    }
  }
  return cells;
};

export const Heatmap: React.FC<Props> = ({payload, projector, alpha}) => {
  const heatmap = payload.heatmap;
  if (!heatmap || alpha <= 0) {
    return null;
  }

  const cellW = projector.width / heatmap.nx;
  const cellH = projector.height / heatmap.ny;

  return (
    <svg
      width={projector.width}
      height={projector.height}
      viewBox={`0 0 ${projector.width} ${projector.height}`}
      style={{position: 'absolute', inset: 0, opacity: alpha}}
    >
      {flattenHeatmap(heatmap).map((cell) => {
        if (cell.value < 0.04) {
          return null;
        }
        const density = cell.value ** 0.72;
        return (
          <rect
            key={`${cell.x}-${cell.y}`}
            x={cell.x * cellW}
            y={cell.y * cellH}
            width={cellW}
            height={cellH}
            fill={colorFor(density)}
            opacity={0.08 + density * 0.54}
          />
        );
      })}
      {Array.from({length: heatmap.nx + 1}).map((_, index) => (
        <line
          key={`vx-${index}`}
          x1={index * cellW}
          y1={0}
          x2={index * cellW}
          y2={projector.height}
          stroke="rgba(255,255,255,0.22)"
          strokeWidth={2}
        />
      ))}
      {Array.from({length: heatmap.ny + 1}).map((_, index) => (
        <line
          key={`hy-${index}`}
          x1={0}
          y1={index * cellH}
          x2={projector.width}
          y2={index * cellH}
          stroke="rgba(255,255,255,0.22)"
          strokeWidth={2}
        />
      ))}
    </svg>
  );
};
