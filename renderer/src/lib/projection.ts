import type {Point} from '../types';

export type Projector = {
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  sx: number;
  sy: number;
  point: (point: Point) => Point;
  valueX: (x: number) => number;
  valueY: (y: number) => number;
};

export const makeProjector = (
  width: number,
  height: number,
  sourceWidth: number,
  sourceHeight: number,
): Projector => {
  const sx = width / sourceWidth;
  const sy = height / sourceHeight;

  return {
    width,
    height,
    sourceWidth,
    sourceHeight,
    sx,
    sy,
    point: ([x, y]) => [x * sx, y * sy],
    valueX: (x) => x * sx,
    valueY: (y) => y * sy,
  };
};
