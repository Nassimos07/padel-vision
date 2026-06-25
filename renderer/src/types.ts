export type Point = [number, number];

export type PlayerState = 'SPRINT' | 'COVER' | 'NET' | 'STRIKE' | 'TRACKING';

export type PlayerFrame = {
  id: string;
  team: 'A' | 'B';
  color: string;
  bbox: [number, number, number, number];
  cutout?: string;
  cutoutBox?: [number, number, number, number];
  feet: Point;
  head: Point;
  speed: number;
  state: PlayerState;
  move?: Point | null;
  mask?: string;
};

export type HeatmapFrame = {
  nx: number;
  ny: number;
  values: number[][];
};

export type FramePayload = {
  frame: number;
  sourceFrame?: number;
  time: number;
  sourceWidth: number;
  sourceHeight: number;
  image?: string;
  players: PlayerFrame[];
  heatmap?: HeatmapFrame;
};

export type RenderPayload = {
  fps: number;
  sourceFps: number;
  sourceWidth: number;
  sourceHeight: number;
  frames: FramePayload[];
};
