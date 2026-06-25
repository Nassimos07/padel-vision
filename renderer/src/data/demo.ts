import type {FramePayload, PlayerFrame} from '../types';

const SOURCE_W = 1463;
const SOURCE_H = 812;

const BASE_PLAYERS: PlayerFrame[] = [
  {
    id: 'P1',
    team: 'A',
    color: '#ff3d8b',
    bbox: [430, 648, 506, 792],
    feet: [468, 792],
    head: [455, 648],
    speed: 18.4,
    state: 'SPRINT',
    move: [-0.55, -1],
  },
  {
    id: 'P2',
    team: 'A',
    color: '#ffb020',
    bbox: [920, 636, 996, 778],
    feet: [958, 778],
    head: [958, 636],
    speed: 12.1,
    state: 'COVER',
    move: [0.35, -1],
  },
  {
    id: 'P3',
    team: 'B',
    color: '#27e08a',
    bbox: [618, 232, 674, 332],
    feet: [646, 332],
    head: [646, 232],
    speed: 6.7,
    state: 'NET',
    move: null,
  },
  {
    id: 'P4',
    team: 'B',
    color: '#2e9bff',
    bbox: [774, 150, 826, 252],
    feet: [800, 252],
    head: [800, 150],
    speed: 9.3,
    state: 'STRIKE',
    move: [0.2, 1],
  },
];

const gaussian = (x: number, y: number, cx: number, cy: number, sx: number, sy: number) => {
  return Math.exp(-(((x - cx) ** 2) / sx + ((y - cy) ** 2) / sy));
};

const makeHeatmap = (players: PlayerFrame[], frame: number) => {
  const nx = 12;
  const ny = 8;
  const values = Array.from({length: ny}, () => Array.from({length: nx}, () => 0));

  for (const [idx, player] of players.entries()) {
    const [fx, fy] = player.feet;
    const cx = (fx / SOURCE_W) * nx - 0.5;
    const cy = (fy / SOURCE_H) * ny - 0.5;
    const wobble = 0.2 * Math.sin(frame / 18 + idx);

    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        values[y][x] += (1 + idx * 0.18) * gaussian(x, y, cx + wobble, cy, 2.4, 1.8);
      }
    }
  }

  const max = Math.max(1e-6, ...values.flat());
  return {nx, ny, values: values.map((row) => row.map((value) => value / max))};
};

export const getDemoFrame = (frame: number, fps: number): FramePayload => {
  const time = frame / fps;
  const players = BASE_PLAYERS.map((player, index) => {
    const sway = Math.sin(time * 1.85 + index) * 3.2;
    const lift = Math.sin(time * 2.4 + index * 0.5) * 1.2;
    const [fx, fy] = player.feet;
    const [hx, hy] = player.head;
    const [x1, y1, x2, y2] = player.bbox;

    return {
      ...player,
      bbox: [x1 + sway, y1 + lift, x2 + sway, y2 + lift] as PlayerFrame['bbox'],
      feet: [fx + sway, fy + lift] as PlayerFrame['feet'],
      head: [hx + sway * 0.8, hy + lift] as PlayerFrame['head'],
      speed: Math.max(0, player.speed + Math.sin(time * 3 + index) * 1.3),
    };
  });

  return {
    frame,
    time,
    sourceWidth: SOURCE_W,
    sourceHeight: SOURCE_H,
    players,
    heatmap: makeHeatmap(players, frame),
  };
};
