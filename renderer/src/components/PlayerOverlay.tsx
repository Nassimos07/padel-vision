import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import type {Projector} from '../lib/projection';
import type {PlayerFrame} from '../types';

type Props = {
  player: PlayerFrame;
  index: number;
  projector: Projector;
  ringAlpha: number;
  layer: 'rings' | 'labels';
};

const ellipseLength = (a: number, b: number) => {
  return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
};

export const PlayerOverlay: React.FC<Props> = ({player, index, projector, ringAlpha, layer}) => {
  const frame = useCurrentFrame();
  const ui = projector.width / 3840;
  const scaled = (value: number) => value * ui;
  const [fx, fy] = projector.point(player.feet);
  const [hx, hy] = projector.point(player.head);
  const rx = projector.valueX(player.team === 'A' ? 50 : 34);
  const ry = projector.valueY(player.team === 'A' ? 19 : 12);
  const color = player.color;

  const entrance = interpolate(frame, [8 + index * 5, 25 + index * 5], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const yOffset = (1 - entrance) * scaled(20);
  const length = ellipseLength(rx, ry);
  const dash = length * 0.16;
  const dashOffset = -((frame / 90) * length);

  const cardW = scaled(292);
  const cardH = scaled(88);
  const margin = scaled(24);
  const cardX = Math.max(margin, Math.min(projector.width - cardW - margin, hx - cardW / 2));
  const cardY = Math.max(margin, hy - scaled(148));

  if (layer === 'rings') {
    return (
      <g opacity={entrance} transform={`translate(0 ${yOffset})`}>
        <defs>
          <radialGradient id={`floor-${player.id}`} cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={color} stopOpacity="0" />
            <stop offset="62%" stopColor={color} stopOpacity="0.10" />
            <stop offset="100%" stopColor={color} stopOpacity="0.34" />
          </radialGradient>
          <filter id={`glow-${player.id}`} x="-70%" y="-70%" width="240%" height="240%">
            <feGaussianBlur stdDeviation="12" />
          </filter>
        </defs>

        <ellipse cx={fx} cy={fy} rx={rx} ry={ry} fill={`url(#floor-${player.id})`} opacity={ringAlpha} />
        <ellipse
          cx={fx}
          cy={fy}
          rx={rx}
          ry={ry}
          fill="none"
          stroke={color}
          strokeWidth={scaled(9)}
          filter={`url(#glow-${player.id})`}
          opacity={0.44 * ringAlpha}
        />
        <ellipse cx={fx} cy={fy} rx={rx} ry={ry} fill="none" stroke={color} strokeWidth={scaled(3.5)} opacity={0.9 * ringAlpha} />
        <ellipse
          cx={fx}
          cy={fy}
          rx={rx}
          ry={ry}
          fill="none"
          stroke="#fff"
          strokeWidth={scaled(3.5)}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${length - dash}`}
          strokeDashoffset={dashOffset}
          opacity={0.85 * ringAlpha}
        />
        <circle cx={fx} cy={fy} r={scaled(4)} fill={color} opacity={ringAlpha} />

        {Array.from({length: 12}).map((_, tick) => {
          const angle = (tick / 12) * Math.PI * 2;
          const x1 = fx + Math.cos(angle) * rx;
          const y1 = fy + Math.sin(angle) * ry;
          const x2 = fx + Math.cos(angle) * (rx + Math.abs(Math.cos(angle)) * scaled(18) + scaled(8));
          const y2 = fy + Math.sin(angle) * (ry + Math.abs(Math.sin(angle)) * scaled(8) + scaled(5));
          return (
            <line
              key={tick}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={color}
              strokeWidth={scaled(2)}
              opacity={0.26 * ringAlpha}
            />
          );
        })}
      </g>
    );
  }

  return (
    <g opacity={entrance} transform={`translate(0 ${yOffset})`}>
      <line
        x1={fx}
        y1={fy}
        x2={hx}
        y2={hy + scaled(22)}
        stroke={color}
        strokeWidth={scaled(3)}
        strokeDasharray={`${scaled(2)} ${scaled(15)}`}
        strokeLinecap="round"
        opacity={0.32}
      />

      <circle cx={hx} cy={hy + scaled(18)} r={scaled(12)} fill="none" stroke={color} strokeWidth={scaled(3)} />
      <circle cx={hx} cy={hy + scaled(18)} r={scaled(5)} fill={color} />

      <line x1={hx} y1={cardY + cardH + scaled(16)} x2={hx} y2={hy + scaled(18)} stroke={color} strokeWidth={scaled(3)} opacity={0.55} />
      <path
        d={`M ${hx - scaled(13)} ${cardY + cardH} L ${hx + scaled(13)} ${cardY + cardH} L ${hx} ${cardY + cardH + scaled(17)} Z`}
        fill="rgba(11,16,24,0.92)"
      />
      <rect
        x={cardX}
        y={cardY}
        width={cardW}
        height={cardH}
        rx={scaled(14)}
        fill="rgba(11,16,24,0.86)"
        stroke="rgba(255,255,255,0.13)"
        filter="url(#soft-shadow)"
      />
      <rect x={cardX} y={cardY + scaled(3)} width={scaled(7)} height={cardH - scaled(6)} rx={scaled(3.5)} fill={color} />
      <text x={cardX + scaled(32)} y={cardY + scaled(59)} fill="#fff" fontFamily="monospace" fontSize={scaled(40)} fontWeight={800}>
        {player.id}
      </text>
      <line x1={cardX + scaled(120)} y1={cardY + scaled(18)} x2={cardX + scaled(120)} y2={cardY + cardH - scaled(18)} stroke="rgba(255,255,255,0.14)" strokeWidth={scaled(1.5)} />
      <text x={cardX + cardW - scaled(22)} y={cardY + scaled(36)} textAnchor="end" fill={color} fontSize={scaled(18)} fontWeight={800} letterSpacing={scaled(2)}>
        {player.state}
      </text>
      <text x={cardX + cardW - scaled(22)} y={cardY + scaled(67)} textAnchor="end" fill="#fff" fontFamily="monospace" fontSize={scaled(29)} fontWeight={800}>
        {player.speed.toFixed(1)}
        <tspan fill="rgba(244,247,251,0.55)" fontSize={scaled(15)}>
          {' '}
          km/h
        </tspan>
      </text>
    </g>
  );
};
