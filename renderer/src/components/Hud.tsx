import React from 'react';
import {interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import type {FramePayload} from '../types';

type Props = {
  framePayload: FramePayload;
  locked: number;
};

export const Hud: React.FC<Props> = ({framePayload, locked}) => {
  const frame = useCurrentFrame();
  const {width} = useVideoConfig();
  const ui = width / 3840;
  const scaled = (value: number) => value * ui;
  const pulse = interpolate(Math.sin(frame / 8), [-1, 1], [0.35, 1]);
  const seconds = framePayload.time;
  const mm = Math.floor(seconds / 60);
  const ss = seconds % 60;

  return (
    <div
      style={{
        position: 'absolute',
        left: scaled(54),
        top: scaled(54),
        width: scaled(720),
        height: scaled(150),
        borderRadius: scaled(28),
        background: 'rgba(11,16,24,0.74)',
        border: '1px solid rgba(255,255,255,0.14)',
        boxShadow: `0 ${scaled(18)}px ${scaled(52)}px rgba(0,0,0,0.45)`,
        color: '#f4f7fb',
        padding: `${scaled(30)}px ${scaled(36)}px`,
      }}
    >
      <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
        <div style={{display: 'flex', alignItems: 'center', gap: scaled(18)}}>
          <div
            style={{
              width: scaled(18),
              height: scaled(18),
              borderRadius: 99,
              background: '#37e0c8',
              opacity: pulse,
              boxShadow: `0 0 ${scaled(28)}px #37e0c8`,
            }}
          />
          <div style={{fontSize: scaled(34), fontWeight: 800, letterSpacing: scaled(5)}}>PADEL ANALYTICS</div>
        </div>
        <div style={{fontSize: scaled(30), fontFamily: 'monospace'}}>
          {String(mm).padStart(2, '0')}:{ss.toFixed(1).padStart(4, '0')}
        </div>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: scaled(20),
          fontSize: scaled(24),
          letterSpacing: scaled(4),
          color: 'rgba(244,247,251,0.58)',
        }}
      >
        <span>LIVE TRACKING</span>
        <span style={{color: '#37e0c8', fontFamily: 'monospace'}}>{locked} / 4 LOCKED</span>
      </div>
    </div>
  );
};
