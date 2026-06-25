import React from 'react';
import {AbsoluteFill} from 'remotion';

export const CourtBackdrop: React.FC = () => {
  return (
    <AbsoluteFill
      style={{
        overflow: 'hidden',
        background:
          'radial-gradient(circle at 50% 38%, rgba(255,255,255,0.12), transparent 22%), #111',
      }}
    >
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'linear-gradient(90deg, #ef3d20 0 19%, #1677d8 19% 81%, #ef3d20 81% 100%)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: '22%',
          right: '22%',
          top: '16%',
          bottom: '8%',
          border: '9px solid rgba(255,255,255,0.72)',
          boxShadow: 'inset 0 0 120px rgba(255,255,255,0.10)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: '49.8%',
          top: '16%',
          bottom: '8%',
          width: 5,
          background: 'rgba(255,255,255,0.75)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: '22%',
          right: '22%',
          top: '49.5%',
          height: 10,
          background: 'rgba(8,14,22,0.72)',
          boxShadow: '0 0 18px rgba(0,0,0,0.7)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'radial-gradient(120% 90% at 50% 42%, transparent 55%, rgba(2,4,8,.34) 100%)',
          mixBlendMode: 'multiply',
        }}
      />
    </AbsoluteFill>
  );
};
