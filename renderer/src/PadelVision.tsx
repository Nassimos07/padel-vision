import React from 'react';
import {AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {CourtBackdrop} from './components/CourtBackdrop';
import {Hud} from './components/Hud';
import {PlayerOverlay} from './components/PlayerOverlay';
import {getDemoFrame} from './data/demo';
import {frameFromPayload, useRenderPayload} from './data/renderData';
import {makeProjector} from './lib/projection';

type Props = {
  sourceWidth: number;
  sourceHeight: number;
};

export const PadelVision: React.FC<Props> = ({sourceWidth, sourceHeight}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const {payload: renderPayload} = useRenderPayload();
  const payload = renderPayload ? frameFromPayload(renderPayload, frame) : getDemoFrame(frame, fps);
  const projector = makeProjector(
    width,
    height,
    payload.sourceWidth || sourceWidth,
    payload.sourceHeight || sourceHeight,
  );

  const ringAlpha = interpolate(frame, [125, 165], [1, 0.35], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill style={{backgroundColor: '#05070b', fontFamily: 'Inter, Arial, sans-serif'}}>
      {payload.image ? (
        <Img
          src={staticFile(payload.image)}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'fill',
          }}
        />
      ) : (
        <CourtBackdrop />
      )}
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{position: 'absolute', inset: 0, overflow: 'visible'}}
      >
        <defs>
          <filter id="soft-shadow" x="-60%" y="-60%" width="220%" height="220%">
            <feDropShadow dx="0" dy="16" stdDeviation="18" floodColor="#000" floodOpacity="0.45" />
          </filter>
          {payload.image
            ? payload.players.map((player) =>
                !player.cutout && player.mask ? (
                  <mask
                    key={`${player.id}-mask`}
                    id={`player-mask-${player.id}`}
                    maskUnits="userSpaceOnUse"
                    x={0}
                    y={0}
                    width={width}
                    height={height}
                  >
                    <rect x={0} y={0} width={width} height={height} fill="black" />
                    <image
                      href={staticFile(player.mask)}
                      x={0}
                      y={0}
                      width={width}
                      height={height}
                      preserveAspectRatio="none"
                    />
                  </mask>
                ) : null,
              )
            : null}
        </defs>
        {payload.players.map((player, index) => (
          <PlayerOverlay
            key={`${player.id}-rings`}
            player={player}
            index={index}
            projector={projector}
            ringAlpha={ringAlpha}
            layer="rings"
          />
        ))}
        {payload.image
          ? payload.players.map((player, index) => {
              if (!player.cutout && !player.mask) {
                return null;
              }
              const entrance = interpolate(frame, [8 + index * 5, 25 + index * 5], [0, 1], {
                extrapolateLeft: 'clamp',
                extrapolateRight: 'clamp',
              });
              if (player.cutout && player.cutoutBox) {
                const [x, y, cutoutWidth, cutoutHeight] = player.cutoutBox;
                return (
                  <image
                    key={`${player.id}-foreground`}
                    href={staticFile(player.cutout)}
                    x={projector.valueX(x)}
                    y={projector.valueY(y)}
                    width={projector.valueX(cutoutWidth)}
                    height={projector.valueY(cutoutHeight)}
                    preserveAspectRatio="none"
                    opacity={entrance}
                  />
                );
              }
              return (
                <image
                  key={`${player.id}-foreground`}
                  href={staticFile(payload.image as string)}
                  x={0}
                  y={0}
                  width={width}
                  height={height}
                  preserveAspectRatio="none"
                  mask={`url(#player-mask-${player.id})`}
                  opacity={entrance}
                />
              );
            })
          : null}
      </svg>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{position: 'absolute', inset: 0, overflow: 'visible'}}
      >
        <defs>
          <filter id="soft-shadow" x="-60%" y="-60%" width="220%" height="220%">
            <feDropShadow dx="0" dy="16" stdDeviation="18" floodColor="#000" floodOpacity="0.45" />
          </filter>
        </defs>
        {payload.players.map((player, index) => (
          <PlayerOverlay
            key={`${player.id}-labels`}
            player={player}
            index={index}
            projector={projector}
            ringAlpha={ringAlpha}
            layer="labels"
          />
        ))}
      </svg>
      <Hud framePayload={payload} locked={payload.players.length} />
    </AbsoluteFill>
  );
};
