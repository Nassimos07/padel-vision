import React from 'react';
import {Composition, staticFile} from 'remotion';
import {PadelVision} from './PadelVision';
import type {RenderPayload} from './types';

const DEFAULT_FPS = 30;
const DEFAULT_DURATION = 180;

const loadRenderMetadata = async () => {
  try {
    const response = await fetch(staticFile('render/frames.json'));
    if (!response.ok) {
      return {durationInFrames: DEFAULT_DURATION, fps: DEFAULT_FPS};
    }
    const payload = (await response.json()) as RenderPayload;
    return {
      durationInFrames: Math.max(1, payload.frames.length || DEFAULT_DURATION),
      fps: Math.max(1, Math.round(payload.fps || DEFAULT_FPS)),
    };
  } catch {
    return {durationInFrames: DEFAULT_DURATION, fps: DEFAULT_FPS};
  }
};

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="PadelVision"
        component={PadelVision}
        durationInFrames={DEFAULT_DURATION}
        fps={DEFAULT_FPS}
        width={1920}
        height={1080}
        calculateMetadata={loadRenderMetadata}
        defaultProps={{
          sourceWidth: 1463,
          sourceHeight: 812,
        }}
      />
      <Composition
        id="PadelVisionPreview"
        component={PadelVision}
        durationInFrames={90}
        fps={DEFAULT_FPS}
        width={1920}
        height={1080}
        defaultProps={{
          sourceWidth: 1463,
          sourceHeight: 812,
        }}
      />
    </>
  );
};

export default Root;
