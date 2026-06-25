import {continueRender, delayRender, staticFile} from 'remotion';
import {useEffect, useState} from 'react';
import type {FramePayload, RenderPayload} from '../types';

export const useRenderPayload = () => {
  const [handle] = useState(() => delayRender('Load exported render data'));
  const [payload, setPayload] = useState<RenderPayload | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;

    fetch(staticFile('render/frames.json'))
      .then(async (response) => {
        if (!response.ok) {
          return null;
        }
        return (await response.json()) as RenderPayload;
      })
      .then((data) => {
        if (!cancelled) {
          setPayload(data && data.frames.length > 0 ? data : null);
          setIsLoaded(true);
          continueRender(handle);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setIsLoaded(true);
          continueRender(handle);
          console.warn('Using demo data because exported render data was not found.', error);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [handle]);

  if (!isLoaded) {
    return {payload: null, isLoaded: false};
  }

  return {payload, isLoaded: true};
};

export const frameFromPayload = (
  payload: RenderPayload,
  frame: number,
): FramePayload => {
  const index = Math.min(Math.max(frame, 0), payload.frames.length - 1);
  return payload.frames[index];
};
