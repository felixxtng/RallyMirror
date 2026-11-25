import type { AnalysisMode } from './types';

export const analysisInfo: Record<
  AnalysisMode,
  { label: string; title: string; description: string; layers: string[] }
> = {
  combined: {
    label: 'Combined · all features',
    title: 'One combined view.',
    description: 'Court, players, body pose, rackets, shuttle and play phases in the same rally.',
    layers: [
      'Court geometry',
      'Player tracking',
      'Body pose',
      'Player-owned racket',
      'Shuttle trajectory',
      'Predicted play phases',
    ],
  },
  court: {
    label: 'Court detection',
    title: 'Court geometry.',
    description: 'Predicted court landmarks establish the floor layout and service areas.',
    layers: ['Court geometry'],
  },
  tracking: {
    label: 'Player tracking',
    title: 'Follow each player.',
    description: 'Player boxes and stable identities make movement easier to follow.',
    layers: ['Player tracking'],
  },
  pose: {
    label: 'Body pose',
    title: 'Movement, frame by frame.',
    description: 'Body landmarks and skeletons reveal how each player moves.',
    layers: ['Player tracking', 'Body pose'],
  },
  racket: {
    label: 'Racket detection',
    title: 'Connect player and racket.',
    description: 'Predicted racket centres are assigned to their player using pose context.',
    layers: ['Player tracking', 'Player-owned racket'],
  },
  shuttle: {
    label: 'Shuttle tracking',
    title: 'Follow the shuttle.',
    description: 'Shuttle detections form a short trail. Gaps remain unknown.',
    layers: ['Shuttle trajectory'],
  },
  phases: {
    label: 'Play phases',
    title: 'Read the rhythm of play.',
    description: 'Temporal predictions describe ready, movement, stroke, recovery and idle phases.',
    layers: ['Player tracking', 'Predicted play phases'],
  },
};

export const exampleVideo = (mode: AnalysisMode) => `/media/${mode}-analysis.mp4`;
export const examplePoster = (mode: AnalysisMode) => `/media/${mode}-analysis.jpg`;
export const exampleReport = (mode: AnalysisMode) => `/media/${mode}-report.json`;
