import type React from 'react';

export const MODULE_COLORS: Record<string, string> = {
  Input:              '#79ff6b',
  LearnableParameter: '#ff7a18',
  Add:                '#ffb000',
  MatMul:             '#ffb000',
  Mult:               '#ffb000',
  Activation:         '#ff4d2e',
  Normalizer:         '#ffd166',
  Pooling:            '#fff23d',
  Concat:             '#ff9f1c',
  SoftMax:            '#ff6b35',
  Transpose:          '#d48a38',
  Shift:              '#d48a38',
  Split:              '#ffe45e',
  EMA:                '#66ff99',
  Accumulator:        '#66ff99',
  EinsteinAggregator: '#ffcf5a',
  Constant:           '#f6a21a',
  Noise:              '#ff3864',
};

export const CARD_STYLE: React.CSSProperties = {
  background: 'var(--panel-bg)',
  backdropFilter: 'blur(14px)',
  border: '1px solid var(--glass-border)',
  borderRadius: '8px',
  boxShadow: 'var(--window-shadow)',
};

export const INPUT_STYLE: React.CSSProperties = {
  background: 'rgba(0, 0, 0, 0.62)',
  color: 'var(--text-primary)',
  border: '1px solid var(--glass-border)',
  padding: '5px 10px',
  borderRadius: '6px',
  fontSize: '13px',
  fontFamily: 'inherit',
  outline: 'none',
};

export const MODULE_SET_OPTIONS = [
  { value: 'Unified', label: 'Unified' },
  { value: 'Rich', label: 'Rich' },
  { value: 'All', label: 'All Modules' },
];
