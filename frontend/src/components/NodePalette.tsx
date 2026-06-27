import { useRef } from 'react';
import type React from 'react';
import { MODULE_COLORS } from '../theme';

const MODULE_TYPES = [
  'Activation', 'Normalizer', 'Pooling', 'SoftMax',
  'Add', 'Mult', 'MatMul', 'Concat', 'Split', 'Transpose', 'Shift',
  'EMA', 'Accumulator', 'EinsteinAggregator', 'LearnableParameter', 'Constant',
  'Noise',
];

interface NodePaletteProps {
  onAddNode: (moduleType: string) => void;
  onClose: () => void;
  onImportArch?: (file: File) => void;
}

export default function NodePalette({ onAddNode, onClose, onImportArch }: NodePaletteProps) {
  const importFileRef = useRef<HTMLInputElement>(null);

  const handleImportFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !onImportArch) return;
    onImportArch(file);
    if (importFileRef.current) importFileRef.current.value = '';
  };

  return (
    <div style={{
      background: 'var(--panel-bg)',
      backdropFilter: 'blur(14px)',
      border: '1px solid var(--glass-border)',
      borderRadius: '8px',
      boxShadow: 'var(--window-shadow)',
      padding: '12px',
      position: 'absolute',
      left: '12px',
      top: '60px',
      zIndex: 20,
      width: '180px',
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
        <span style={{ color: 'var(--text-primary)', fontSize: '13px', fontWeight: 800 }}>Add Module</span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '14px' }}>X</button>
      </div>

      {MODULE_TYPES.map(type => (
        <button
          key={type}
          onClick={() => onAddNode(type)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '5px 8px',
            borderRadius: '6px',
            border: '1px solid transparent',
            background: 'rgba(0, 0, 0, 0.48)',
            color: 'var(--text-primary)',
            cursor: 'pointer',
            fontSize: '11px',
            fontWeight: 700,
            fontFamily: 'inherit',
            transition: 'background 0.15s, border-color 0.15s',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = 'rgba(var(--theme-primary-rgb), 0.18)';
            e.currentTarget.style.borderColor = 'var(--glass-border)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = 'rgba(0, 0, 0, 0.48)';
            e.currentTarget.style.borderColor = 'transparent';
          }}
        >
          <span style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            background: MODULE_COLORS[type] ?? 'var(--text-secondary)',
            flexShrink: 0,
            boxShadow: '0 0 8px currentColor',
          }} />
          {type}
          {type === 'Noise' && (
            <span style={{
              marginLeft: 'auto',
              fontSize: '8px',
              fontWeight: 800,
              color: '#ff3864',
              background: 'rgba(255,56,100,0.15)',
              padding: '1px 5px',
              borderRadius: '4px',
              textTransform: 'uppercase',
            }}>rare</span>
          )}
        </button>
      ))}

      <div style={{ height: '1px', background: 'var(--glass-border)', margin: '6px 0' }} />

      <input ref={importFileRef} type="file" accept=".pkl" onChange={handleImportFile} style={{ display: 'none' }} />
      <button
        onClick={() => importFileRef.current?.click()}
        disabled={!onImportArch}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 8px',
          borderRadius: '6px',
          border: '1px solid rgba(var(--theme-accent-rgb), 0.2)',
          background: 'rgba(var(--theme-primary-rgb), 0.14)',
          color: 'var(--theme-accent)',
          cursor: onImportArch ? 'pointer' : 'not-allowed',
          fontSize: '11px',
          fontWeight: 700,
          fontFamily: 'inherit',
          transition: 'background 0.15s',
          opacity: onImportArch ? 1 : 0.4,
        }}
        onMouseEnter={e => { if (onImportArch) e.currentTarget.style.background = 'rgba(var(--theme-primary-rgb), 0.26)'; }}
        onMouseLeave={e => { if (onImportArch) e.currentTarget.style.background = 'rgba(var(--theme-primary-rgb), 0.14)'; }}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        Import .pkl
      </button>
    </div>
  );
}
