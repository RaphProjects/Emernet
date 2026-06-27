import { useState, useEffect } from 'react';
import { getNodeParams } from '../api';

const HIDDEN_PARAMS = new Set([
  'n_parameters', 'norm_layers', 'projections', 'p_projections', 'f_projections',
  'raw_alpha', 'raw_sharpness', 'raw_symmetry', 'raw_gate', 'value',
]);

interface NodeInspectorProps {
  archId: string;
  nodeId: number;
  pendingChanges: Record<string, any>;
  onParamChange: (nodeId: number, key: string, value: any) => void;
  onClose: () => void;
}

export default function NodeInspector({ archId, nodeId, pendingChanges, onParamChange, onClose }: NodeInspectorProps) {
  const [nodeInfo, setNodeInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadParams();
  }, [archId, nodeId]);

  const loadParams = async () => {
    setLoading(true);
    try {
      const info = await getNodeParams(archId, nodeId);
      setNodeInfo(info);
    } catch (e) {
      console.error('Failed to load node params:', e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="glass-panel" style={{ width: '280px', padding: '16px', position: 'absolute', right: '12px', top: '60px', zIndex: 20 }}>
        <p style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>Loading...</p>
      </div>
    );
  }

  if (!nodeInfo) return null;

  const editableParams = Object.entries(nodeInfo.params || {}).filter(
    ([key]) => !HIDDEN_PARAMS.has(key) && !key.startsWith('_')
  );

  return (
    <div className="glass-panel" style={{ width: '280px', padding: '16px', position: 'absolute', right: '12px', top: '60px', zIndex: 20, display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ color: 'var(--text-primary)', fontSize: '14px', margin: 0, fontWeight: 800 }}>Node {nodeId}</h3>
        <button
          onClick={onClose}
          style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '16px', padding: '2px 6px', borderRadius: '4px' }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(var(--theme-primary-rgb),0.12)'}
          onMouseLeave={e => e.currentTarget.style.background = 'none'}
        >
          X
        </button>
      </div>

      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
        <div>Type: <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{nodeInfo.module_type}</span></div>
        <div>Mapping: <span style={{ color: 'var(--text-primary)' }}>{nodeInfo.mapping_type}</span></div>
      </div>

      {editableParams.length > 0 && (
        <>
          <div style={{ borderTop: '1px solid var(--glass-border)', margin: '4px 0' }} />
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 800, textTransform: 'uppercase' }}>Parameters</div>
        </>
      )}

      {editableParams.map(([key, originalValue]) => {
        const editValue = pendingChanges?.[key] !== undefined ? pendingChanges[key] : originalValue;
        const changed = pendingChanges?.[key] !== undefined;
        return (
          <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <label style={{ fontSize: '11px', color: changed ? 'var(--theme-accent)' : 'var(--text-muted)' }}>{key}</label>
            {typeof originalValue === 'boolean' ? (
              <button
                onClick={() => onParamChange(nodeId, key, !editValue)}
                style={{
                  padding: '4px 10px',
                  borderRadius: '6px',
                  border: '1px solid var(--glass-border)',
                  fontSize: '12px',
                  cursor: 'pointer',
                  textAlign: 'left',
                  background: editValue ? 'rgba(var(--theme-primary-rgb), 0.2)' : 'rgba(0, 0, 0, 0.6)',
                  color: editValue ? 'var(--theme-accent)' : 'var(--text-secondary)',
                  fontWeight: editValue ? 700 : 400,
                  fontFamily: 'inherit',
                  transition: 'all 0.15s',
                }}
              >
                {String(editValue)}
              </button>
            ) : (
              <input
                type={typeof originalValue === 'number' ? 'number' : 'text'}
                value={String(editValue ?? '')}
                onChange={e => {
                  const raw = e.target.value;
                  const parsed = typeof originalValue === 'number' ? (raw === '' ? '' : Number(raw)) : raw;
                  onParamChange(nodeId, key, parsed);
                }}
                style={{
                  padding: '5px 10px',
                  borderRadius: '6px',
                  border: changed ? '1px solid rgba(var(--theme-accent-rgb), 0.34)' : '1px solid var(--glass-border)',
                  background: 'rgba(0, 0, 0, 0.6)',
                  color: 'var(--text-primary)',
                  fontSize: '12px',
                  fontFamily: 'inherit',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                }}
                onFocus={e => e.target.style.borderColor = 'rgba(var(--theme-primary-rgb), 0.42)'}
                onBlur={e => e.target.style.borderColor = changed ? 'rgba(var(--theme-accent-rgb), 0.34)' : 'var(--glass-border)'}
              />
            )}
          </div>
        );
      })}

      {editableParams.length === 0 && (
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '8px 0' }}>
          No editable parameters
        </div>
      )}
    </div>
  );
}
