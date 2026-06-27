import { useEffect, useMemo, useRef, useState } from 'react';
import type React from 'react';
import { getRealDatasets, runRealDatasetUploadTest, uploadArch } from '../api';
import { INPUT_STYLE } from '../theme';
import LooseNumberInput from '../components/LooseNumberInput';

type DatasetInfo = {
  id: string;
  label: string;
  kind: string;
  target_hint: string;
  source_url?: string;
  upload?: string;
};

const panel: React.CSSProperties = {
  background: 'var(--glass-bg)',
  border: '1px solid var(--glass-border)',
  borderRadius: 8,
  boxShadow: 'var(--window-shadow)',
};

const input = { ...INPUT_STYLE, width: '100%' };

export default function RealDatasetTester() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selected, setSelected] = useState<Record<string, boolean>>({
    california_housing: true,
    diabetes: true,
  });
  const [paths, setPaths] = useState<Record<string, string>>({});
  const [targets, setTargets] = useState<Record<string, string>>({});
  const [uploads, setUploads] = useState<Record<string, File[]>>({});
  const [arch, setArch] = useState<{ id: string; name: string } | null>(null);
  const [maxIter, setMaxIter] = useState(80);
  const [subsample, setSubsample] = useState(2000);
  const [testSize, setTestSize] = useState(0.3);
  const [status, setStatus] = useState('Load an architecture, then run selected datasets.');
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<Record<string, any> | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getRealDatasets()
      .then(data => setDatasets(data.datasets ?? []))
      .catch(err => setStatus(err.message));
  }, []);

  const selectedIds = useMemo(() => Object.entries(selected).filter(([, on]) => on).map(([id]) => id), [selected]);
  const averages = useMemo(() => averageResults(results), [results]);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setStatus(`Loading ${file.name}...`);
    try {
      const data = await uploadArch(file);
      setArch({ id: data.arch_id, name: file.name });
      setResults(null);
      setStatus(`Loaded ${file.name}.`);
    } catch (err: any) {
      setStatus(`Architecture upload failed: ${err.message ?? err}`);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const run = async () => {
    if (!arch || selectedIds.length === 0) return;
    setBusy(true);
    setResults(null);
    setStatus(`Running ${selectedIds.length} dataset test(s).`);
    try {
      const data = await runRealDatasetUploadTest({
        arch_id: arch.id,
        datasets: selectedIds,
        paths,
        target_columns: targets,
        max_iter: maxIter,
        subsample,
        test_size: testSize,
      }, uploads);
      setResults(data.results);
      setStatus('Real dataset test complete.');
    } catch (err: any) {
      setStatus(`Real dataset test failed: ${err.message ?? err}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto', height: '100%' }}>
      <div style={{ maxWidth: 1120, margin: '0 auto', padding: '32px 20px' }}>
        <h1 style={{ color: 'var(--text-primary)', fontSize: 28, margin: '0 0 8px' }}>Real Dataset Test</h1>
        <p style={{ color: 'var(--text-secondary)', margin: '0 0 22px' }}>
          Train one architecture on external regression tasks and compare its fit quality.
        </p>

        <section style={{ ...panel, padding: 18 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 12, alignItems: 'end' }}>
            <Field label="Architecture">
              <input ref={fileRef} type="file" accept=".pkl" onChange={handleUpload} style={{ display: 'none' }} />
              <button className="btn btn-back" onClick={() => fileRef.current?.click()} disabled={busy}>
                {arch ? arch.name : 'Load .pkl'}
              </button>
            </Field>
            <Field label="Max Iter">
              <LooseNumberInput style={input} min={1} value={maxIter} onChange={setMaxIter} fallback={1} />
            </Field>
            <Field label="Train Subsample">
              <LooseNumberInput style={input} min={16} value={subsample} onChange={setSubsample} fallback={16} />
            </Field>
            <Field label="Test Split">
              <LooseNumberInput style={input} min={0.05} max={0.8} step={0.05} value={testSize} onChange={setTestSize} fallback={0.2} />
            </Field>
            <button className="btn btn-primary" disabled={!arch || busy || selectedIds.length === 0} onClick={run}>
              {busy ? 'Running...' : 'Run Test'}
            </button>
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: 12, marginTop: 12 }}>{status}</div>
        </section>

        <section style={{ ...panel, padding: 18, marginTop: 18 }}>
          <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 14px' }}>Datasets</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12 }}>
            {datasets.map(ds => (
              <DatasetCard
                key={ds.id}
                dataset={ds}
                selected={!!selected[ds.id]}
                path={paths[ds.id] ?? ''}
                target={targets[ds.id] ?? ''}
                files={uploads[ds.id] ?? []}
                uploadLimit={subsample}
                onSelected={checked => setSelected(prev => ({ ...prev, [ds.id]: checked }))}
                onPath={value => setPaths(prev => ({ ...prev, [ds.id]: value }))}
                onTarget={value => setTargets(prev => ({ ...prev, [ds.id]: value }))}
                onFiles={files => setUploads(prev => ({ ...prev, [ds.id]: files }))}
              />
            ))}
          </div>
        </section>

        {results && (
          <section style={{ ...panel, padding: 18, marginTop: 18 }}>
            <h2 style={{ color: 'var(--text-primary)', fontSize: 17, margin: '0 0 14px' }}>Results</h2>
            {averages && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, marginBottom: 14 }}>
                {[
                  ['Datasets', averages.count],
                  ['Avg MSE', fmt(averages.mse)],
                  ['Avg MAE', fmt(averages.mae)],
                  ['Avg R2', fmt(averages.r2)],
                  ['Avg Score', fmt(averages.score)],
                  ['Total Fit', `${fmt(averages.fit_delay, 1)}s`],
                ].map(([label, value]) => (
                  <div key={label} style={{ border: '1px solid var(--glass-border)', borderRadius: 8, padding: 10, background: 'rgba(0,0,0,0.22)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>{label}</div>
                    <div style={{ color: 'var(--text-primary)', fontSize: 16, marginTop: 4 }}>{value}</div>
                  </div>
                ))}
              </div>
            )}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', color: 'var(--text-secondary)', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                    <th style={{ padding: 8 }}>Dataset</th>
                    <th style={{ padding: 8 }}>Status</th>
                    <th style={{ padding: 8 }}>MSE</th>
                    <th style={{ padding: 8 }}>MAE</th>
                    <th style={{ padding: 8 }}>R2</th>
                    <th style={{ padding: 8 }}>Score</th>
                    <th style={{ padding: 8 }}>Fit</th>
                    <th style={{ padding: 8 }}>Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {averages && (
                    <tr style={{ borderTop: '1px solid var(--glass-border)', background: 'rgba(var(--theme-primary-rgb),0.08)' }}>
                      <td style={{ padding: 8 }}>Average</td>
                      <td style={{ padding: 8, color: 'var(--success)' }}>{averages.count} ok</td>
                      <td style={{ padding: 8 }}>{fmt(averages.mse)}</td>
                      <td style={{ padding: 8 }}>{fmt(averages.mae)}</td>
                      <td style={{ padding: 8 }}>{fmt(averages.r2)}</td>
                      <td style={{ padding: 8 }}>{fmt(averages.score)}</td>
                      <td style={{ padding: 8 }}>{fmt(averages.fit_delay, 1)}s</td>
                      <td style={{ padding: 8 }}>-</td>
                    </tr>
                  )}
                  {Object.entries(results).map(([id, row]) => (
                    <tr key={id} style={{ borderTop: '1px solid var(--glass-border)' }}>
                      <td style={{ padding: 8 }}>{datasets.find(d => d.id === id)?.label ?? id}</td>
                      <td style={{ padding: 8, color: row.status === 'ok' ? 'var(--success)' : 'var(--danger)' }}>{row.status}</td>
                      <td style={{ padding: 8 }}>{fmt(row.mse)}</td>
                      <td style={{ padding: 8 }}>{fmt(row.mae)}</td>
                      <td style={{ padding: 8 }}>{fmt(row.r2)}</td>
                      <td style={{ padding: 8 }}>{fmt(row.score)}</td>
                      <td style={{ padding: 8 }}>{fmt(row.fit_delay, 1)}s</td>
                      <td style={{ padding: 8 }}>{row.status === 'ok' ? `${row.train_samples}/${row.test_samples}` : row.error}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: 'grid', gap: 5, color: 'var(--text-secondary)', fontSize: 12 }}>
      <span>{label}</span>
      {children}
    </label>
  );
}

function DatasetCard({
  dataset,
  selected,
  path,
  target,
  files,
  uploadLimit,
  onSelected,
  onPath,
  onTarget,
  onFiles,
}: {
  dataset: DatasetInfo;
  selected: boolean;
  path: string;
  target: string;
  files: File[];
  uploadLimit: number;
  onSelected: (checked: boolean) => void;
  onPath: (value: string) => void;
  onTarget: (value: string) => void;
  onFiles: (files: File[]) => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const isBuiltin = dataset.kind === 'sklearn';
  const fileSummary = files.length
    ? `${files.length} file${files.length === 1 ? '' : 's'} selected`
    : isBuiltin ? 'Built in' : 'No local files selected';
  const applyFiles = (selectedFiles: File[]) => {
    onFiles(sampleUploadFiles(dataset, selectedFiles, uploadLimit));
  };

  return (
    <div style={{ border: '1px solid var(--glass-border)', borderRadius: 8, padding: 14, background: 'rgba(0,0,0,0.22)', display: 'grid', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
        <input
          type="checkbox"
          checked={selected}
          onChange={e => onSelected(e.target.checked)}
          style={{ marginTop: 3 }}
        />
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ color: 'var(--text-primary)', fontSize: 14, fontWeight: 800 }}>{dataset.label}</div>
          <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 3 }}>{isBuiltin ? 'Bundled through sklearn' : dataset.target_hint}</div>
        </div>
        {dataset.source_url && (
          <a className="btn btn-back" href={dataset.source_url} target="_blank" rel="noreferrer" style={{ minHeight: 28, padding: '4px 8px', fontSize: 11 }}>
            Dataset
          </a>
        )}
      </div>

      {!isBuiltin && (
        <>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            {dataset.kind === 'csv' && (
              <>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv,.txt"
                  onChange={e => applyFiles(Array.from(e.target.files ?? []))}
                  style={{ display: 'none' }}
                />
                <button className="btn btn-back" onClick={() => fileRef.current?.click()} style={{ minHeight: 28, padding: '4px 8px', fontSize: 11 }}>
                  Upload CSV
                </button>
              </>
            )}
            <input
              ref={folderRef}
              type="file"
              multiple
              {...({ webkitdirectory: '', directory: '' } as any)}
              onChange={e => applyFiles(Array.from(e.target.files ?? []))}
              style={{ display: 'none' }}
            />
            <button className="btn btn-back" onClick={() => folderRef.current?.click()} style={{ minHeight: 28, padding: '4px 8px', fontSize: 11 }}>
              Upload Folder
            </button>
            <span style={{ color: files.length ? 'var(--text-secondary)' : 'var(--text-muted)', fontSize: 11 }}>{fileSummary}</span>
          </div>

          {dataset.kind === 'csv' && (
            <input
              style={input}
              placeholder="target column, optional"
              value={target}
              onChange={e => onTarget(e.target.value)}
            />
          )}

          <details>
            <summary style={{ color: 'var(--text-muted)', fontSize: 11, cursor: 'pointer' }}>Server path instead</summary>
            <input
              style={{ ...input, marginTop: 8 }}
              placeholder="Path on backend machine"
              value={path}
              onChange={e => onPath(e.target.value)}
            />
          </details>
        </>
      )}
    </div>
  );
}

function averageResults(results: Record<string, any> | null) {
  if (!results) return null;
  const ok = Object.values(results).filter((row: any) => row.status === 'ok');
  if (!ok.length) return null;
  const avg = (key: string) => {
    const vals = ok.map((row: any) => Number(row[key])).filter(Number.isFinite);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN;
  };
  const sum = (key: string) => {
    const vals = ok.map((row: any) => Number(row[key])).filter(Number.isFinite);
    return vals.reduce((a, b) => a + b, 0);
  };
  return {
    count: ok.length,
    mse: avg('mse'),
    mae: avg('mae'),
    r2: avg('r2'),
    score: avg('score'),
    fit_delay: sum('fit_delay'),
  };
}

function sampleUploadFiles(dataset: DatasetInfo, files: File[], limit: number) {
  if (dataset.kind !== 'image_folder') {
    return files;
  }
  const imageFiles = files.filter(file => {
    const rel = ((file as any).webkitRelativePath || file.name).split(/[\\/]/).pop() || file.name;
    return /\.(jpg|jpeg|png|bmp)$/i.test(rel) && /^\d{1,3}/.test(rel);
  });
  const maxFiles = Math.max(16, Math.floor(limit || 2000));
  return imageFiles
    .slice()
    .sort((a, b) => stableHash((a as any).webkitRelativePath || a.name) - stableHash((b as any).webkitRelativePath || b.name))
    .slice(0, maxFiles);
}

function stableHash(value: string) {
  let h = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function fmt(value: any, digits = 4) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-';
}
