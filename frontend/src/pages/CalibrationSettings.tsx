import NormalizationSettings from './NormalizationSettings';
import SpeedBalanceSettings from './SpeedBalanceSettings';

export default function CalibrationSettings() {
  return (
    <div style={{ flex: 1, overflowY: 'auto', height: '100%' }}>
      <div style={{ maxWidth: 1120, margin: '0 auto', padding: '32px 20px', display: 'grid', gap: 22 }}>
        <header>
          <h1 style={{ color: 'var(--text-primary)', fontSize: 28, margin: '0 0 8px' }}>Calibration</h1>
          <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
            Tune the arena constants that shape scores: normalization values, raw opponent simplicity, and speed balance.
          </p>
        </header>

        <section>
          <NormalizationSettings embedded />
        </section>

        <section>
          <SpeedBalanceSettings embedded />
        </section>
      </div>
    </div>
  );
}
