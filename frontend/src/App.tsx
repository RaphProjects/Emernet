import { useState } from 'react';
import ArchViewer from './pages/ArchViewer';
import FightViewer from './pages/FightViewer';
import '@xyflow/react/dist/style.css';
import './App.css';

type Page = 'menu' | 'viewer' | 'saved' | 'fight' | 'tournament';

export default function App() {
  const [page, setPage] = useState<Page>('menu');

  return (
    <div className="app">
      <div className="topbar">
        <h2 className="topbar-title" onClick={() => setPage('menu')}>
          Emernet
        </h2>
        {page !== 'menu' && (
          <button className="btn btn-back" onClick={() => setPage('menu')}>
            ← Main Menu
          </button>
        )}
      </div>

      {page === 'menu' && <MainMenu onNavigate={setPage} />}
      {page === 'viewer' && <ArchViewer />}
      {page === 'saved' && <Placeholder title="Saved Architectures" />}
      {page === 'fight' && <FightViewer />}
      {page === 'tournament' && <Placeholder title="Launch a Tournament" />}
    </div>
  );
}

function MainMenu({ onNavigate }: { onNavigate: (p: Page) => void }) {
  const items: { page: Page; label: string; icon: string; desc: string }[] = [
    {
      page: 'viewer',
      label: 'Architecture Visualizer',
      icon: '🧬',
      desc: 'Generate and visualize random architectures',
    },
    {
      page: 'saved',
      label: 'Saved Architectures',
      icon: '💾',
      desc: 'Browse and load saved .pkl architectures',
    },
    {
      page: 'fight',
      label: 'Launch a Fight',
      icon: '⚔️',
      desc: 'Pit two architectures against each other',
    },
    {
      page: 'tournament',
      label: 'Launch a Tournament',
      icon: '🏆',
      desc: 'Run a full round-robin tournament',
    },
  ];

  return (
    <div className="menu-container">
      <h1 className="menu-title">Zero-Data Neural Architecture Search</h1>
      <p className="menu-subtitle">via Mutual Imitation</p>
      <div className="menu-grid">
        {items.map((item) => (
          <button
            key={item.page}
            className="menu-card"
            onClick={() => onNavigate(item.page)}
          >
            <span className="menu-card-icon">{item.icon}</span>
            <span className="menu-card-label">{item.label}</span>
            <span className="menu-card-desc">{item.desc}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function Placeholder({ title }: { title: string }) {
  return (
    <div className="placeholder">
      <h2>{title}</h2>
      <p>Coming soon</p>
    </div>
  );
}