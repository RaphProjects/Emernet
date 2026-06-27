import { useEffect, useState } from 'react';
import type React from 'react';
import ArchViewer from './pages/ArchViewer';
import FightViewer from './pages/FightViewer';
import TournamentViewer from './pages/TournamentViewer';
import About from './pages/About';
import RlSearchViewer from './pages/RlSearchViewer';
import CalibrationSettings from './pages/CalibrationSettings';
import PolicyInspector from './pages/PolicyInspector';
import RealDatasetTester from './pages/RealDatasetTester';
import EmernetLogo from './components/EmernetLogo';
import '@xyflow/react/dist/style.css';
import './App.css';

type Page = 'menu' | 'viewer' | 'fight' | 'tournament' | 'encoder' | 'evolve' | 'rlsearch' | 'calibration' | 'normalization' | 'speedbal' | 'policy' | 'realdata' | 'about';
type EffectLevel = 'full' | 'balanced' | 'low';

interface ThemeSettings {
  primary: string;
  accent: string;
  background: string;
  effects: EffectLevel;
}

const DEFAULT_THEME: ThemeSettings = {
  primary: '#ff7a18',
  accent: '#fff23d',
  background: '#050403',
  effects: 'balanced',
};

const THEME_STORAGE_KEY = 'emernet-theme-v2';

function hexToRgb(hex: string) {
  const normalized = hex.replace('#', '');
  const value = normalized.length === 3
    ? normalized.split('').map(c => c + c).join('')
    : normalized.padEnd(6, '0').slice(0, 6);
  const int = Number.parseInt(value, 16);
  return `${(int >> 16) & 255}, ${(int >> 8) & 255}, ${int & 255}`;
}

const PAGE_PATHS: Record<Page, string> = {
  menu: '/',
  viewer: '/viewer',
  fight: '/fight',
  tournament: '/tournament',
  encoder: '/train-arch-encoder',
  evolve: '/evolve-architecture',
  rlsearch: '/rlsearch',
  calibration: '/calibration',
  normalization: '/arena-normalization',
  speedbal: '/speed-balance',
  policy: '/policy-inspector',
  realdata: '/real-datasets',
  about: '/about',
};

function pathToPage(path: string): Page {
  for (const [page, p] of Object.entries(PAGE_PATHS)) {
    if (p === path) return page as Page;
  }
  return 'menu';
}

export default function App() {
  const [page, setPage] = useState<Page>(() => pathToPage(window.location.pathname));
  const [theme, setTheme] = useState<ThemeSettings>(() => {
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY);
      return saved ? { ...DEFAULT_THEME, ...JSON.parse(saved) } : DEFAULT_THEME;
    } catch {
      return DEFAULT_THEME;
    }
  });

  useEffect(() => {
    const root = document.documentElement;
    root.style.setProperty('--theme-primary', theme.primary);
    root.style.setProperty('--theme-primary-rgb', hexToRgb(theme.primary));
    root.style.setProperty('--theme-accent', theme.accent);
    root.style.setProperty('--theme-accent-rgb', hexToRgb(theme.accent));
    root.style.setProperty('--page-bg', theme.background);
    root.dataset.effects = theme.effects;
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(theme));
  }, [theme]);

  useEffect(() => {
    const handlePopState = () => setPage(pathToPage(window.location.pathname));
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const navigate = (p: Page) => {
    setPage(p);
    const url = PAGE_PATHS[p];
    if (window.location.pathname !== url) {
      window.history.pushState({ page: p }, '', url);
    }
  };

  return (
    <div className="app">
      <LightningBackdrop />
      <div className="topbar">
        <a className="topbar-title" href="/" onClick={e => {
          if (e.button === 0 && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
            e.preventDefault();
            navigate('menu');
          }
        }}>
          Emernet
        </a>
        <ThemeControls theme={theme} onThemeChange={setTheme} />
        {page !== 'menu' && (
          <a className="btn btn-back" href="/" onClick={e => {
            if (e.button === 0 && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
              e.preventDefault();
              navigate('menu');
            }
          }}>
            Back to Main Menu
          </a>
        )}
      </div>

      {page === 'menu' && <MainMenu onNavigate={navigate} />}
      {page === 'viewer' && <ArchViewer />}
      {page === 'fight' && <FightViewer />}
      {page === 'tournament' && <TournamentViewer />}
      {page === 'encoder' && <RlSearchViewer key="encoder" initialMode="train_encoder" />}
      {page === 'evolve' && <RlSearchViewer key="evolve" initialMode="evolve" />}
      {page === 'rlsearch' && <RlSearchViewer key="rlsearch" initialMode="train_encoder" />}
      {(page === 'calibration' || page === 'normalization' || page === 'speedbal') && <CalibrationSettings />}
      {page === 'policy' && <PolicyInspector />}
      {page === 'realdata' && <RealDatasetTester />}
      {page === 'about' && <About />}
    </div>
  );
}

const CARD_ACCENTS: Record<string, { bg: string; glow: string }> = {
  viewer:     { bg: 'rgba(var(--theme-primary-rgb), 0.16)', glow: 'rgba(var(--theme-primary-rgb), 0.26)' },
  fight:      { bg: 'rgba(255, 61, 0, 0.16)', glow: 'rgba(255, 122, 24, 0.26)' },
  tournament: { bg: 'rgba(255, 176, 0, 0.14)', glow: 'rgba(255, 176, 0, 0.24)' },
  encoder:    { bg: 'rgba(0, 200, 255, 0.14)', glow: 'rgba(0, 200, 255, 0.24)' },
  evolve:     { bg: 'rgba(98, 255, 138, 0.12)', glow: 'rgba(98, 255, 138, 0.22)' },
  rlsearch:   { bg: 'rgba(0, 200, 255, 0.14)', glow: 'rgba(0, 200, 255, 0.24)' },
  calibration: { bg: 'rgba(255, 176, 0, 0.14)', glow: 'rgba(255, 176, 0, 0.24)' },
  normalization: { bg: 'rgba(255, 176, 0, 0.14)', glow: 'rgba(255, 176, 0, 0.24)' },
  speedbal:   { bg: 'rgba(98, 255, 138, 0.12)', glow: 'rgba(98, 255, 138, 0.22)' },
  policy:     { bg: 'rgba(170, 120, 255, 0.14)', glow: 'rgba(170, 120, 255, 0.22)' },
  realdata:   { bg: 'rgba(0, 200, 255, 0.12)', glow: 'rgba(0, 200, 255, 0.20)' },
  about:      { bg: 'rgba(var(--theme-accent-rgb), 0.12)', glow: 'rgba(var(--theme-accent-rgb), 0.18)' },
};

function ThemeControls({
  theme,
  onThemeChange,
}: {
  theme: ThemeSettings;
  onThemeChange: React.Dispatch<React.SetStateAction<ThemeSettings>>;
}) {
  const update = (patch: Partial<ThemeSettings>) => onThemeChange(prev => ({ ...prev, ...patch }));

  return (
    <details className="theme-dock">
      <summary className="theme-trigger" title="Customize terminal theme">
        Theme
      </summary>
      <div className="theme-panel">
        <label>
          <span>Orange</span>
          <input type="color" value={theme.primary} onChange={e => update({ primary: e.target.value })} />
        </label>
        <label>
          <span>Burst</span>
          <input type="color" value={theme.accent} onChange={e => update({ accent: e.target.value })} />
        </label>
        <label>
          <span>Black</span>
          <input type="color" value={theme.background} onChange={e => update({ background: e.target.value })} />
        </label>
        <label>
          <span>FX</span>
          <select value={theme.effects} onChange={e => update({ effects: e.target.value as EffectLevel })}>
            <option value="full">Full</option>
            <option value="balanced">Balanced</option>
            <option value="low">Reduced</option>
          </select>
        </label>
        <button type="button" className="theme-reset" onClick={() => onThemeChange(DEFAULT_THEME)}>
          Reset
        </button>
      </div>
    </details>
  );
}

function LightningBackdrop() {
  const paths = [
    'M-60 118 C120 40 210 245 390 150 S650 58 850 170 S1110 250 1300 95',
    'M-80 430 C130 280 310 520 515 365 S840 210 1060 390 S1280 555 1460 305',
    'M80 760 C260 620 390 735 560 605 S880 470 1015 620 S1240 790 1440 585',
    'M-40 610 C150 500 250 645 430 540 S670 350 870 500 S1090 720 1330 545',
    'M170 80 C310 180 430 30 595 135 S810 335 1010 190 S1220 65 1420 210',
    'M20 245 C155 315 305 165 455 260 S735 470 905 330 S1155 160 1345 285',
  ];

  return (
    <svg className="network-backdrop" viewBox="0 0 1440 900" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <filter id="connectionGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      {paths.map((d, i) => (
        <path key={d} className={`network-arc arc-${i + 1}`} d={d} pathLength="1" />
      ))}
      {[...Array(18)].map((_, i) => {
        const x = 70 + ((i * 233) % 1320);
        const y = 70 + ((i * 149) % 760);
        return <circle key={i} className={`network-node node-${(i % 6) + 1}`} cx={x} cy={y} r="2.5" />;
      })}
    </svg>
  );
}

function MainMenu({ onNavigate }: { onNavigate: (p: Page) => void }) {
  const items: { page: Page; label: string; icon: React.ReactNode; desc: string }[] = [
    {
      page: 'viewer',
      label: 'Architecture\nVisualizer',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="18" cy="5" r="3"></circle>
          <circle cx="6" cy="12" r="3"></circle>
          <circle cx="18" cy="19" r="3"></circle>
          <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
          <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
        </svg>
      ),
      desc: 'Generate and visualize random neural architectures',
    },
    {
      page: 'fight',
      label: 'Launch\na Fight',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
        </svg>
      ),
      desc: 'Pit two architectures against each other in mutual imitation',
    },
    {
      page: 'tournament',
      label: 'Launch a\nTournament',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path>
          <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path>
          <path d="M4 22h16"></path>
          <path d="M10 14.66V20"></path>
          <path d="M14 14.66V20"></path>
          <path d="M18 4H6v7a6 6 0 0 0 12 0V4z"></path>
        </svg>
      ),
      desc: 'Full round-robin tournament to rank architectures',
    },
    {
      page: 'encoder',
      label: 'Train Arch\nEncoder',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M12 2v4m0 12v4m10-10h-4M6 12H2m15.07-7.07l-2.83 2.83M9.76 14.24l-2.83 2.83m12.14 0l-2.83-2.83M9.76 9.76L6.93 6.93"></path>
        </svg>
      ),
      desc: 'Train the GNN encoder from full tournament score prediction',
    },
    {
      page: 'evolve',
      label: 'Evolve\nArchitecture',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 12h4l3 8 4-16 3 8h4"></path>
          <path d="M17 5h4v4"></path>
          <path d="M21 5l-6 6"></path>
        </svg>
      ),
      desc: 'Start from a favorite architecture and evolve it with tournaments',
    },
    {
      page: 'realdata',
      label: 'Real Dataset\nTest',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 3v18h18"></path>
          <path d="M7 15l3-3 3 2 5-7"></path>
          <path d="M7 8h2"></path>
          <path d="M11 8h2"></path>
          <path d="M15 8h2"></path>
        </svg>
      ),
      desc: 'Train an architecture on selected real-world datasets',
    },
    {
      page: 'calibration',
      label: 'Arena\nCalibration',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 19V5"></path>
          <path d="M4 19h16"></path>
          <path d="M7 15l3-3 3 2 5-7"></path>
          <path d="M18 4v6"></path>
          <path d="M15 7h6"></path>
        </svg>
      ),
      desc: 'Normalize metrics and calibrate the speed reward',
    },
    {
      page: 'policy',
      label: 'Policy\nInspector',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="4" width="18" height="14" rx="2"></rect>
          <path d="M7 8h10"></path>
          <path d="M7 12h6"></path>
          <path d="M10 18v2"></path>
          <path d="M14 18v2"></path>
        </svg>
      ),
      desc: 'Inspect policy replay size and hybrid model weights',
    },
    {
      page: 'about',
      label: 'About\nEmernet',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
      ),
      desc: 'Concept, challenges, and future plans',
    },
  ];
  const groupFor: Record<Page, string> = {
    menu: '',
    viewer: 'Arena',
    fight: 'Arena',
    tournament: 'Arena',
    encoder: 'Training',
    evolve: 'Training',
    realdata: 'Training',
    rlsearch: 'Training',
    calibration: 'Tools',
    normalization: 'Tools',
    speedbal: 'Tools',
    policy: 'Tools',
    about: 'Tools',
  };
  const groups = ['Arena', 'Training', 'Tools'];

  return (
    <div className="menu-container">
      <EmernetLogo />
      <h1 className="menu-title">Zero-Data Neural Architecture Search</h1>
      <p className="menu-subtitle">via Mutual Imitation</p>

      <div className="menu-section-grid">
        {groups.map(group => (
          <section className="menu-section" key={group}>
            <h2>{group}</h2>
            <div className="menu-grid">
              {items.filter(item => groupFor[item.page] === group).map((item) => {
                const accent = CARD_ACCENTS[item.page];
                return (
                  <a
                    key={item.page}
                    className="menu-card"
                    href={PAGE_PATHS[item.page]}
                    onClick={e => {
                      if (e.button === 0 && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
                        e.preventDefault();
                        onNavigate(item.page);
                      }
                    }}
                    onMouseEnter={e => {
                      const card = e.currentTarget;
                      card.style.borderColor = 'rgba(var(--theme-accent-rgb), 0.42)';
                      card.style.boxShadow = `0 0 42px ${accent.glow}, 0 12px 36px rgba(0,0,0,0.28)`;
                    }}
                    onMouseLeave={e => {
                      const card = e.currentTarget;
                      card.style.borderColor = 'var(--glass-border)';
                      card.style.boxShadow = 'var(--window-shadow)';
                    }}
                  >
                    <span className="menu-card-icon" style={{ background: accent.bg }}>{item.icon}</span>
                    <span className="menu-card-label" style={{ whiteSpace: 'pre-line' }}>{item.label}</span>
                    <span className="menu-card-desc">{item.desc}</span>
                  </a>
                );
              })}
            </div>
          </section>
        ))}
      </div>

      <footer className="menu-footer">
        Project by Raphael Le Lain -{' '}
        <a href="mailto:raphael.le_lain@utt.fr">raphael.le_lain@utt.fr</a>
      </footer>
    </div>
  );
}
