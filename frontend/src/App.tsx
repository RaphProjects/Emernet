import { useState } from 'react';
import ArchViewer from './pages/ArchViewer';
import FightViewer from './pages/FightViewer';
import TournamentViewer from './pages/TournamentViewer';
{/* Import the new page */}
import About from './pages/About'; 
import EmernetLogo from './components/EmernetLogo';
import '@xyflow/react/dist/style.css';
import './App.css';

{/* Add about to the Page type */}
type Page = 'menu' | 'viewer' | 'fight' | 'tournament' | 'about';

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
            Back to Main Menu
          </button>
        )}
      </div>

      {page === 'menu' && <MainMenu onNavigate={setPage} />}
      {page === 'viewer' && <ArchViewer />}
      {page === 'fight' && <FightViewer />}
      {page === 'tournament' && <TournamentViewer />}
      {/* Render the About page */}
      {page === 'about' && <About />} 
    </div>
  );
}

function MainMenu({ onNavigate }: { onNavigate: (p: Page) => void }) {
  {/* The array of menu items */}
  const items: { page: Page; label: string; icon: React.ReactNode; desc: string }[] = [
    {
      page: 'viewer',
      label: 'Architecture Visualizer',
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
      label: 'Launch a Fight',
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
        </svg>
      ),
      desc: 'Pit two architectures against each other in mutual imitation',
    },
    {
      page: 'tournament',
      label: 'Launch a Tournament',
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path>
          <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path>
          <path d="M4 22h16"></path>
          <path d="M10 14.66V20"></path>
          <path d="M14 14.66V20"></path>
          <path d="M18 4H6v7a6 6 0 0 0 12 0V4z"></path>
        </svg>
      ),
      desc: 'Run a full round-robin tournament to find the best architecture',
    },
    {
      page: 'about',
      label: 'About Emernet',
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
      ),
      desc: 'Learn about the concept, current challenges, and future plans',
    },
  ];

  return (
    <div className="menu-container">
      <EmernetLogo />
      <h1 className="menu-title">Zero-Data Neural Architecture Search</h1>
      <p className="menu-subtitle">via Mutual Imitation</p>
      
      {/* Change three-cols to two-cols for a clean grid */}
      <div className="menu-grid two-cols">
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
      
      <footer className="menu-footer">
        Project by Raphaël Le Lain -{' '}
        <a href="mailto:raphael.le_lain@utt.fr">raphael.le_lain@utt.fr</a>
      </footer>
    </div>
  );
}