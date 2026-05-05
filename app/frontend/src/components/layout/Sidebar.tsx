import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, AlertCircle, User, TrendingUp, GitBranch,
  Scale, MessageCircle, Users, Activity, CalendarPlus,
  ClipboardList, BookOpen, Clock, ChevronRight,
} from 'lucide-react'
import { useFilterStore } from '../../store/filterStore'

const NAV_ITEMS = [
  { path: '/',                 label: 'Overview',          icon: LayoutDashboard, end: true },
  { path: '/alerts',           label: 'Alert Center',      icon: AlertCircle },
  { path: '/students',         label: 'Student Profile',   icon: User },
  { path: '/model-performance',label: 'Model Performance', icon: TrendingUp },
  { path: '/shap',             label: 'SHAP Explainability',icon: GitBranch },
  { path: '/fairness',         label: 'Fairness Analysis', icon: Scale },
  { path: '/ai-counselor',     label: 'AI Counselor',      icon: MessageCircle },
  { path: '/multi-agent',      label: 'Multi-Agent Planner',icon: Users },
  { path: '/risk-monitor',     label: 'Risk Monitor',      icon: Activity },
  { path: '/book',             label: 'Book Appointment',  icon: CalendarPlus },
  { path: '/bookings',         label: 'Booking Queue',     icon: ClipboardList },
  { path: '/kb',               label: 'KB Assistant',      icon: BookOpen },
  { path: '/early-warning',    label: 'Early Warning',     icon: Clock },
]

const TIERS = ['High', 'Medium', 'Low']
const MODULES = ['All', 'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']

export default function Sidebar() {
  const { module, tiers, setModule, setTiers } = useFilterStore()

  function toggleTier(t: string) {
    setTiers(tiers.includes(t) ? tiers.filter(x => x !== t) : [...tiers, t])
  }

  return (
    <aside className="w-56 h-screen flex flex-col bg-base-100 border-r border-base-300 shrink-0 sticky top-0">

      {/* Brand */}
      <div className="px-4 py-5 border-b border-base-300">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-sm bg-primary flex items-center justify-center shrink-0">
            <Activity size={13} className="text-primary-content" />
          </div>
          <div>
            <div className="text-xs font-semibold text-base-content leading-tight tracking-wide">
              MH Early Warning
            </div>
            <div className="text-[10px] text-neutral-content mt-0.5 font-mono">
              AUC 0.975 · n=32,594
            </div>
          </div>
        </div>
      </div>

      {/* Global filters */}
      <div className="px-3 py-4 border-b border-base-300 space-y-3">
        <div>
          <div className="text-[10px] font-semibold text-neutral-content uppercase tracking-widest mb-1.5 px-1">
            Module
          </div>
          <select
            className="select select-xs select-bordered w-full bg-base-200 border-base-300 text-xs font-mono"
            value={module}
            onChange={e => setModule(e.target.value)}
          >
            {MODULES.map(m => <option key={m}>{m}</option>)}
          </select>
        </div>

        <div>
          <div className="text-[10px] font-semibold text-neutral-content uppercase tracking-widest mb-1.5 px-1">
            Risk Tier
          </div>
          <div className="space-y-1">
            {TIERS.map(t => (
              <label key={t} className="flex items-center gap-2 px-1 cursor-pointer group">
                <input
                  type="checkbox"
                  className="checkbox checkbox-xs checkbox-primary rounded-sm"
                  checked={tiers.includes(t)}
                  onChange={() => toggleTier(t)}
                />
                <span className="text-xs text-neutral-content group-hover:text-base-content transition-colors">{t}</span>
                <span className={`ml-auto w-1.5 h-1.5 rounded-full ${
                  t === 'High' ? 'bg-error' : t === 'Medium' ? 'bg-warning' : 'bg-success'
                }`} />
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-2">
        {NAV_ITEMS.map(({ path, label, icon: Icon, end }) => (
          <NavLink
            key={path}
            to={path}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2 mx-1.5 rounded-md text-xs font-medium transition-all ${
                isActive ? 'nav-active' : 'nav-item'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon size={14} className={isActive ? 'text-primary' : ''} />
                <span className="flex-1">{label}</span>
                {isActive && <ChevronRight size={10} className="text-primary opacity-60" />}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-base-300">
        <div className="text-[10px] text-neutral-content font-mono opacity-60 text-center">
          OULAD · Capstone 2024
        </div>
      </div>
    </aside>
  )
}
