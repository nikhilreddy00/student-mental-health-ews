import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/layout/Layout'
import Overview from './pages/Overview'
import AlertCenter from './pages/AlertCenter'
import StudentProfile from './pages/StudentProfile'
import ModelPerformance from './pages/ModelPerformance'
import ShapExplainability from './pages/ShapExplainability'
import FairnessAnalysis from './pages/FairnessAnalysis'
import AICounselor from './pages/AICounselor'
import MultiAgentPlanner from './pages/MultiAgentPlanner'
import LiveRiskMonitor from './pages/LiveRiskMonitor'
import BookAppointment from './pages/BookAppointment'
import BookingQueue from './pages/BookingQueue'
import KBAssistant from './pages/KBAssistant'
import EarlyWarningAnalysis from './pages/EarlyWarningAnalysis'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/alerts" element={<AlertCenter />} />
            <Route path="/students" element={<StudentProfile />} />
            <Route path="/students/:studentId" element={<StudentProfile />} />
            <Route path="/model-performance" element={<ModelPerformance />} />
            <Route path="/shap" element={<ShapExplainability />} />
            <Route path="/fairness" element={<FairnessAnalysis />} />
            <Route path="/ai-counselor" element={<AICounselor />} />
            <Route path="/multi-agent" element={<MultiAgentPlanner />} />
            <Route path="/risk-monitor" element={<LiveRiskMonitor />} />
            <Route path="/book" element={<BookAppointment />} />
            <Route path="/bookings" element={<BookingQueue />} />
            <Route path="/kb" element={<KBAssistant />} />
            <Route path="/early-warning" element={<EarlyWarningAnalysis />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
