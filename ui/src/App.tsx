import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import RoutingConfig from './pages/RoutingConfig'
import Governance from './pages/Governance'
import Guardrails from './pages/Guardrails'
import Prompts from './pages/Prompts'
import Observability from './pages/Observability'

export default function App() {
  return (
    <BrowserRouter basename="/ui">
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="routing" element={<RoutingConfig />} />
          <Route path="governance" element={<Governance />} />
          <Route path="guardrails" element={<Guardrails />} />
          <Route path="prompts" element={<Prompts />} />
          <Route path="observability" element={<Observability />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
