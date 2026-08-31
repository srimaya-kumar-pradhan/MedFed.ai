import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

export function LoginPage() {
  const { signIn } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const nav = useNavigate();
  const loc = useLocation() as { state?: { from?: string } };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await signIn(username, password);
      nav(loc.state?.from ?? "/dashboard", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-ink-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-6">
          <div className="text-2xl font-medium tracking-tight">MedFed AI</div>
          <div className="text-sm text-ink-500 mt-1">Privacy-Preserving Federated Learning for Medical Diagnostics</div>
        </div>

        <div className="card">
          <form onSubmit={onSubmit} className="space-y-3">
            <div>
              <label className="label block mb-1">Email</label>
              <input
                className="input"
                type="email"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoFocus
                required
                placeholder="dr.sharma@hospitala.com"
              />
            </div>
            <div>
              <label className="label block mb-1">Password</label>
              <input
                className="input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="demo123"
              />
            </div>

            {error && (
              <div className="border border-status-bad text-status-bad text-sm px-2 py-1.5 rounded-sm">
                {error}
              </div>
            )}

            <button type="submit" className="btn-primary w-full" disabled={loading}>
              {loading ? "Signing in…" : "Sign in"}
            </button>
          </form>
        </div>

        <div className="mt-4 text-xs text-ink-500 border border-ink-200 bg-paper p-3 rounded-md">
          <div className="label mb-1.5">Demo accounts</div>
          <table className="w-full">
            <tbody>
              <tr><td className="py-0.5 font-mono text-ink-700">dr.sharma@hospitala.com</td><td className="py-0.5 font-mono">demo123</td><td className="py-0.5 text-ink-500">Doctor · Hospital A</td></tr>
              <tr><td className="py-0.5 font-mono text-ink-700">dr.lee@hospitalc.com</td><td className="py-0.5 font-mono">demo123</td><td className="py-0.5 text-ink-500">Doctor · Hospital C</td></tr>
              <tr><td className="py-0.5 font-mono text-ink-700">researcher@institution1.com</td><td className="py-0.5 font-mono">research123</td><td className="py-0.5 text-ink-500">Researcher · Hospital B</td></tr>
              <tr><td className="py-0.5 font-mono text-ink-700">admin@hospitala.com</td><td className="py-0.5 font-mono">admin123</td><td className="py-0.5 text-ink-500">Institution Admin</td></tr>
              <tr><td className="py-0.5 font-mono text-ink-700">platform@medfed.ai</td><td className="py-0.5 font-mono">platform123</td><td className="py-0.5 text-ink-500">Platform Admin</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
