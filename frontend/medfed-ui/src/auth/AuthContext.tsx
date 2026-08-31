import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { auth, setToken, MedFedError } from "../api/client";
import type { PublicUser } from "../api/client";

type AuthState =
  | { status: "loading" }
  | { status: "anon" }
  | { status: "authed"; user: PublicUser };

type AuthContextValue = {
  state: AuthState;
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => void;
  hasPermission: (perm: string) => boolean;
};

const AuthContext = createContext<AuthContextValue | null>(null);

const ROLE_PERMS: Record<string, string[]> = {
  doctor: ["predict", "explain", "view_history", "view_model_info"],
  researcher: [
    "predict", "explain", "view_history", "view_model_info",
    "start_training", "stop_training", "view_training_runs",
    "view_model_registry", "view_nodes",
  ],
  institution_admin: [
    "predict", "explain", "view_history", "view_model_info",
    "start_training", "stop_training", "view_training_runs",
    "view_model_registry", "deploy_model", "archive_model",
    "view_nodes", "manage_nodes",
  ],
  platform_admin: ["*"],
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ status: "loading" });

  useEffect(() => {
    const token = localStorage.getItem("medfed_token");
    if (!token) {
      setState({ status: "anon" });
      return;
    }
    auth
      .me()
      .then((u) => setState({ status: "authed", user: u }))
      .catch(() => {
        setToken(null);
        setState({ status: "anon" });
      });
  }, []);

  const signIn = async (username: string, password: string) => {
    const res = await auth.login(username, password);
    setToken(res.access_token);
    setState({ status: "authed", user: res.user });
  };

  const signOut = () => {
    setToken(null);
    setState({ status: "anon" });
  };

  const hasPermission = (perm: string) => {
    if (state.status !== "authed") return false;
    const perms = ROLE_PERMS[state.user.role] ?? [];
    return perms.includes("*") || perms.includes(perm);
  };

  return (
    <AuthContext.Provider value={{ state, signIn, signOut, hasPermission }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}

export { MedFedError };
