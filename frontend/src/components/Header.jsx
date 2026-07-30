import React, { useEffect, useState } from "react";
import { Activity, Wifi, WifiOff, Gauge } from "lucide-react";
import { fetchHealth } from "../api/client";

export default function Header() {
  const [health, setHealth] = useState(null);
  const [online, setOnline] = useState(false);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      try {
        const data = await fetchHealth();
        if (mounted) {
          setHealth(data);
          setOnline(data.status === "ok");
        }
      } catch {
        if (mounted) {
          setHealth(null);
          setOnline(false);
        }
      }
    };
    poll();
    const interval = setInterval(poll, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="relative bg-pitwall-card border-b border-pitwall-border px-6 py-3">
      {/* Subtle top accent line */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-neon-cyan/40 to-transparent" />

      <div className="flex items-center justify-between">
        {/* Left: Branding */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-neon-cyan/10 border border-neon-cyan/30 flex items-center justify-center">
              <Gauge className="w-4.5 h-4.5 text-neon-cyan" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-wide text-slate-100 leading-tight">
                PIT-WALL <span className="text-neon-cyan">TELEMATICS</span>
              </h1>
              <p className="text-[10px] font-mono text-slate-500 tracking-widest uppercase">
                Surrogate Recourse Engine
              </p>
            </div>
          </div>

          {/* API Status Badge */}
          <div
            className={`neon-badge ${
              online
                ? "bg-neon-emerald/10 text-neon-emerald border border-neon-emerald/20"
                : "bg-neon-crimson/10 text-neon-crimson border border-neon-crimson/20"
            }`}
          >
            <span className="relative flex h-2 w-2">
              <span
                className={`absolute inline-flex h-full w-full rounded-full opacity-75 ${
                  online ? "bg-neon-emerald animate-ping" : "bg-neon-crimson"
                }`}
              />
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${
                  online ? "bg-neon-emerald" : "bg-neon-crimson"
                }`}
              />
            </span>
            {online ? "API CONNECTED" : "API OFFLINE"}
          </div>
        </div>

        {/* Right: Model + Track Info */}
        <div className="flex items-center gap-6 text-xs font-mono">
          {health && (
            <>
              <div className="flex flex-col items-end gap-0.5">
                <span className="text-slate-500 text-[10px] uppercase tracking-wider">
                  Model R²
                </span>
                <span className="text-neon-emerald font-semibold text-sm">
                  {health.model_holdout_r2 != null
                    ? health.model_holdout_r2.toFixed(4)
                    : "—"}
                </span>
              </div>
              <div className="w-px h-8 bg-pitwall-border" />
              <div className="flex flex-col items-end gap-0.5">
                <span className="text-slate-500 text-[10px] uppercase tracking-wider">
                  Trained
                </span>
                <span className="text-slate-300 text-[11px]">
                  {health.model_trained_at
                    ? new Date(health.model_trained_at).toLocaleDateString("en-GB", {
                        day: "2-digit",
                        month: "short",
                        year: "numeric",
                      })
                    : "—"}
                </span>
              </div>
            </>
          )}

          {!health && (
            <div className="flex items-center gap-2 text-slate-500">
              <WifiOff className="w-3.5 h-3.5" />
              <span>Waiting for backend…</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
