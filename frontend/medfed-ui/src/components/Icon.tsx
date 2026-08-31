/**
 * Lightweight, monochrome icon set. No emoji, no decorative graphics.
 * All icons rendered as 1.5px-stroke line drawings via inline SVG.
 */
import type { SVGProps } from "react";

const base = (props: SVGProps<SVGSVGElement>) => ({
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  ...props,
});

export const Icon = {
  Upload: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
  ),
  Image: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" /></svg>
  ),
  Play: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><polygon points="5 3 19 12 5 21 5 3" /></svg>
  ),
  Server: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><rect x="2" y="3" width="20" height="8" rx="1" /><rect x="2" y="13" width="20" height="8" rx="1" /><line x1="6" y1="7" x2="6.01" y2="7" /><line x1="6" y1="17" x2="6.01" y2="17" /></svg>
  ),
  Network: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><circle cx="12" cy="5" r="2" /><circle cx="5" cy="19" r="2" /><circle cx="19" cy="19" r="2" /><path d="M12 7v3M12 10l-5 6M12 10l5 6" /></svg>
  ),
  Check: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><polyline points="20 6 9 17 4 12" /></svg>
  ),
  X: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
  ),
  Clock: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><circle cx="12" cy="12" r="9" /><polyline points="12 7 12 12 15 14" /></svg>
  ),
  Layers: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" /></svg>
  ),
  Shield: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
  ),
  Logout: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" /></svg>
  ),
  Arrow: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" /></svg>
  ),
  Download: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
  ),
  Help: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><circle cx="12" cy="12" r="9" /><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
  ),
  Refresh: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" /><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" /></svg>
  ),
  Trash: (p: SVGProps<SVGSVGElement>) => (
    <svg {...base(p)}><polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
  ),
};
