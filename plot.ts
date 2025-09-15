/*
 TypeScript plotting helpers that mirror schrodinger_1d_lab1.py plotting:
 - renderPotentialAndStatesSVG: plots V(x) in eV and overlays the first K eigenstates
   as |ψ|^2 (scaled) above their energy lines E_n (in eV), similar to the python plots.

 Inputs expect SI units: V in Joules, energies in Joules, x in meters.
*/

export type RenderOptions = {
  showPotential?: boolean;
  showProb?: boolean;
  showWave?: boolean;
  showEnergy?: boolean;
  colors?: string[];
};

const E_CHARGE = 1.602176634e-19; // J/eV
const NM = 1e-9;

function ensureEl(container: string | HTMLElement): HTMLElement {
  if (typeof container === 'string') {
    const el = document.getElementById(container);
    if (!el) throw new Error(`Container not found: ${container}`);
    return el;
  }
  return container;
}

export function renderPotentialAndStatesSVG(
  container: string | HTMLElement,
  x: number[],
  V: number[],
  energiesJ: number[],
  psi: number[][],
  title: string,
  opts: RenderOptions = {}
): void {
  const wrap = ensureEl(container);
  wrap.innerHTML = '';
  const W = wrap.clientWidth || 1000;
  const H = wrap.clientHeight || 560;

  const svgns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgns, 'svg');
  svg.setAttribute('width', String(W));
  svg.setAttribute('height', String(H));
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.style.background = '#ffffff';
  wrap.appendChild(svg);

  const showPotential = opts.showPotential !== false;
  const showProb = opts.showProb !== false;
  const showWave = opts.showWave === true;
  const showEnergy = opts.showEnergy !== false;
  const colors = opts.colors ?? ['#2563eb', '#059669', '#dc2626', '#7c3aed', '#f59e0b', '#0ea5e9'];

  // helpers
  const line = (x1: number, y1: number, x2: number, y2: number, color: string, width = 1, dash?: string) => {
    const l = document.createElementNS(svgns, 'line');
    l.setAttribute('x1', String(x1)); l.setAttribute('y1', String(y1));
    l.setAttribute('x2', String(x2)); l.setAttribute('y2', String(y2));
    l.setAttribute('stroke', color); l.setAttribute('stroke-width', String(width));
    if (dash) l.setAttribute('stroke-dasharray', dash);
    svg.appendChild(l);
  };
  const path = (d: string, stroke?: string, width?: number, fill?: string, opacity?: number, dash?: string) => {
    const p = document.createElementNS(svgns, 'path'); p.setAttribute('d', d);
    p.setAttribute('fill', fill ?? 'none');
    if (stroke) p.setAttribute('stroke', stroke);
    if (width) p.setAttribute('stroke-width', String(width));
    if (opacity != null) p.setAttribute('opacity', String(opacity));
    if (dash) p.setAttribute('stroke-dasharray', dash);
    svg.appendChild(p);
  };
  const text = (x: number, y: number, s: string, size = 12, color = '#111827', anchor: 'start'|'middle'|'end' = 'start') => {
    const t = document.createElementNS(svgns, 'text');
    t.setAttribute('x', String(x)); t.setAttribute('y', String(y));
    t.setAttribute('fill', color); t.setAttribute('font-size', String(size));
    t.setAttribute('font-family', 'Arial, sans-serif');
    t.setAttribute('text-anchor', anchor);
    t.textContent = s; svg.appendChild(t);
  };

  // convert units
  const x_nm = x.map(xi => xi / NM);
  const V_eV = V.map(v => v / E_CHARGE);
  const E_eV = energiesJ.map(e => e / E_CHARGE);

  // compute y-limits
  const Vmin = Math.min(...V_eV), Vmax = Math.max(...V_eV);
  const Emin = Math.min(...E_eV), Emax = Math.max(...E_eV);
  let ymin = Math.min(Vmin, Emin); let ymax = Math.max(Vmax, Emax);
  const pad = Math.max(1e-6, (ymax - ymin) * 0.08); ymin -= pad; ymax += pad;

  // scales
  const margin = {left: 60, right: 20, top: 30, bottom: 36};
  const xToPx = (xv: number) => {
    const x0 = x_nm[0], x1 = x_nm[x_nm.length - 1];
    return (xv - x0) * (W - margin.left - margin.right) / (x1 - x0) + margin.left;
  };
  const yToPx = (ev: number) => H - margin.bottom - (ev - ymin) * (H - margin.top - margin.bottom) / (ymax - ymin);

  // axes
  line(margin.left, margin.top, margin.left, H - margin.bottom, '#9ca3af', 1);
  line(margin.left, H - margin.bottom, W - margin.right, H - margin.bottom, '#9ca3af', 1);
  text(W - 60, H - 10, 'x (nm)'); text(margin.left, 18, title, 14);
  text(18, 30, 'Energy / Potential (eV)');

  // potential
  if (showPotential) {
    let d = '';
    for (let i = 0; i < x_nm.length; i++) {
      const px = xToPx(x_nm[i]); const py = yToPx(V_eV[i]);
      d += (i === 0 ? `M ${px} ${py}` : ` L ${px} ${py}`);
    }
    path(d, '#111827', 2);
  }

  // scale factors
  let maxProb = 0, maxAmp = 0;
  for (let j = 0; j < psi[0].length; j++) {
    for (let i = 0; i < x.length; i++) {
      const v = psi[i][j]; maxProb = Math.max(maxProb, v*v); maxAmp = Math.max(maxAmp, Math.abs(v));
    }
  }
  const probScale = (ymax - ymin) * 0.18 / (maxProb > 0 ? maxProb : 1);
  const waveScale = (ymax - ymin) * 0.06 / (maxAmp > 0 ? maxAmp : 1);

  for (let j = 0; j < psi[0].length; j++) {
    const color = colors[j % colors.length]; const Ej = E_eV[j];
    if (showProb) {
      let d = '';
      for (let i = 0; i < x.length; i++) {
        const prob = psi[i][j] * psi[i][j] * probScale;
        const px = xToPx(x_nm[i]); const py = yToPx(Ej + prob);
        d += (i === 0 ? `M ${px} ${py}` : ` L ${px} ${py}`);
      }
      path(d, color, 2.5);
      d += ` L ${xToPx(x_nm[x_nm.length-1])} ${yToPx(Ej)} L ${xToPx(x_nm[0])} ${yToPx(Ej)} Z`;
      path(d, 'none', 0, color, 0.10);
    }
    if (showWave) {
      let d = '';
      for (let i = 0; i < x.length; i++) {
        const px = xToPx(x_nm[i]); const py = yToPx(Ej + psi[i][j] * waveScale);
        d += (i === 0 ? `M ${px} ${py}` : ` L ${px} ${py}`);
      }
      path(d, color, 1.6);
    }
    if (showEnergy) {
      const y = yToPx(Ej); line(xToPx(x_nm[0]), y, xToPx(x_nm[x_nm.length-1]), y, color, 1, '5,3');
    }
  }
}

export function renderProbabilityDensitySVG(
  container: string | HTMLElement,
  x: number[], energiesJ: number[], psi: number[][], title: string
) {
  const Vzero = new Array(x.length).fill(0);
  renderPotentialAndStatesSVG(container, x, Vzero, energiesJ, psi, title, {
    showPotential: false, showEnergy: true, showProb: true, showWave: false,
  });
}

// Expose as global for non-module pages
declare global {
  interface Window { QMPlot?: { renderPotentialAndStatesSVG: typeof renderPotentialAndStatesSVG; renderProbabilityDensitySVG: typeof renderProbabilityDensitySVG; }; }
}
if (typeof window !== 'undefined') {
  window.QMPlot = { renderPotentialAndStatesSVG, renderProbabilityDensitySVG };
}


