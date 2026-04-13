#!/usr/bin/env node
/**
 * marpit_render.js
 * ================
 * Standalone Node.js script called by slide_generator.py.
 *
 * Usage:
 *   node marpit_render.js <markdown_file> [theme_file]
 *
 * Reads Markdown from <markdown_file>, renders it with Marpit using the
 * OASIS theme (inline or from <theme_file>), then writes a complete
 * self-contained HTML document to stdout.
 *
 * Exit codes:
 *   0 — success
 *   1 — error (message on stderr)
 */

'use strict';

const fs   = require('fs');
const path = require('path');
const { Marpit } = require('@marp-team/marpit');

// ── CLI args ──────────────────────────────────────────────────────────────────
const [,, mdFile, themeFile] = process.argv;
if (!mdFile) {
  process.stderr.write('Usage: node marpit_render.js <markdown_file> [theme_file]\n');
  process.exit(1);
}

// ── Read inputs ───────────────────────────────────────────────────────────────
let markdown;
try {
  markdown = fs.readFileSync(mdFile, 'utf8');
} catch (e) {
  process.stderr.write(`Cannot read markdown file: ${e.message}\n`);
  process.exit(1);
}

// ── OASIS theme CSS ───────────────────────────────────────────────────────────
const OASIS_THEME = `
/* @theme oasis */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root {
  --color-bg:     #E8E0D5;
  --color-teal:   #7DD9D0;
  --color-navy:   #1A2280;
  --color-text:   #2B2B2B;
  --color-muted:  #7A7A9A;
  --color-white:  #FFFFFF;
  --color-card:   #FAFAC8;
  --color-panel:  #F0ECE6;
  --font:         'Inter', 'Segoe UI', Arial, sans-serif;
}

/* ── Base slide ─────────────────────────────────────────────────────────────── */
section {
  width: 960px;
  height: 540px;
  background: var(--color-bg);
  font-family: var(--font);
  color: var(--color-text);
  padding: 40px 56px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  overflow: hidden;
  box-sizing: border-box;
  font-size: 16px;
  line-height: 1.55;
}

/* ── Typography ─────────────────────────────────────────────────────────────── */
h1 {
  font-size: 40px;
  font-weight: 800;
  color: var(--color-navy);
  line-height: 1.15;
  margin: 0 0 12px;
}
h2 {
  font-size: 26px;
  font-weight: 700;
  color: var(--color-navy);
  margin: 0 0 10px;
}
h3 {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-navy);
  margin: 10px 0 4px;
}
p {
  font-size: 14px;
  margin: 0 0 8px;
}
strong { font-weight: 700; }
em { font-style: italic; color: var(--color-muted); }

ul, ol {
  font-size: 14px;
  padding-left: 20px;
  margin: 0 0 8px;
}
li { margin-bottom: 4px; }

/* ── Slide variants (via _class directive) ──────────────────────────────────── */

/* Cover */
section.cover {
  background: var(--color-bg);
  justify-content: space-between;
}
section.cover h1 {
  font-size: 52px;
  font-weight: 800;
  color: var(--color-text);
  max-width: 600px;
}

/* Section divider — teal */
section.divider {
  background: var(--color-teal);
  justify-content: center;
  gap: 8px;
}
section.divider h1 {
  font-size: 56px;
  color: var(--color-navy);
}
section.divider h2 {
  font-size: 20px;
  color: var(--color-muted);
  font-weight: 600;
}
section.divider .sec-number {
  font-size: 100px;
  font-weight: 800;
  color: rgba(255,255,255,0.55);
  line-height: 1;
  margin-bottom: 4px;
}

/* Section divider — linen (light) */
section.divider-light {
  background: var(--color-bg);
  justify-content: center;
  gap: 8px;
}
section.divider-light h1 { font-size: 56px; color: var(--color-navy); }
section.divider-light h2 { font-size: 20px; color: var(--color-muted); font-weight: 600; }

/* Issue slide — 40/60 split via CSS grid on the section */
section.issue {
  display: grid;
  grid-template-columns: 40% 60%;
  grid-template-rows: 1fr;
  padding: 0;
  gap: 0;
}
section.issue .left {
  padding: 32px 20px 32px 48px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 6px;
  overflow: hidden;
}
section.issue .right {
  display: flex;
  align-items: stretch;
  gap: 12px;
  padding: 20px 20px 40px 8px;
  min-height: 0;
}

/* Strength slide — 50/50 split */
section.strength {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr;
  padding: 0;
  gap: 0;
}
section.strength .left {
  padding: 48px 20px 48px 48px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 12px;
  overflow: hidden;
}
section.strength .right {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

/* Image / iframe panel */
.panel {
  border-radius: 8px;
  overflow: hidden;
  background: var(--color-white);
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.panel img,
.panel iframe {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: top;
  border: none;
  display: block;
  flex: 1;
}
.panel-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-navy);
  text-align: center;
  padding: 4px 0 0;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  flex-shrink: 0;
}

/* Ghost text (large watermark-style heading) */
.ghost {
  font-size: 72px;
  font-weight: 800;
  color: rgba(255,255,255,0.55);
  line-height: 1;
  margin-bottom: 12px;
}

/* Persona quote callout */
.quote {
  background: rgba(125,217,208,0.18);
  border-left: 3px solid var(--color-teal);
  padding: 5px 8px;
  margin-top: 5px;
  font-size: 11.5px;
  font-style: italic;
  color: #444;
  border-radius: 0 4px 4px 0;
  overflow: hidden;
}
.quote strong { font-weight: 700; font-style: normal; color: var(--color-navy); }

/* Logo card (cover / back cover) */
.logo-card {
  background: var(--color-card);
  padding: 16px 28px;
  border-radius: 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.logo-card img { max-height: 70px; max-width: 260px; object-fit: contain; }
.logo-card .brand-text { font-size: 28px; font-weight: 800; color: var(--color-text); }

/* TOC */
section.toc {
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: 32px;
  padding: 48px 56px;
  align-items: start;
}
section.toc h1 {
  font-size: 38px;
  color: var(--color-navy);
  line-height: 1.2;
}
.toc-list { display: flex; flex-direction: column; }
.toc-item {
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding: 12px 0;
  border-bottom: 1px dashed #B0A89A;
  font-size: 18px;
  font-weight: 600;
}
.toc-item:last-child { border-bottom: none; }
.toc-num { font-size: 13px; font-weight: 700; color: var(--color-navy); min-width: 28px; }

/* Back cover */
section.back {
  justify-content: space-between;
}

/* Print */
@media print {
  * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
}
`;

// ── Instantiate Marpit ────────────────────────────────────────────────────────
const marpit = new Marpit({
  markdown: {
    html: true,   // allow raw HTML in Markdown (needed for panels, iframes, etc.)
    breaks: false,
  },
  // Use inline SVG slide wrapping for precise 960×540 layout
  inlineSVG: false,
});

// Load theme
let themeCss = OASIS_THEME;
if (themeFile) {
  try {
    themeCss = fs.readFileSync(themeFile, 'utf8');
  } catch (e) {
    process.stderr.write(`Warning: cannot read theme file ${themeFile}: ${e.message}\n`);
  }
}

marpit.themeSet.default = marpit.themeSet.add(themeCss);

// ── Render ────────────────────────────────────────────────────────────────────
let html, css;
try {
  ({ html, css } = marpit.render(markdown));
} catch (e) {
  process.stderr.write(`Marpit render error: ${e.message}\n${e.stack}\n`);
  process.exit(1);
}

// ── Assemble full HTML document ───────────────────────────────────────────────
const title = (markdown.match(/^#\s+(.+)/m) || [])[1] || 'Usability Test Report';

const fullHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #999; font-family: 'Inter', sans-serif; }
    .marpit { display: block; }
    .marpit > svg,
    .marpit > section {
      display: block;
      width: 960px;
      height: 540px;
      overflow: hidden;
      margin: 0 auto 24px;
      page-break-after: always;
      break-after: page;
    }
    @media print {
      * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
      body { background: white; }
      .marpit > svg,
      .marpit > section { margin: 0; }
      @page { size: 960px 540px landscape; margin: 0; }
    }
    ${css}
  </style>
</head>
<body>
${html}
</body>
</html>`;

process.stdout.write(fullHtml);
