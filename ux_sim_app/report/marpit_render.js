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
  --color-text:   #000000;
  --color-muted:  #444444;
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
  padding: 32px 48px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  overflow: hidden;
  box-sizing: border-box;
  font-size: 13px;
  line-height: 1.5;
}

/* ── Typography ─────────────────────────────────────────────────────────────── */
h1 {
  font-size: 32px;
  font-weight: 800;
  color: var(--color-navy);
  line-height: 1.15;
  margin: 0 0 8px;
}
h2 {
  font-size: 20px;
  font-weight: 700;
  color: var(--color-navy);
  margin: 0 0 8px;
}
h3 {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--color-navy);
  margin: 8px 0 3px;
}
p {
  font-size: 13px;
  margin: 0 0 6px;
  color: #000;
}
strong { font-weight: 700; color: #000; }
em { font-style: italic; color: var(--color-muted); }

ul, ol {
  font-size: 13px;
  padding-left: 18px;
  margin: 0 0 6px;
}
li { margin-bottom: 3px; color: #000; }

/* ── Slide variants (via _class directive) ──────────────────────────────────── */

/* Cover */
section.cover {
  background: var(--color-bg);
  justify-content: space-between;
}
section.cover h1 {
  font-size: 42px;
  font-weight: 800;
  color: #000;
  max-width: 580px;
}

/* Section divider — teal */
section.divider {
  background: var(--color-teal);
  justify-content: center;
  gap: 8px;
}
section.divider h1 {
  font-size: 44px;
  color: var(--color-navy);
}
section.divider h2 {
  font-size: 17px;
  color: #333;
  font-weight: 600;
}
section.divider .sec-number {
  font-size: 80px;
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
section.divider-light h1 { font-size: 44px; color: var(--color-navy); }
section.divider-light h2 { font-size: 17px; color: #333; font-weight: 600; }

/* Issue slide — 40/60 split via CSS grid on the section */
section.issue {
  display: grid;
  grid-template-columns: 40% 60%;
  grid-template-rows: 1fr;
  padding: 0;
  gap: 0;
}
section.issue .left {
  padding: 24px 16px 24px 40px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
  overflow: hidden;
}
section.issue .right {
  display: flex;
  align-items: stretch;
  gap: 10px;
  padding: 16px 16px 32px 6px;
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
  padding: 36px 16px 36px 40px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 10px;
  overflow: hidden;
}
section.strength .right {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
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
/* On teal dividers: white ghost; on linen slides: use a dark-tinted ghost */
.ghost {
  font-size: 56px;
  font-weight: 800;
  color: rgba(26,34,128,0.12);
  line-height: 1;
  margin-bottom: 8px;
}
section.divider .ghost,
section.divider .sec-number { color: rgba(255,255,255,0.45); }

/* Persona quote callout */
.quote {
  background: rgba(125,217,208,0.18);
  border-left: 3px solid var(--color-teal);
  padding: 4px 7px;
  margin-top: 4px;
  font-size: 10.5px;
  font-style: italic;
  color: #000;
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
.logo-card .brand-text { font-size: 24px; font-weight: 800; color: #000; }

/* TOC */
section.toc {
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: 32px;
  padding: 48px 56px;
  align-items: start;
}
section.toc h1 {
  font-size: 30px;
  color: var(--color-navy);
  line-height: 1.2;
}
.toc-list { display: flex; flex-direction: column; }
.toc-item {
  display: flex;
  align-items: baseline;
  gap: 14px;
  padding: 9px 0;
  border-bottom: 1px dashed #B0A89A;
  font-size: 15px;
  font-weight: 600;
  color: #000;
}
.toc-item:last-child { border-bottom: none; }
.toc-num { font-size: 11px; font-weight: 700; color: var(--color-navy); min-width: 24px; }

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
