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
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,600;0,700;0,800;1,400&display=swap');

/*
  OASIS Type Scale  (960x540 canvas)
  ====================================
  --fs-hero     : cover / divider h1         42px
  --fs-title    : content slide h1           26px
  --fs-subtitle : h2 subheadings             16px
  --fs-label    : h3 eyebrow / ALL-CAPS tag   9px
  --fs-body     : p, li, default             12px
  --fs-small    : quotes, panel labels       10px
  --fs-ghost    : decorative watermark       52px
  --fs-toc-h    : TOC heading                28px
  --fs-toc-i    : TOC item                   13px
  --fs-brand    : logo-card brand name       22px
*/
:root {
  /* Palette */
  --color-bg:     #E8E0D5;
  --color-teal:   #7DD9D0;
  --color-navy:   #1A2280;
  --color-text:   #000000;
  --color-muted:  #333333;
  --color-white:  #FFFFFF;
  --color-card:   #FAFAC8;
  --color-panel:  #F0ECE6;

  /* Font */
  --font: 'Inter', 'Segoe UI', Arial, sans-serif;

  /* Type scale */
  --fs-hero:     42px;
  --fs-title:    26px;
  --fs-subtitle: 16px;
  --fs-label:     9px;
  --fs-body:     12px;
  --fs-small:    10px;
  --fs-ghost:    52px;
  --fs-toc-h:    28px;
  --fs-toc-i:    13px;
  --fs-brand:    22px;

  /* Line heights */
  --lh-tight:  1.15;
  --lh-normal: 1.45;
  --lh-loose:  1.6;

  /* Spacing */
  --sp-xs:  4px;
  --sp-sm:  8px;
  --sp-md: 16px;
  --sp-lg: 28px;
  --sp-xl: 44px;
}

/* =========================================================
   BASE SLIDE
   ========================================================= */
section {
  width: 960px;
  height: 540px;
  background: var(--color-bg);
  font-family: var(--font);
  font-size: var(--fs-body);
  line-height: var(--lh-normal);
  color: var(--color-text);
  padding: 28px 44px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  overflow: hidden;
  box-sizing: border-box;
}

/* =========================================================
   GLOBAL TYPOGRAPHY
   Every element has an explicit rule so Marpit's CSS
   scoping cannot silently override them.
   ========================================================= */
h1 {
  font-size: var(--fs-title);
  font-weight: 800;
  line-height: var(--lh-tight);
  color: var(--color-navy);
  margin: 0 0 var(--sp-sm);
  letter-spacing: -0.01em;
}
h2 {
  font-size: var(--fs-subtitle);
  font-weight: 700;
  line-height: var(--lh-tight);
  color: var(--color-navy);
  margin: 0 0 var(--sp-sm);
}
h3 {
  font-size: var(--fs-label);
  font-weight: 700;
  line-height: 1.2;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--color-navy);
  margin: var(--sp-sm) 0 var(--sp-xs);
}
p {
  font-size: var(--fs-body);
  line-height: var(--lh-normal);
  color: var(--color-text);
  margin: 0 0 var(--sp-xs);
}
strong {
  font-weight: 700;
  color: var(--color-text);
}
em {
  font-style: italic;
  color: var(--color-muted);
}
ul, ol {
  font-size: var(--fs-body);
  line-height: var(--lh-normal);
  color: var(--color-text);
  padding-left: 16px;
  margin: 0 0 var(--sp-xs);
}
li {
  margin-bottom: 2px;
  color: var(--color-text);
}

/* =========================================================
   COVER SLIDE
   ========================================================= */
section.cover {
  background: var(--color-bg);
  justify-content: space-between;
  padding: 36px 52px;
}
section.cover h1 {
  font-size: var(--fs-hero);
  font-weight: 800;
  line-height: var(--lh-tight);
  color: var(--color-text);
  max-width: 560px;
  letter-spacing: -0.02em;
}
section.cover h2 {
  font-size: var(--fs-subtitle);
  font-weight: 400;
  color: var(--color-muted);
  margin: var(--sp-sm) 0 0;
}

/* =========================================================
   SECTION DIVIDER — teal
   ========================================================= */
section.divider {
  background: var(--color-teal);
  justify-content: center;
  align-items: flex-start;
  gap: var(--sp-sm);
  padding: 36px 52px;
}
section.divider h1 {
  font-size: var(--fs-hero);
  font-weight: 800;
  color: var(--color-navy);
  line-height: var(--lh-tight);
  letter-spacing: -0.02em;
}
section.divider h2 {
  font-size: var(--fs-subtitle);
  font-weight: 600;
  color: #1a1a1a;
  line-height: var(--lh-normal);
}
section.divider .sec-number {
  font-size: var(--fs-ghost);
  font-weight: 800;
  color: rgba(255,255,255,0.40);
  line-height: 1;
  margin-bottom: var(--sp-xs);
}

/* =========================================================
   SECTION DIVIDER — linen (light)
   ========================================================= */
section.divider-light {
  background: var(--color-bg);
  justify-content: center;
  align-items: flex-start;
  gap: var(--sp-sm);
  padding: 36px 52px;
}
section.divider-light h1 {
  font-size: var(--fs-hero);
  font-weight: 800;
  color: var(--color-navy);
  line-height: var(--lh-tight);
  letter-spacing: -0.02em;
}
section.divider-light h2 {
  font-size: var(--fs-subtitle);
  font-weight: 600;
  color: var(--color-muted);
}

/* =========================================================
   ISSUE SLIDE — 40 / 60 grid
   ========================================================= */
section.issue {
  display: grid;
  grid-template-columns: 40% 60%;
  grid-template-rows: 1fr;
  padding: 0;
  gap: 0;
}
section.issue .left {
  padding: 24px 14px 24px 36px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: var(--sp-xs);
  overflow: hidden;
}
section.issue .left h1 {
  font-size: 18px;
  font-weight: 800;
  line-height: 1.2;
  color: var(--color-navy);
  margin: 0 0 var(--sp-xs);
}
section.issue .left h2 {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-muted);
  margin: 0 0 var(--sp-xs);
}
section.issue .left p,
section.issue .left li {
  font-size: 11px;
  line-height: 1.45;
  color: var(--color-text);
}
section.issue .right {
  display: flex;
  align-items: stretch;
  gap: 8px;
  padding: 14px 14px 28px 6px;
  min-height: 0;
}

/* =========================================================
   STRENGTH SLIDE — 50 / 50 grid
   ========================================================= */
section.strength {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr;
  padding: 0;
  gap: 0;
}
section.strength .left {
  padding: 32px 14px 32px 36px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: var(--sp-sm);
  overflow: hidden;
}
section.strength .left h1 {
  font-size: 20px;
  font-weight: 800;
  line-height: 1.2;
  color: var(--color-navy);
}
section.strength .left h2 {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-muted);
}
section.strength .left p,
section.strength .left li {
  font-size: 11px;
  line-height: 1.45;
  color: var(--color-text);
}
section.strength .right {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 14px;
}

/* =========================================================
   IMAGE / IFRAME PANEL
   ========================================================= */
.panel {
  border-radius: 8px;
  overflow: hidden;
  background: var(--color-white);
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  box-shadow: 0 1px 4px rgba(0,0,0,0.10);
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
  font-size: var(--fs-small);
  font-weight: 700;
  color: var(--color-navy);
  text-align: center;
  padding: 3px 0 0;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  flex-shrink: 0;
  background: var(--color-panel);
}

/* =========================================================
   GHOST / WATERMARK TEXT
   ========================================================= */
.ghost {
  font-size: var(--fs-ghost);
  font-weight: 800;
  color: rgba(26,34,128,0.10);
  line-height: 1;
  margin-bottom: var(--sp-sm);
  letter-spacing: -0.02em;
}
section.divider .ghost,
section.divider .sec-number {
  color: rgba(255,255,255,0.38);
}

/* =========================================================
   PERSONA QUOTE CALLOUT
   ========================================================= */
.quote {
  background: rgba(125,217,208,0.15);
  border-left: 3px solid var(--color-teal);
  padding: 4px 8px;
  margin-top: var(--sp-xs);
  font-size: var(--fs-small);
  font-style: italic;
  line-height: var(--lh-loose);
  color: var(--color-text);
  border-radius: 0 4px 4px 0;
  overflow: hidden;
}
.quote strong {
  font-weight: 700;
  font-style: normal;
  color: var(--color-navy);
}

/* =========================================================
   LOGO CARD (cover / back cover)
   ========================================================= */
.logo-card {
  background: var(--color-card);
  padding: 14px 24px;
  border-radius: 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.logo-card img { max-height: 64px; max-width: 240px; object-fit: contain; }
.logo-card .brand-text {
  font-size: var(--fs-brand);
  font-weight: 800;
  color: var(--color-text);
}

/* =========================================================
   TABLE OF CONTENTS
   ========================================================= */
section.toc {
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: 28px;
  padding: 40px 52px;
  align-items: start;
}
section.toc h1 {
  font-size: var(--fs-toc-h);
  font-weight: 800;
  color: var(--color-navy);
  line-height: var(--lh-tight);
  letter-spacing: -0.01em;
}
.toc-list { display: flex; flex-direction: column; }
.toc-item {
  display: flex;
  align-items: baseline;
  gap: 12px;
  padding: 7px 0;
  border-bottom: 1px dashed #B0A89A;
  font-size: var(--fs-toc-i);
  font-weight: 600;
  line-height: var(--lh-normal);
  color: var(--color-text);
}
.toc-item:last-child { border-bottom: none; }
.toc-num {
  font-size: 10px;
  font-weight: 700;
  color: var(--color-navy);
  min-width: 22px;
  flex-shrink: 0;
}

/* =========================================================
   BACK COVER
   ========================================================= */
section.back {
  justify-content: space-between;
  padding: 36px 52px;
}
section.back h1 {
  font-size: var(--fs-title);
  font-weight: 800;
  color: var(--color-navy);
}
section.back p {
  font-size: var(--fs-body);
  color: var(--color-muted);
}

/* =========================================================
   INTRO / METHODOLOGY SLIDE
   ========================================================= */
section.intro h1 {
  font-size: var(--fs-title);
  font-weight: 800;
  color: var(--color-navy);
  margin-bottom: var(--sp-md);
}
section.intro p {
  font-size: var(--fs-body);
  line-height: var(--lh-loose);
  color: var(--color-text);
  max-width: 720px;
}

/* =========================================================
   PRINT — preserve backgrounds in PDF
   ========================================================= */
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
    body { background: #888; font-family: 'Inter', sans-serif; }
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
