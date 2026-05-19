#!/usr/bin/env node
/**
 * marpit_smoke_test.js
 * ====================
 * Quick smoke test for the Marpit render pipeline.
 * Runs without any external dependencies beyond @marp-team/marpit.
 *
 * Usage:
 *   node ux_sim_app/report/marpit_smoke_test.js
 *
 * Exit code 0 = all assertions passed.
 * Exit code 1 = one or more assertions failed.
 */

"use strict";

const path = require("path");
const { execSync } = require("child_process");
const fs = require("fs");
const os = require("os");

let failures = 0;

function assert(condition, message) {
  if (!condition) {
    console.error(`  ✗ FAIL: ${message}`);
    failures++;
  } else {
    console.log(`  ✓ PASS: ${message}`);
  }
}

// ── 1. Marpit module loads ────────────────────────────────────────────────────
console.log("\n[1] Marpit module loads");
let Marpit;
try {
  ({ Marpit } = require("@marp-team/marpit"));
  assert(typeof Marpit === "function", "@marp-team/marpit exports Marpit class");
} catch (e) {
  console.error("  ✗ FAIL: could not require @marp-team/marpit:", e.message);
  failures++;
  process.exit(1); // can't continue without Marpit
}

// ── 2. marpit_render.js script exists ─────────────────────────────────────────
console.log("\n[2] marpit_render.js exists");
const renderScript = path.join(__dirname, "marpit_render.js");
assert(fs.existsSync(renderScript), "marpit_render.js is present on disk");

// ── 3. Render a minimal slide deck ────────────────────────────────────────────
console.log("\n[3] Render minimal slide deck");
const sampleMd = `---
<!-- _class: cover -->

# Test Cover

---
<!-- _class: issue -->

## Issue Slide

Some body text.

---
<!-- _class: back -->

# Back Cover
`;

const tmpMd = path.join(os.tmpdir(), `marpit_smoke_${Date.now()}.md`);
fs.writeFileSync(tmpMd, sampleMd, "utf8");

let html = "";
try {
  html = execSync(
    `node "${renderScript}" "${tmpMd}"`,
    { encoding: "utf8", timeout: 15000 }
  );
  assert(html.length > 500, "Rendered HTML is non-trivial (> 500 chars)");
} catch (e) {
  console.error("  ✗ FAIL: marpit_render.js threw:", e.message);
  failures++;
}
fs.unlinkSync(tmpMd);

// ── 4. HTML structure checks ───────────────────────────────────────────────────
console.log("\n[4] HTML structure checks");
assert(html.includes("<!DOCTYPE html>"), "Output starts with DOCTYPE");
assert(html.includes("<section"), "Output contains <section> slide elements");
assert(html.includes("Test Cover"), "Cover slide content is present");
assert(html.includes("Issue Slide"), "Issue slide content is present");
assert(html.includes("Back Cover"), "Back cover content is present");

// ── 5. OASIS theme CSS tokens are present ─────────────────────────────────────
console.log("\n[5] OASIS theme CSS tokens");
assert(html.includes("--fs-hero"), "CSS token --fs-hero is present");
assert(html.includes("--fs-body"), "CSS token --fs-body is present");
assert(html.includes("--color-teal"), "CSS token --color-teal is present");
assert(html.includes("--color-bg"), "CSS token --color-bg (linen background) is present");

// ── 6. print-color-adjust fix is present ──────────────────────────────────────
console.log("\n[6] Print colour-adjust fix");
assert(
  html.includes("print-color-adjust") || html.includes("-webkit-print-color-adjust"),
  "print-color-adjust rule is injected for PDF export"
);

// ── Summary ───────────────────────────────────────────────────────────────────
console.log(`\n${"─".repeat(50)}`);
if (failures === 0) {
  console.log(`✅  All smoke tests passed.\n`);
  process.exit(0);
} else {
  console.error(`❌  ${failures} smoke test(s) failed.\n`);
  process.exit(1);
}
