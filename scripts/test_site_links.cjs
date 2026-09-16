// Run after npm run build, using the same ELEVENTY_PATH_PREFIX if set.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const prefix = (process.env.ELEVENTY_PATH_PREFIX || "/").replace(/\/$/, "");
const html = fs.readFileSync("_site/index.html", "utf8");
for (const route of ["references", "shell-workout"]) {
  assert(html.includes(`href="${prefix}/${route}/"`), `Missing ${route} link`);
  assert(fs.existsSync(`_site/${route}/index.html`), `Missing ${route} page`);
}
assert(!/href="[^"]*(?:references|shell_workout)\.md"/.test(html));
const lecture = fs.readFileSync("_site/01/index.html", "utf8");
for (const file of ["github-fork.png", "github-clone-url.png", "vscode-clone.png"]) {
  const route = `/01/assignment/media/${file}`;
  assert(lecture.includes(`src="${prefix}${route}"`), `Missing local screenshot URL: ${file}`);
  assert(fs.existsSync(`_site${route}`), `Missing copied screenshot: ${file}`);
}
console.log("Course page links resolve to generated pages.");
