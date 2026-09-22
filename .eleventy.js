const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
const markdownItAnchor = require("markdown-it-anchor");
const path = require("path");

const repositoryUrl = "https://github.com/christopherseaman/datasci_217";
const excludedContent = /^(?:\d{2}\/assignment\/|\d{2}\/demo\/)/;

function markdownHeadingSlug(value) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}\s_-]/gu, "")
    .replace(/\s+/g, "-");
}

function renderedSourceLink(url, sourcePath) {
  const prefix = (process.env.ELEVENTY_PATH_PREFIX || "/").replace(/\/$/, "");
  if (/^(?:[a-z]+:|\/\/|#)/i.test(url)) return url;
  if (url.startsWith("/")) {
    if (!prefix || url === prefix || url.startsWith(`${prefix}/`)) return url;
    return `${prefix}${url}`;
  }
  const match = url.match(/^([^?#]+?)([?#].*)?$/);
  if (!match) return url;

  const target = path.normalize(path.join(path.dirname(sourcePath), match[1]));
  const repositoryPath = path.relative(process.cwd(), target).split(path.sep).join("/");
  const suffix = match[2] || "";
  if (/^\d{2}\/assignment\/media\//.test(repositoryPath)) {
    return `${prefix}/${repositoryPath}${suffix}`;
  }
  if (excludedContent.test(repositoryPath)) {
    return `${repositoryUrl}/blob/main/${repositoryPath}${suffix}`;
  }
  const coursePages = { "index.md": "", "references.md": "references/", "shell_workout.md": "shell-workout/", "wsl_troubleshooting.md": "wsl-troubleshooting/", "02/LECTURE_01_CATCHUP.md": "02/lecture-01-catchup/" };
  let pagePath = coursePages[repositoryPath] ?? repositoryPath.replace(/\/README\.md$/i, "/");
  pagePath = pagePath.replace(/\/BONUS\.md$/i, "/bonus/");
  const outputPath = pagePath === repositoryPath ? repositoryPath : pagePath;
  return `${prefix}/${outputPath.replace(/^\//, "")}${suffix}`;
}

module.exports = function (eleventyConfig) {
  eleventyConfig.addPlugin(syntaxHighlight);
  eleventyConfig.amendLibrary("md", (markdownLibrary) => {
    // Match the fragments produced by GitHub-flavored Markdown source links.
    markdownLibrary.use(markdownItAnchor, { slugify: markdownHeadingSlug });
  });

  // Passthrough copy — media folders and CSS
  eleventyConfig.addPassthroughCopy("css");
  eleventyConfig.addPassthroughCopy("media/wsl-troubleshooting");
  eleventyConfig.addPassthroughCopy("*/media/**");
  eleventyConfig.addPassthroughCopy("*/assignment/media/**");

  // Keep source Markdown URLs GitHub-friendly while routing links and images
  // to generated pages, repository assets, or main-branch activity blobs.
  eleventyConfig.addTransform("source-markdown-links", function (content, outputPath) {
    if (!outputPath || !outputPath.endsWith(".html")) return content;
    const sourcePath = this.page?.inputPath;
    if (!sourcePath) return content;
    return content.replace(/((?:href|src)=["'])([^"']+)(["'])/gi, (_, prefix, url, quote) => {
      return `${prefix}${renderedSourceLink(url, sourcePath)}${quote}`;
    });
  });

  // Notion turns an image's link text into the picture's caption; render the
  // same text as a figcaption so a standalone image reads the same on the site.
  eleventyConfig.addTransform("image-captions", function (content, outputPath) {
    if (!outputPath || !outputPath.endsWith(".html")) return content;
    return content.replace(/<p>(<img\b[^>]*>)<\/p>/gi, (paragraph, image) => {
      const alt = image.match(/\balt=(["'])(.*?)\1/i);
      if (!alt || !alt[2].trim()) return paragraph;
      return `<figure>${image}<figcaption>${alt[2]}</figcaption></figure>`;
    });
  });

  // Computed data — assign layout and clean URLs without frontmatter
  eleventyConfig.addGlobalData("eleventyComputed", {
    layout: (data) => data.layout || "layout.njk",
    title: (data) => {
      if (data.title) return data.title;
      const inputPath = data.page?.inputPath || "";
      const match = inputPath.match(/\/(\d{2})\//);
      if (!match) return undefined;
      const nav = require("./_data/nav.js");
      const lecture = nav.lectures.find((l) => l.id === match[1]);
      return lecture ? `${match[1]}: ${lecture.label}` : undefined;
    },
    lectureId: (data) => {
      const match = (data.page?.inputPath || "").match(/\/(\d{2})\//);
      return match ? match[1] : undefined;
    },
    isBonus: (data) => {
      const p = data.page?.inputPath || "";
      return /\/\d{2}\/BONUS\.md$/.test(p) || /\/\d{2}\/bonus\//.test(p);
    },
    permalink: (data) => {
      if (data.permalink) return data.permalink;
      const inputPath = data.page?.inputPath || "";
      const lectureMatch = inputPath.match(/\/(\d{2})\/README\.md$/);
      if (lectureMatch) return `/${lectureMatch[1]}/`;
      const bonusMatch = inputPath.match(/\/(\d{2})\/BONUS\.md$/);
      if (bonusMatch) return `/${bonusMatch[1]}/bonus/`;
      return undefined;
    },
  });

  // Lecture collection for index page listing
  eleventyConfig.addCollection("lectures", (collectionApi) => {
    return collectionApi
      .getFilteredByGlob("*/README.md")
      .filter((item) => /\/\d{2}\/README\.md$/.test(item.inputPath))
      .sort((a, b) => {
        const numA = a.inputPath.match(/\/(\d{2})\//)?.[1] || "0";
        const numB = b.inputPath.match(/\/(\d{2})\//)?.[1] || "0";
        return numA.localeCompare(numB);
      });
  });

  eleventyConfig.addWatchTarget("css/");

  return {
    dir: { input: ".", output: "_site", includes: "_includes", data: "_data" },
    markdownTemplateEngine: "njk",
    pathPrefix: process.env.ELEVENTY_PATH_PREFIX || "/",
  };
};
