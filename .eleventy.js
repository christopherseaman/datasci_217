const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
const markdownIt = require("markdown-it");
const markdownItAnchor = require("markdown-it-anchor");

module.exports = function (eleventyConfig) {
  // Syntax highlighting (Prism)
  eleventyConfig.addPlugin(syntaxHighlight);

  // Markdown config
  const md = markdownIt({ html: true, linkify: true, typographer: true })
    .use(markdownItAnchor, { permalink: false });
  eleventyConfig.setLibrary("md", md);

  // Passthrough copy — media folders and CSS
  eleventyConfig.addPassthroughCopy("css");
  eleventyConfig.addPassthroughCopy("*/media/**");

  // Computed data — assign layouts based on file path, no frontmatter needed
  eleventyConfig.addGlobalData("eleventyComputed", {
    layout: (data) => {
      if (data.layout) return data.layout;
      const inputPath = data.page?.inputPath || "";
      // NN/README.md → lecture layout
      if (/\/\d{2}\/README\.md$/.test(inputPath)) return "lecture.njk";
      // NN/BONUS.md → bonus layout
      if (/\/\d{2}\/BONUS\.md$/.test(inputPath)) return "bonus.njk";
      // NN/bonus/*.md → bonus layout
      if (/\/\d{2}\/bonus\/[^/]+\.md$/.test(inputPath)) return "bonus.njk";
      // Top-level .md files (index, references, etc.) → base layout
      return "base.njk";
    },
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
      const inputPath = data.page?.inputPath || "";
      const match = inputPath.match(/\/(\d{2})\//);
      return match ? match[1] : undefined;
    },
    permalink: (data) => {
      if (data.permalink) return data.permalink;
      const inputPath = data.page?.inputPath || "";
      // NN/README.md → /NN/
      const lectureMatch = inputPath.match(/\/(\d{2})\/README\.md$/);
      if (lectureMatch) return `/${lectureMatch[1]}/`;
      // NN/BONUS.md → /NN/bonus/
      const bonusMatch = inputPath.match(/\/(\d{2})\/BONUS\.md$/);
      if (bonusMatch) return `/${bonusMatch[1]}/bonus/`;
      return undefined;
    },
  });

  // Lecture collection — sorted by folder number
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

  // Bonus collection
  eleventyConfig.addCollection("bonus", (collectionApi) => {
    const bonusFiles = collectionApi
      .getFilteredByGlob(["*/BONUS.md", "*/bonus/*.md"])
      .filter(
        (item) =>
          /\/\d{2}\/BONUS\.md$/.test(item.inputPath) ||
          /\/\d{2}\/bonus\/[^/]+\.md$/.test(item.inputPath)
      );
    return bonusFiles.sort((a, b) =>
      a.inputPath.localeCompare(b.inputPath)
    );
  });

  // Watch targets
  eleventyConfig.addWatchTarget("css/");

  return {
    dir: {
      input: ".",
      output: "_site",
      includes: "_includes",
      data: "_data",
    },
    markdownTemplateEngine: "njk",
    pathPrefix: process.env.ELEVENTY_PATH_PREFIX || "/",
  };
};
