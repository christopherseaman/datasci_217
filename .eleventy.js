const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");

module.exports = function (eleventyConfig) {
  eleventyConfig.addPlugin(syntaxHighlight);

  // Passthrough copy — media folders and CSS
  eleventyConfig.addPassthroughCopy("css");
  eleventyConfig.addPassthroughCopy("*/media/**");

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
