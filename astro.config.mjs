// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://not.badmath.org/ds217',
  integrations: [
    starlight({
      title: 'DataSci-217',
      expressiveCode: {
        styleOverrides: {
          codeBackground: ({ theme }) =>
            theme.type === 'dark' ? '#0d1117' : theme.colors['editor.background'],
        },
      },
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/christopherseaman/datasci_217' }],
      customCss: [
        '@fontsource/fira-mono/400.css',
        '@fontsource/fira-mono/500.css',
        '@fontsource/fira-mono/700.css',
        './src/styles/custom.css',
      ],
      sidebar: [
        {
          label: 'Course',
          items: [
            { slug: 'index', label: 'Overview' },
            { slug: 'references', label: 'References' },
            { slug: 'shell-workout', label: 'Shell Workout' },
          ],
        },
        {
          label: 'Foundational',
          items: [
            { slug: '01', label: '01: Command Line + Python' },
            { slug: '02', label: '02: Python + Git' },
            { slug: '03', label: '03: Data Structures' },
            { slug: '04', label: '04: NumPy' },
            { slug: '05', label: '05: Pandas' },
          ],
        },
        {
          label: 'Extended',
          items: [
            { slug: '06', label: '06: Data Loading' },
            { slug: '07', label: '07: Data Cleaning' },
            { slug: '08', label: '08: Data Wrangling' },
            { slug: '09', label: '09: Visualization' },
            { slug: '10', label: '10: Aggregation' },
            { slug: '11', label: '11: Time Series' },
          ],
        },
        {
          label: 'Resources',
          items: [
            { label: 'GitHub', link: 'https://github.com/christopherseaman/datasci_217' },
            { label: 'The Missing Semester', link: 'https://missing.csail.mit.edu/' },
            { label: 'The Linux Command Line', link: 'http://linuxcommand.org/tlcl.php' },
          ],
        },
      ],
    }),
  ],
});
