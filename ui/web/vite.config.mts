import { defineConfig } from 'vite';

import { reactRouter } from '@react-router/dev/vite';
import tailwindcss from '@tailwindcss/vite';
import tsconfigPaths from 'vite-tsconfig-paths';
import { defineConfig as defineVitestConfig } from 'vitest/config';
import path from 'path';

export default defineConfig(() => {
  // TODO: customize Vite configuration as needed
  return defineVitestConfig({
    plugins: [tailwindcss(), reactRouter(), tsconfigPaths()],
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./test/setup.ts'],
      include: ['**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
      coverage: {
        provider: 'v8',
        reporter: ['text', 'json', 'html'],
      },
    },

    resolve: {
      alias: {
        app: path.resolve(__dirname, './app'),
        src: path.resolve(__dirname, './src'),
      },
    },
  });
});
