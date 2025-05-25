const tailwind = require('eslint-plugin-tailwindcss');

module.exports = [
  {
    ignores: ['**/node_modules/**', '**/dist/**', '**/.turbo/**'],
  },
  {
    ...tailwind.configs['flat/recommended'],
    files: ['app/**/*.{ts,tsx}'],
    languageOptions: {
      globals: {
        // Use languageOptions.globals instead of env
        browser: true,
        es6: true,
        node: true,
      },
      parser: require('@typescript-eslint/parser'),
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        jsx: true,
      },
    },
    plugins: {
      react: require('eslint-plugin-react'),
      'react-hooks': require('eslint-plugin-react-hooks'),
      'jsx-a11y': require('eslint-plugin-jsx-a11y'),
      import: require('eslint-plugin-import'),
    },
    settings: {
      react: {
        version: 'detect',
      },
    },
    rules: {
      'no-unused-vars': 'warn',
      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off',
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'jsx-a11y/accessible-emoji': 'warn',
      'jsx-a11y/alt-text': 'warn',
      'jsx-a11y/anchor-is-valid': 'warn',
      'jsx-a11y/aria-props': 'warn',
      'jsx-a11y/aria-role': 'warn',
      'jsx-a11y/heading-has-content': 'warn',
      'jsx-a11y/iframe-has-title': 'warn',
      'jsx-a11y/no-autofocus': 'warn',
      'jsx-a11y/no-redundant-roles': 'warn',
      'import/order': [
        'warn',
        {
          'newlines-between': 'always-and-inside-groups',
          groups: [
            // built-in types are first
            'builtin',
            // then external modules
            'external',
            // then parent types
            'parent',
            // then siblings
            'sibling',
            // Then the index file
            'index',
            // Then the rest: internal and external type
            'object',
          ],
        },
      ],
      'import/newline-after-import': 'warn',
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: {
      parser: require('@typescript-eslint/parser'),
      parserOptions: {
        tsconfigRootDir: __dirname,
        project: ['./tsconfig.node.json'],
        ecmaVersion: 'latest',
        sourceType: 'module',
        jsx: true,
      },
      globals: {
        // Use languageOptions.globals instead of env
        browser: true,
        es6: true,
        node: true,
      },
    },
    plugins: {
      '@typescript-eslint': require('@typescript-eslint/eslint-plugin'),
      import: require('eslint-plugin-import'),
    },
    rules: {
      // Add any specific rules here
      '@typescript-eslint/no-explicit-any': 'off',
      // 'no-unused-vars': 'warn', // Example ESLint recommended rule
      '@typescript-eslint/no-unused-vars': [
        'warn', // or 'error'
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      // 'no-console': 'warn', // Example ESLint recommended rule
      'import/order': [
        'warn',
        {
          groups: [
            'builtin',
            'external',
            'internal',
            'parent',
            'sibling',
            'index',
          ],
          'newlines-between': 'always',
        },
      ],
      'import/newline-after-import': 'warn',
    },
  },
];
