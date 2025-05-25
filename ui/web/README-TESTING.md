# Testing Setup for React Router 7 in NestJS

## Overview

This project includes test setup for the React Router 7 integration with NestJS. The tests are designed to validate:

1. The custom server integration between React Router 7 and NestJS
2. Route function behavior (loaders, actions, meta functions)
3. Server-side rendering capabilities

## Testing Structure

The testing is organized as follows:

- `src/react-router.spec.ts` - Tests for the NestJS-React Router integration
- `test/router-test-utils.tsx` - Utilities for testing React Router components
- `test/setup.ts` - Global test setup for Vitest
- `docs/TESTING.md` - Detailed documentation on testing strategies

## Working Tests

- ✅ `src/react-router.spec.ts` - Successfully tests the NestJS integration with React Router
- ✅ Unit tests for NestJS services and controllers

## Current Limitations

There are some limitations when testing React Router 7 components:

1. **Client/Server Boundaries**: React Router 7 enforces strict separation between server and client code. This makes it difficult to directly test certain files like `entry.server.tsx` or route components.

2. **Vite Plugin Restrictions**: The React Router 7 Vite plugin enforces these boundaries during testing, resulting in errors like "React Router Vite plugin can't detect preamble" when attempting to import route components.

3. **Testing Route Components**: Due to the limitations above, testing route components directly can be challenging. Instead, we recommend:
   - Testing individual exports (meta, loader, action) in isolation
   - Using mock data to test the rendering outside the context of React Router

## Recommended Testing Approach

1. **Unit Test Functions**: Test loader, action, and meta functions directly with mocked context.

2. **Test NestJS Integration**: Test how NestJS mounts the React Router handler.

3. **Component Testing**: Test UI components with mocked props rather than within the React Router context.

## Example Test Structure

```
// For testing loader functions
import { loader } from './route';
import { createMockLoaderArgs } from '../../test/router-test-utils';

describe('Route Loader', () => {
  it('returns expected data', async () => {
    const args = createMockLoaderArgs();
    const result = await loader(args);
    expect(result).toEqual(/*...*/);
  });
});
```

## Future Improvements

We can improve the testing setup by:

1. Configuring a special test environment that bypasses the Vite plugin restrictions
2. Creating more comprehensive mocks for the React Router context
3. Investigating ways to unit test server-side rendering behavior

## Running Tests

```bash
# Run all tests
pnpm test

# Run tests in watch mode
pnpm test:watch

# Run tests with coverage
pnpm test:cov
```

For more detailed information about testing approaches, see the [TESTING.md](./docs/TESTING.md) documentation.
