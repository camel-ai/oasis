# Testing React Router 7 in the NestJS Project

This document provides guidelines for testing React Router 7 components in this NestJS-based application.

## Testing Structure

The testing environment is set up using [Vitest](https://vitest.dev/) and follows these principles:

1. **Unit Tests**: Test individual functions and components in isolation
2. **Integration Tests**: Test the interaction between React Router and the NestJS server
3. **Mocking**: Proper mocking of dependencies to avoid issues with client/server code boundaries

## Test Types

### 1. Unit Tests for Router Functions

Unit tests focus on testing individual functions from React Router components:

- `meta` function tests
- `loader` function tests
- `action` function tests

**Example**:

```tsx
// Importing only the functions to test without rendering components
import { meta, loader } from './my-route';
import { createMockLoaderArgs } from '../../test/router-test-utils';

describe('MyRoute functions', () => {
  it('meta returns correct title', () => {
    const result = meta();
    expect(result[0].title).toBe('Expected Title');
  });
  
  it('loader returns correct data', async () => {
    const args = createMockLoaderArgs();
    const result = await loader(args);
    expect(result).toEqual({ data: 'expected' });
  });
});
```

### 2. Component Tests

Component tests verify the rendering and behavior of React Router components:

```tsx
import { render, screen } from '@testing-library/react';
import MyComponent from './MyComponent';
import { createMockRouterContext } from '../../test/router-test-utils';

describe('MyComponent', () => {
  it('renders correctly with route data', () => {
    // Prepare mock data that would normally come from loader
    const mockData = { hello: 'world' };
    
    // Create props similar to what React Router would provide
    const props = {
      loaderData: mockData,
      params: {},
      matches: [
        // Mock matches array with required structure
        {
          id: 'root',
          pathname: '/',
          params: {},
          data: { user: null },
          handle: {}
        },
        {
          id: 'routes/test',
          pathname: '/',
          params: {},
          data: mockData,
          handle: {}
        }
      ]
    };
    
    render(<MyComponent {...props} />);
    
    expect(screen.getByText('world')).toBeInTheDocument();
  });
});
```

### 3. Integration Tests for NestJS Integration

For testing React Router integration with NestJS, focus on:

1. Mocking the Express adapter
2. Testing mount points and middleware setup
3. Testing route resolution

**Example**:

```tsx
// See src/react-router.spec.ts for a complete example
import { mountReactRouterHandler } from '../src/react-router';

describe('React Router Integration', () => {
  it('should set up React Router handler correctly', async () => {
    const mockApp = {
      getHttpAdapter: () => ({
        getInstance: () => ({ use: vi.fn(), all: vi.fn() })
      })
    };
    
    await mountReactRouterHandler(mockApp);
    
    // Verify the Express routes were set up correctly
    expect(mockApp.getHttpAdapter().getInstance().all)
      .toHaveBeenCalledWith('/{*all}', expect.any(Function));
  });
});
```

## Important Limitations

1. **Client/Server Boundaries**: React Router 7 enforces strict client/server code separation through the Vite plugin. This means:
   - You cannot directly test entry.client.tsx or entry.server.tsx files without proper mocking
   - Server-only code cannot be imported by client-side tests

2. **Import Order**: When mocking modules, all `vi.mock()` calls must come before any imports due to hoisting

3. **Route File Testing**: When testing route files, focus on testing the individual exports (loader, action, meta) rather than the default component export when possible

## Helpful Testing Utilities

We've created several testing utilities to make testing React Router components easier:

- `createMockRouterContext()`: Creates a mock router context object
- `createMockLoaderArgs()`: Creates a mock loader function arguments object with NestJS context
- `testReactRouterSetup()`: Verifies React Router test setup is working correctly

These utilities are available in `test/router-test-utils.tsx`.
