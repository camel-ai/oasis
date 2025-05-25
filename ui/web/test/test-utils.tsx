import React from 'react';
import type { ReactElement } from 'react';
import { render } from '@testing-library/react';
import type { RenderOptions } from '@testing-library/react';

// Create a custom render function that can be used to render components with providers 
// such as context providers, theme providers, etc.

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  // Add any custom render options here, such as initial router state, etc.
  initialRouterEntries?: string[];
}

/**
 * Custom render function to render components with common providers
 * This allows components to access context providers and other wrappers
 * that would normally be available in the application.
 */
function customRender(
  ui: ReactElement,
  options?: CustomRenderOptions
) {
  const AllProviders = ({ children }: { children: React.ReactNode }) => {
    // Add providers here as needed
    return <>{children}</>;
  };

  return render(ui, { wrapper: AllProviders, ...options });
}

// Re-export everything from testing-library
export * from '@testing-library/react';

// Override the render method
export { customRender as render };

/**
 * Helper function to mock a React Router route component's props
 * This helps reduce boilerplate when testing route components.
 */
export function createMockRouteProps<LoaderData = any>(
  loaderData: LoaderData,
  params: Record<string, string> = {},
) {
  return {
    loaderData,
    params,
    matches: [
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
        params,
        data: loaderData,
        handle: {}
      }
    ]
  };
}
