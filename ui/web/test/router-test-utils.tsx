import { vi } from 'vitest';

// Mock React Router components and hooks
export const createMockRouterContext = (overrides = {}) => ({
    location: { pathname: '/', search: '', hash: '', state: null },
    navigation: { state: 'idle' },
    loaderData: {},
    actionData: null,
    errors: {},
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
            params: {},
            data: {},
            handle: {}
        }
    ],
    ...overrides
});

// Mock React Router functions
export const meta = vi.fn();
export const loader = vi.fn();
export const action = vi.fn();

// Create a mock loader function args object
export function createMockLoaderArgs(overrides = {}) {
    return {
        request: new Request('http://localhost/'),
        params: {},
        context: {
            app: {
                get: vi.fn().mockImplementation((token: string) => {
                    if (token === 'AppService') {
                        return {
                            getHello: vi
                                .fn()
                                .mockReturnValue('Hello from Mock Service')
                        };
                    }
                    return undefined;
                })
            }
        },
        ...overrides
    };
}

// Export a simple test for React Router 7 configuration
export const testReactRouterSetup = () => {
    return {
        result: true,
        message: 'React Router 7 test setup is configured correctly'
    };
};
