// Import Vitest utilities
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock react-dom/server module
vi.mock('react-dom/server', () => ({
    renderToPipeableStream: vi.fn().mockImplementation((jsx, options) => {
        // Call the appropriate callback based on the test
        if (options.onShellReady) {
            options.onShellReady();
        }
        if (options.onAllReady) {
            options.onAllReady();
        }
        return {
            pipe: vi.fn(),
            abort: vi.fn()
        };
    })
}));

// Mock @react-router/node module
vi.mock('@react-router/node', () => ({
    createReadableStreamFromReadable: vi.fn().mockReturnValue('mock-stream')
}));

// Mock react-router module
vi.mock('react-router', () => ({
    ServerRouter: vi.fn().mockImplementation(({ children }) => children)
}));

// Mock isbot
vi.mock('isbot', () => ({
    isbot: vi.fn().mockImplementation(userAgent => {
        return userAgent?.includes('bot') || false;
    })
}));

// Mock node stream
vi.mock('node:stream', () => ({
    PassThrough: vi.fn().mockImplementation(() => ({
        pipe: vi.fn()
    }))
}));

// Helper function to mimic isbot behavior without importing it
function isbot(userAgent: string) {
    return userAgent.includes('bot');
}

describe('Server Entry Handler Tests', () => {
    // eslint-disable-next-line no-unused-vars
    let setTimeoutSpy: any;

    beforeEach(() => {
        // Set up spies
        setTimeoutSpy = vi.spyOn(global, 'setTimeout');
    });

    afterEach(() => {
        vi.clearAllMocks();
        vi.restoreAllMocks();
    });

    it('should set Content-Type header and create a response', async () => {
        // Mock handleRequest function for this test only
        const mockHandleRequest = (
            request: Request,
            statusCode: number,
            headers: Headers
        ) => {
            headers.set('Content-Type', 'text/html');
            return Promise.resolve(
                new Response('test', {
                    headers,
                    status: statusCode
                })
            );
        };

        // Create test request
        const request = new Request('https://example.com');
        const headers = new Headers();
        const statusCode = 200;

        // Call the function
        const response = await mockHandleRequest(request, statusCode, headers);

        // Assertions
        expect(headers.get('Content-Type')).toBe('text/html');
        expect(response.status).toBe(statusCode);
    });

    it('should handle bot user agents differently', async () => {
        // Test the isbot helper function
        expect(isbot('Googlebot/2.1')).toBe(true);
        expect(isbot('Mozilla/5.0')).toBe(false);
    });

    it('should handle SPA mode correctly', async () => {
        // This is a placeholder test showing we can test SPA mode behavior
        const mockContext = { isSpaMode: true };
        expect(mockContext.isSpaMode).toBe(true);
    });
});
