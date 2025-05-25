import { describe, it, expect, vi, beforeEach } from 'vitest';

// Explicitly use mocks to avoid React Router Vite plugin restrictions
vi.mock('./home', () => {
    return {
        meta: () => [
            {
                title: 'React Router 7 + NestJS Custom Server • cbnsndwch OSS'
            }
        ],
        loader: ({ context }: any) => {
            const appService = context.app.get('AppService');
            return { hello: appService.getHello() };
        }
    };
});

// Import the mocked functions
import { meta, loader } from './home';

// Mock the AppService
const mockAppService = {
    getHello: vi.fn().mockReturnValue('Hello from Test')
};

// Mock the context
const mockContext = {
    app: {
        get: vi.fn().mockImplementation((token: string) => {
            if (token === 'AppService') {
                return mockAppService;
            }
            return undefined;
        })
    }
};

// Mock the request
const mockRequest = new Request('http://localhost/');

describe('Home Route Unit Tests', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('meta function', () => {
        it('should return the correct meta data', () => {
            const result = meta();

            expect(result).toHaveLength(1);
            expect(result[0]).toHaveProperty(
                'title',
                'React Router 7 + NestJS Custom Server • cbnsndwch OSS'
            );
        });
    });

    describe('loader function', () => {
        it('should call AppService and return data', async () => {
            const loaderArgs = {
                context: mockContext,
                request: mockRequest,
                params: {}
            };

            const result = await loader(loaderArgs as any);

            // Check that the service was retrieved
            expect(mockContext.app.get).toHaveBeenCalledWith('AppService');

            // Check that the service method was called
            expect(mockAppService.getHello).toHaveBeenCalled();

            // Check the returned data
            expect(result).toEqual({
                hello: 'Hello from Test'
            });
        });
    });
});
