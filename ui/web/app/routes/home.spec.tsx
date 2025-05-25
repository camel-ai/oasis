import { describe, it, expect, vi, beforeEach } from 'vitest';

import { createMockLoaderArgs } from '../../test/router-test-utils';

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

describe('Home Route Functions', () => {
    const mockAppService = {
        getHello: vi.fn().mockReturnValue('Hello from NestJS and React Router!')
    };

    const mockContext = {
        app: {
            get: vi.fn().mockReturnValue(mockAppService)
        }
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('meta function', () => {
        it('should return correct meta descriptors', () => {
            const metaResult = meta();

            expect(metaResult).toHaveLength(1);
            expect(metaResult[0]).toHaveProperty(
                'title',
                'React Router 7 + NestJS Custom Server • cbnsndwch OSS'
            );
        });
    });

    describe('loader function', () => {
        it('should fetch greeting from AppService', async () => {
            // Create test args using the utility function
            const args = createMockLoaderArgs({ context: mockContext });

            // Call the loader function
            const loaderResult = await loader(args as any);

            // Verify service was called
            expect(mockContext.app.get).toHaveBeenCalledWith('AppService');
            expect(mockAppService.getHello).toHaveBeenCalled();

            // Verify correct result
            expect(loaderResult).toEqual({
                hello: 'Hello from NestJS and React Router!'
            });
        });
    });
});
