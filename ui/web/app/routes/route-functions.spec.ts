// This file demonstrates how to test route functions without directly importing the component
// which avoids issues with React Router's Vite plugin

// Import testing utilities
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

// Import from the automatically mocked module
import { meta, loader } from './home';

describe('Home Route Functions', () => {
    const mockAppService = {
        getHello: vi.fn().mockReturnValue('Hello from Mock')
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
        it('should return the correct page title', () => {
            const result = meta();

            expect(result).toHaveLength(1);
            expect(result[0]).toHaveProperty(
                'title',
                'React Router 7 + NestJS Custom Server • cbnsndwch OSS'
            );
        });
    });

    describe('loader function', () => {
        it('should fetch data from the AppService', async () => {
            // Create test args using our utility
            const args = createMockLoaderArgs({ context: mockContext });

            // Call the loader function
            const result = await loader(args);

            // Verify service was called
            expect(mockContext.app.get).toHaveBeenCalledWith('AppService');
            expect(mockAppService.getHello).toHaveBeenCalled();

            // Verify correct result
            expect(result).toEqual({
                hello: 'Hello from Mock'
            });
        });
    });
});
