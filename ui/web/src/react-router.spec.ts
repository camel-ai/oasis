// In Vitest, vi.mock calls are hoisted to the top of the file
// So we must define them before any imports

// Mock all dependencies
vi.mock('@react-router/express', () => ({
    createRequestHandler: vi.fn().mockReturnValue(vi.fn())
}));

vi.mock('vite', () => ({
    createServer: vi.fn().mockResolvedValue({
        ssrLoadModule: vi.fn().mockResolvedValue({}),
        middlewares: {}
    })
}));

vi.mock('express', () => ({
    static: vi.fn().mockReturnValue({})
}));

vi.mock('node:path', async () => {
    const actual = await vi.importActual('node:path');
    return {
        ...actual,
        resolve: vi.fn().mockReturnValue('mocked-build-path')
    };
});

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Test } from '@nestjs/testing';
import { NestExpressApplication } from '@nestjs/platform-express';

import { AppModule } from './app.module';
import { mountReactRouterHandler } from './react-router';

describe('React Router Integration', () => {
    let app: NestExpressApplication;
    let mockExpressInstance: any;

    beforeEach(async () => {
        // Reset all mocks
        vi.clearAllMocks();

        // Mock express instance
        mockExpressInstance = {
            use: vi.fn(),
            all: vi.fn()
        };

        // Set up NestJS app
        const moduleRef = await Test.createTestingModule({
            imports: [AppModule]
        }).compile();

        app = moduleRef.createNestApplication<NestExpressApplication>();
        app.getHttpAdapter = vi.fn().mockReturnValue({
            getInstance: vi.fn().mockReturnValue(mockExpressInstance)
        });
    });

    it('should set up React Router handler correctly', async () => {
        // Execute function under test
        await mountReactRouterHandler(app);

        // Verify correct behavior
        expect(mockExpressInstance.all).toHaveBeenCalledWith(
            '/{*all}',
            expect.any(Function)
        );
    });
});
