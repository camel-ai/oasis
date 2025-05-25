// Mock of the home.tsx route exports for testing
// This avoids the React Router Vite plugin restrictions

// Mock the meta function
export const meta = () => {
    return [{ title: 'React Router 7 + NestJS Custom Server • cbnsndwch OSS' }];
};

// Mock the loader function
export async function loader({ context }: any) {
    // Extract the AppService from the context
    const appService = context.app.get('AppService');

    // Return the expected data
    return {
        hello: appService.getHello()
    };
}

// No need to mock the default export (component) as it's not used in tests
