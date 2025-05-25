import {
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
} from 'react-router';

import './index.css';
import type { Route } from './+types/root';
import I18nProvider from './i18n/i18nProvider';
import { Toaster } from './components/ui/sonner';

export async function loader() {
  return { user: { foo: 'bar' } };
}

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang='en'>
      <head>
        <meta charSet='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <Meta />
        <Links />
        <title>Simulation</title>
      </head>
      <body>
        {children}
        <Toaster
          toastOptions={{
            className: 'rounded !border-none',
          }}
        />
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}
export default function App() {
  return (
    <>
      <I18nProvider>
        <Outlet />
      </I18nProvider>
      <div
        className='pointer-events-none fixed inset-0 z-[1000]'
        style={{
          // @ts-ignore
          backgroundImage: `url('${import.meta.env.BASE_URL ?? ''}assets/noise.png')`,
          backgroundRepeat: 'repeat',
          backgroundSize: '128px',
          opacity: 0.06,
        }}
      ></div>
    </>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  let message = 'Oops!';
  let details = 'An unexpected error occurred.';
  let stack: string | undefined;

  if (isRouteErrorResponse(error)) {
    message = error.status === 404 ? '404' : 'Error';
    details =
      error.status === 404
        ? 'The requested page could not be found.'
        : error.statusText || details;
  } else if (import.meta.env.DEV && error && error instanceof Error) {
    details = error.message;
    stack = error.stack;
  }

  return (
    <main className='pt-16 p-4 container mx-auto'>
      <h1>{message}</h1>
      <p>{details}</p>
      {stack && (
        <pre className='w-full p-4 overflow-x-auto'>
          <code>{stack}</code>
        </pre>
      )}
    </main>
  );
}
