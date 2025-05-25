/* eslint-disable spaced-comment */
/// <reference types="vite/client" />
/// <reference types="react-router" />
/// <reference types="@react-router/node" />

import type { LoaderFunction as RRLoaderFunction } from 'react-router';
import type { NestApplication } from '@nestjs/core';

declare global {
  interface AppLoadContext {
    app: NestApplication;
  }

  type LoaderFunction = RRLoaderFunction<AppLoadContext>;

  type LoaderFunctionArgs = Parameters<LoaderFunction>[0];
}
