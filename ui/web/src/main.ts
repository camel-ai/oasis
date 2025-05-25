import { Logger, ValidationPipe, VersioningType } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { NestExpressApplication } from '@nestjs/platform-express';

import { AppModule } from './app.module';
import { mountReactRouterHandler } from './react-router';
import { makeTable } from './utils';

const PORT = process.env.SERVER_PORT || 4003;
const PUBLIC_URL = process.env.SERVER_PUBLIC_URL || `http://localhost:${PORT}`;

const logger = new Logger('Bootstrap');

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule, {
    cors: true,
    logger,
  });

  app.setGlobalPrefix('api');
  app.disable('x-powered-by');
  app.useGlobalPipes(
    new ValidationPipe({
      transform: true,
      forbidUnknownValues: false,
    })
  );
  app.enableVersioning({
    type: VersioningType.URI,
  });

  await mountReactRouterHandler(app);

  await app.listen(PORT);

  const banner = makeTable(
    {
      'Base URL': PUBLIC_URL,
      'Hello API': `${PUBLIC_URL}/api/hello`,
    },
    `🚀 Server running 🚀`
  );
  banner.forEach((line) => logger.log(line));
}

bootstrap();
