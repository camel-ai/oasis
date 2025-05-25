import {
  type RouteConfig,
  index,
  layout,
  route,
} from '@react-router/dev/routes';

export default [
  layout('./routes/layout.tsx', [
    index('./routes/home.tsx'),
    route('simulation', './routes/simulation/index.tsx'),
    route('history', './routes/history/index.tsx'),
  ]),
] satisfies RouteConfig;
