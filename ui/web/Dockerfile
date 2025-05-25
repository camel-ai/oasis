# Stage 1: Build the application
FROM node:22-alpine AS builder

# Enable pnpm and set up cache
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable

WORKDIR /app

# Copy package.json and lockfile first to leverage Docker cache
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/server/package.json ./apps/server/

# Install dependencies using pnpm with store cache
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store pnpm install --no-frozen-lockfile

# Copy the rest of the code
COPY . .

# Build the application
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store pnpm --filter @cbnsndwch/react-router-nest-server run build

# Stage 2: Production image
FROM node:22-alpine

# Add metadata labels
LABEL org.opencontainers.image.title="React Router Nest Server"
LABEL org.opencontainers.image.description="A demo server application built with React Router and NestJS"
LABEL org.opencontainers.image.source="https://github.com/cbnsndwch/react-router-nest"
LABEL org.opencontainers.image.licenses="MIT"

# Set environment
ENV NODE_ENV=production
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable

WORKDIR /app

# Copy package.json and lockfile
COPY --from=builder /app/apps/server/package.json /app/package.json

# Install production dependencies using the same cache
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store pnpm install --prod --no-frozen-lockfile

# Copy all built application artifacts
COPY --from=builder /app/apps/server/dist /app/dist
COPY --from=builder /app/apps/server/build /app/build

# Expose the application port
EXPOSE 3000

# Run the application
CMD ["node", "dist/main.js"]
