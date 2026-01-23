/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverActions: true,
  },
  // Enable WebAssembly
  webpack: (config, { isServer }) => {
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    };

    // Optimize bundle size
    config.optimization = {
      ...config.optimization,
      moduleIds: 'deterministic',
    };

    return config;
  },
  // Optimize images
  images: {
    formats: ['image/avif', 'image/webp'],
    domains: [],
  },
};

module.exports = nextConfig;
