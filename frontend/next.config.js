/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable fast refresh for better development experience
  reactStrictMode: true,
  
  // Ensure proper hot reloading
  webpack: (config, { dev, isServer }) => {
    if (dev && !isServer) {
      // Enable hot reloading
      config.watchOptions = {
        poll: 1000,
        aggregateTimeout: 300,
      }
    }
    return config
  },
  
  // Experimental features for better development
  experimental: {
    // App directory is now stable in Next.js 15
  },
}

module.exports = nextConfig 