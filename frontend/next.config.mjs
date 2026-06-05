/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The backend (FastAPI on Railway/Render) is reached via NEXT_PUBLIC_API_URL.
  // No rewrites are required, but we expose a typed env default for local dev.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
};

export default nextConfig;
