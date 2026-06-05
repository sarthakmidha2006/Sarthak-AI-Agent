/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The backend (FastAPI on Railway/Render) is reached via NEXT_PUBLIC_API_URL.
  // No rewrites are required, but we expose a typed env default for local dev.
  env: {
    // Falls back to the production Railway backend so a fresh Vercel deploy
    // works even before NEXT_PUBLIC_API_URL is set in the dashboard.
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL ?? "https://sarthak-ai-agent-production.up.railway.app",
  },
};

export default nextConfig;
