import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rack Guardian",
  description: "AI thermal co-pilot for GPU clusters"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
