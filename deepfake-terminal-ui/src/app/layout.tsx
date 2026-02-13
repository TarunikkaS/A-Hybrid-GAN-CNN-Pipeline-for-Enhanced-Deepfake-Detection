import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DEEPFAKE DETECTOR // TERMINAL v1.0",
  description: "Neural network-powered deepfake detection system with visual attribution",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {/* CRT Scanline Effects */}
        <div className="crt-overlay" />
        <div className="crt-scanline" />
        
        {/* Main Content */}
        {children}
      </body>
    </html>
  );
}
