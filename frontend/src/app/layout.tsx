import type { Metadata } from 'next'
import './globals.css'
import Navbar from '@/components/Navbar'
import MoraqibPromo from '@/components/MoraqibPromo'

export const metadata: Metadata = {
  title: 'Mirqab - Advanced Detection System',
  description: 'AI-powered detection and monitoring system',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main className="min-h-screen relative">
          {children}
        </main>
        
        {/* Promotional CTA Box */}
        <MoraqibPromo />
        
        {/* Simplified Footer */}
        <footer className="relative bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white py-12 mt-16">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600"></div>
          <div className="container mx-auto px-4">
            <div className="max-w-4xl mx-auto text-center">
              {/* Mirqab Logo */}
              <div className="flex justify-center mb-4">
                <img 
                  src="/Mirqab_white (1).png" 
                  alt="Mirqab Logo" 
                  className="h-16 object-contain"
                />
              </div>
              
              {/* System Description */}
              <p className="text-gray-300 mb-8 max-w-2xl mx-auto">
                Advanced AI-powered detection and monitoring system for intelligent analysis and automated reporting
              </p>
              
              {/* Project Info */}
              <div className="border-t border-gray-700 pt-6">
                <h4 className="text-sm font-semibold mb-2 text-blue-400 uppercase tracking-wider">
                  Project Info
                </h4>
                <p className="text-gray-400">
                  Data Science & ML Bootcamp<br />
                  Tuwaiq Academy 2025
                </p>
              </div>
            </div>
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-blue-500 to-transparent opacity-30"></div>
        </footer>
      </body>
    </html>
  )
}
