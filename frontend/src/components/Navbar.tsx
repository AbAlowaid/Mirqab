'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

export default function Navbar() {
  const pathname = usePathname()

  const isActive = (path: string) => pathname === path

  return (
    <nav className="relative bg-gradient-to-r from-blue-900 via-blue-800 to-blue-900 text-white shadow-2xl">
      {/* Top line */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600"></div>
      
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-20">
          {/* Logo Section */}
          <Link href="/" className="flex items-center space-x-3 group">
            <img 
              src="/Mirqab_white (1).png" 
              alt="Mirqab Logo" 
              className="h-12 object-contain group-hover:scale-105 transition-transform"
            />
            <div>
              <div className="text-2xl font-bold tracking-wider text-white group-hover:text-blue-300 transition-colors">
                Mirqab
              </div>
              <div className="text-xs text-gray-300 uppercase tracking-widest">
                Detection System
              </div>
            </div>
          </Link>
          
          {/* Navigation Links */}
          <div className="flex space-x-2">
            <NavLink href="/" isActive={isActive('/')}>
              Home
            </NavLink>
            <NavLink href="/detection-reports" isActive={isActive('/detection-reports')}>
              Detection Reports
            </NavLink>
            <NavLink href="/upload" isActive={isActive('/upload')}>
              Upload
            </NavLink>
            <NavLink href="/moraqib" isActive={isActive('/moraqib')}>
              Moraqib AI
            </NavLink>
            <NavLink href="/about" isActive={isActive('/about')}>
              About
            </NavLink>
          </div>
        </div>
      </div>
      
      {/* Bottom line */}
      <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-blue-400 to-transparent"></div>
    </nav>
  )
}

// Navigation Link Component
function NavLink({ href, isActive, children }: { href: string; isActive: boolean; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className={`
        relative px-6 py-2 font-semibold text-sm uppercase tracking-wider
        transition-all duration-300 rounded-md
        ${isActive 
          ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-glow-blue scale-105' 
          : 'text-gray-200 hover:text-white hover:bg-blue-700/50'
        }
      `}
    >
      <span>{children}</span>
      {isActive && (
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white"></div>
      )}
    </Link>
  )
}
