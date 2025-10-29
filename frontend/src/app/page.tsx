'use client'

import Link from 'next/link'

export default function HomePage() {
  const metrics = {
    train: {
      iou: 0.9248,
      precision: 0.9572,
      recall: 0.9625,
      f1: 0.9598,
      map: 0.9598
    },
    validation: {
      iou: 0.9245,
      precision: 0.9574,
      recall: 0.9610,
      f1: 0.9592,
      map: 0.9592
    }
  }

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-blue-900 via-blue-800 to-purple-900 text-white py-20 fade-in">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="container mx-auto px-4 relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            {/* Mirqab Logo */}
            <div className="flex justify-center mb-8">
              <img 
                src="/Mirqab_Logo.png" 
                alt="Mirqab Logo" 
                className="h-60 object-contain"
              />
            </div>
            <p className="text-2xl text-blue-100 mb-8 font-light">
              Advanced AI-powered detection system using state-of-the-art deep learning technology for intelligent monitoring and automated analysis
            </p>
            
            {/* Ready to Test CTA Box */}
            <div className="mt-12 bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl p-8 shadow-2xl max-w-xl mx-auto">
              <h2 className="text-3xl font-bold mb-4">Ready to Test the System?</h2>
              <p className="text-blue-100 mb-6">
                Experience the power of AI-driven detection technology
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link href="/upload">
                  <button className="bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white font-bold py-4 px-8 rounded-lg text-lg transition-all transform hover:scale-105 shadow-glow-blue w-full sm:w-auto">
                    Upload Analysis
                  </button>
                </Link>
                <Link href="/moraqib">
                  <button className="bg-gradient-to-r from-purple-600 to-purple-500 hover:from-purple-700 hover:to-purple-600 text-white font-bold py-4 px-8 rounded-lg text-lg transition-all transform hover:scale-105 shadow-glow-purple w-full sm:w-auto">
                    Try Moraqib AI
                  </button>
                </Link>
              </div>
            </div>
          </div>
        </div>
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600"></div>
      </section>

      <div className="container mx-auto px-4 py-16">
        {/* Project Overview Section */}
        <section className="mb-16 max-w-5xl mx-auto">
          <div className="professional-card rounded-2xl shadow-lg p-8">
            <h3 className="text-4xl font-bold text-gray-900 mb-8 text-center">Project Overview</h3>
            
            <div className="grid md:grid-cols-2 gap-8 mb-8">
              {/* The Challenge */}
              <div className="bg-gradient-to-br from-red-50 to-orange-50 rounded-xl p-6 border-l-4 border-red-500">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-red-500 rounded-lg flex items-center justify-center mr-4">
                    <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <h4 className="text-2xl font-bold text-gray-800">The Challenge</h4>
                </div>
                <p className="text-gray-700 leading-relaxed">
                  Manual camouflage detection is slow, error-prone, and inefficient. Defense organizations need 
                  <strong> automated, accurate, real-time solutions</strong> for threat identification.
                </p>
              </div>

              {/* Our Solution */}
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border-l-4 border-green-500">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center mr-4">
                    <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <h4 className="text-2xl font-bold text-gray-800">Our Solution</h4>
                </div>
                <p className="text-gray-700 leading-relaxed">
                  AI-powered detection using <strong>DeepLabV3 Computer Vision Model</strong> to instantly identify 
                  camouflaged personnel with automated intelligence reports.
                </p>
              </div>
            </div>

            {/* Brief Description */}
            <div className="bg-blue-50 rounded-xl p-6 mb-8 border border-blue-200">
              <p className="text-lg text-gray-800 text-center leading-relaxed">
                A <strong>capstone project</strong> combining machine learning, computer vision, and web development 
                to deliver practical defense solutions for Saudi Arabia's security organizations.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg border-2 border-green-200">
                <svg className="w-12 h-12 mx-auto mb-2 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l9-5-9-5-9 5 9 5z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l9-5-9-5-9 5 9 5zm0 0l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14zm-4 6v-7.5l4-2.222" />
                </svg>
                <h4 className="font-bold text-lg mb-2 text-gray-800">Academic Excellence</h4>
                <p className="text-sm text-gray-600">
                  Built on cutting-edge research in deep learning
                </p>
              </div>
              <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border-2 border-blue-200">
                <svg className="w-12 h-12 mx-auto mb-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <h4 className="font-bold text-lg mb-2 text-gray-800">Modern Tech Stack</h4>
                <p className="text-sm text-gray-600">
                  Leveraging React, FastAPI, and PyTorch
                </p>
              </div>
              <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg border-2 border-purple-200">
                <svg className="w-12 h-12 mx-auto mb-2 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <h4 className="font-bold text-lg mb-2 text-gray-800">Real-World Impact</h4>
                <p className="text-sm text-gray-600">
                  Practical applications in security and defense
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* System Capabilities Section */}
        <section className="mb-16 max-w-5xl mx-auto">
          <h3 className="text-4xl font-bold text-gray-900 mb-8 text-center">
            System Capabilities
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="professional-card p-6">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-blue-500 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h4 className="text-xl font-semibold mb-3">Accurate Detection</h4>
              <p className="text-gray-700">
                High precision detection using DeepLabV3 ResNet-101 semantic segmentation model
              </p>
            </div>

            <div className="professional-card p-6">
              <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-purple-500 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h4 className="text-xl font-semibold mb-3">Intelligent Reporting</h4>
              <p className="text-gray-700">
                AI-powered reports with detailed analysis using OpenAI GPT-4 Turbo Vision API
              </p>
            </div>

            <div className="professional-card p-6">
              <div className="w-12 h-12 bg-gradient-to-br from-green-600 to-green-500 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </div>
              <h4 className="text-xl font-semibold mb-3">User-Friendly Interface</h4>
              <p className="text-gray-700">
                Intuitive web platform for image upload and real-time analysis
              </p>
            </div>

            <div className="professional-card p-6">
              <div className="w-12 h-12 bg-gradient-to-br from-orange-600 to-orange-500 rounded-lg flex items-center justify-center mb-4">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <h4 className="text-xl font-semibold mb-3">Automated Documentation</h4>
              <p className="text-gray-700">
                Exportable PDF reports with detection metadata and location tracking
              </p>
            </div>
          </div>
        </section>

        {/* Project Beneficiaries Section */}
        <section className="mb-16 max-w-5xl mx-auto">
          <div className="professional-card rounded-2xl shadow-lg p-8">
            <h3 className="text-4xl font-bold text-gray-900 mb-4 text-center">
              Project Beneficiaries
            </h3>
            <p className="text-lg text-gray-700 text-center mb-8">
              This system is designed to benefit key defense and security organizations in Saudi Arabia
            </p>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center p-6 bg-white rounded-lg shadow-md hover:shadow-xl transition">
                <div className="flex items-center justify-center h-24 mb-4">
                  <img 
                    src="/SAMI.webp" 
                    alt="SAMI Logo" 
                    className="max-h-20 max-w-full object-contain"
                  />
                </div>
                <h4 className="font-bold text-lg mb-2">SAMI</h4>
                <p className="text-sm text-gray-600">
                  Saudi Arabian Military Industries - Advancing defense manufacturing capabilities
                </p>
              </div>
              <div className="text-center p-6 bg-white rounded-lg shadow-md hover:shadow-xl transition">
                <div className="flex items-center justify-center h-24 mb-4">
                  <img 
                    src="/GAMI.webp" 
                    alt="GAMI Logo" 
                    className="max-h-20 max-w-full object-contain"
                  />
                </div>
                <h4 className="font-bold text-lg mb-2">GAMI</h4>
                <p className="text-sm text-gray-600">
                  General Authority for Military Industries - Supporting defense sector development
                </p>
              </div>
              <div className="text-center p-6 bg-white rounded-lg shadow-md hover:shadow-xl transition">
                <div className="flex items-center justify-center h-24 mb-4">
                  <img 
                    src="/SAFCSP.png" 
                    alt="SAFCSP Logo" 
                    className="max-h-20 max-w-full object-contain"
                  />
                </div>
                <h4 className="font-bold text-lg mb-2">SAFCSP</h4>
                <p className="text-sm text-gray-600">
                  Saudi Federation for Cybersecurity, Programming and Drones
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
