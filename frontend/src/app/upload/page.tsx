'use client'

import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import ReportModal from '@/components/ReportModal'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface DetectionResult {
  success: boolean
  message?: string
  has_camouflage?: boolean
  detection: boolean
  soldier_count: number
  civilian_count?: number
  total_detections?: number
  detections: any[]
  overlay_image: string
  original_image: string
  class_breakdown?: {
    camouflage_soldiers: number
    civilians: number
    total: number
  }
  report?: {
    report_id: string
    timestamp: string
    location: { latitude: number; longitude: number } | null
    analysis: {
      summary: string
      environment: string
      soldier_count: number
      attire_and_camouflage: string
      equipment: string
    }
    images: {
      original_base64: string
      masked_base64: string
    }
  }
}

export default function UploadPage() {
  const [uploadMode, setUploadMode] = useState<'image' | 'video'>('image')
  const [file, setFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [showReport, setShowReport] = useState(false)
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null)
  const [videoProgress, setVideoProgress] = useState(0)
  const [estimatedTime, setEstimatedTime] = useState(0)
  const [elapsedTime, setElapsedTime] = useState(0)
  const [noCamouflageDetected, setNoCamouflageDetected] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (selectedFile: File) => {
    if (uploadMode === 'image') {
      const validTypes = ['image/jpeg', 'image/png', 'image/jpg']
      if (validTypes.includes(selectedFile.type)) {
        setFile(selectedFile)
        setResult(null)
        setProcessedVideoUrl(null)
      } else {
        alert('Please upload a valid image (JPG, PNG) file')
      }
    } else {
      const validTypes = ['video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo']
      if (validTypes.includes(selectedFile.type) || selectedFile.name.endsWith('.mp4') || selectedFile.name.endsWith('.avi') || selectedFile.name.endsWith('.mov')) {
        setFile(selectedFile)
        setResult(null)
        setProcessedVideoUrl(null)
      } else {
        alert('Please upload a valid video (MP4, AVI, MOV) file')
      }
    }
  }

  const handleModeChange = (mode: 'image' | 'video') => {
    setUploadMode(mode)
    setFile(null)
    setResult(null)
    setProcessedVideoUrl(null)
    setVideoProgress(0)
    setEstimatedTime(0)
    setElapsedTime(0)
    setNoCamouflageDetected(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const getLocation = (): Promise<{ latitude: number; longitude: number }> => {
    return new Promise((resolve) => {
      if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            resolve({
              latitude: position.coords.latitude,
              longitude: position.coords.longitude
            })
          },
          () => {
            resolve({ latitude: 0, longitude: 0 })
          }
        )
      } else {
        resolve({ latitude: 0, longitude: 0 })
      }
    })
  }

  const handleUpload = async () => {
    if (!file) return

    setLoading(true)
    setResult(null)
    setProcessedVideoUrl(null)
    setVideoProgress(0)
    setElapsedTime(0)

    try {
      const loc = await getLocation()

      const formData = new FormData()
      formData.append('file', file)
      formData.append('location', JSON.stringify({ lat: loc.latitude, lng: loc.longitude }))

      if (uploadMode === 'video') {
        // Estimate processing time based on file size (optimized: ~4-6 seconds per MB with frame skipping)
        const fileSizeMB = file.size / (1024 * 1024)
        const estimated = Math.ceil(fileSizeMB * 5) // seconds (3x faster with optimization)
        setEstimatedTime(estimated)

        // Start elapsed time counter
        const startTime = Date.now()
        const progressInterval = setInterval(() => {
          const elapsed = Math.floor((Date.now() - startTime) / 1000)
          setElapsedTime(elapsed)
          
          // Calculate progress based on elapsed vs estimated time
          // Cap at 95% until actual completion
          const calculatedProgress = Math.min((elapsed / estimated) * 100, 95)
          setVideoProgress(calculatedProgress)
        }, 1000)

        try {
          // Process video - no timeout limit for large videos
          const response = await axios.post(`${API_URL}/api/process_video`, formData, {
            headers: { 
              'Content-Type': 'multipart/form-data',
              'ngrok-skip-browser-warning': 'true'
            },
            timeout: 0, // No timeout - let it process completely
            responseType: 'blob' // Receive video file as blob
          })

          clearInterval(progressInterval)
          setVideoProgress(100)

          // Create URL for the processed video
          const videoBlob = new Blob([response.data], { type: 'video/mp4' })
          const videoUrl = URL.createObjectURL(videoBlob)
          setProcessedVideoUrl(videoUrl)
        } catch (error) {
          clearInterval(progressInterval)
          throw error
        }

      } else {
        // Process image
        const response = await axios.post(`${API_URL}/api/analyze_media`, formData, {
          headers: { 
            'Content-Type': 'multipart/form-data',
            'ngrok-skip-browser-warning': 'true'
          },
          timeout: 120000 // 2 minutes for detection + AI analysis
        })

        const data: DetectionResult = response.data
        console.log('Detection result:', {
          hasOverlay: !!data.overlay_image,
          overlayLength: data.overlay_image?.length,
          soldierCount: data.soldier_count,
          success: data.success,
          hasCamouflage: (data as any).has_camouflage
        })

        // Check if no camouflaged soldiers were detected
        if (data.success === false && (data as any).message) {
          setNoCamouflageDetected(true)
          setResult(null)
          return
        }

        setResult(data)
        setNoCamouflageDetected(false)

        // Show report modal automatically if soldiers detected
        if (data.detection && data.report) {
          setShowReport(true)
        }
      }

    } catch (error: any) {
      console.error('Upload error:', error)
      alert('Error analyzing file: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setShowReport(false)
    setProcessedVideoUrl(null)
    setNoCamouflageDetected(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
            Upload Analysis
          </h1>
          <p className="text-xl text-gray-600">
            Upload media to detect and segment camouflaged soldiers
          </p>
        </div>

        {/* Mode Selection */}
        <div className="max-w-xl mx-auto mb-8">
          <div className="professional-card p-2 flex gap-2">
            <button
              onClick={() => handleModeChange('image')}
              className={`flex-1 py-4 px-6 rounded-lg font-bold text-lg transition-all duration-300 ${
                uploadMode === 'image'
                  ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-glow-blue scale-105'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <span className="flex items-center justify-center gap-3">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span>Image Analysis</span>
              </span>
            </button>
            <button
              onClick={() => handleModeChange('video')}
              className={`flex-1 py-4 px-6 rounded-lg font-bold text-lg transition-all duration-300 ${
                uploadMode === 'video'
                  ? 'bg-gradient-to-r from-purple-600 to-purple-500 text-white shadow-glow-purple scale-105'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <span className="flex items-center justify-center gap-3">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>Video Processing</span>
              </span>
            </button>
          </div>
          <p className="text-center text-sm text-gray-500 mt-3">
            {uploadMode === 'image' 
              ? 'Analyze single images with AI-powered detection reports' 
              : 'Process videos to generate segmented overlay footage'}
          </p>
        </div>

        <div className="max-w-3xl mx-auto mb-12">
          <div
            className={`professional-card border-4 border-dashed rounded-2xl p-12 text-center transition-all ${
              dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={uploadMode === 'image' ? 'image/*' : 'video/*'}
              onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
              className="hidden"
            />

            {file ? (
              <div className="space-y-4">
                <svg className="w-20 h-20 mx-auto text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-2xl font-semibold text-blue-600">{file.name}</p>
                <p className="text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                <button
                  onClick={handleReset}
                  className="px-6 py-3 bg-gray-200 hover:bg-gray-300 rounded-lg font-semibold transition-colors"
                >
                  Choose Different File
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <svg className="w-20 h-20 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  {uploadMode === 'image' ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  )}
                </svg>
                <p className="text-2xl font-semibold">Drag & Drop your {uploadMode} here</p>
                <p className="text-gray-500">or</p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-primary px-8 py-4 text-lg"
                >
                  Browse {uploadMode === 'image' ? 'Images' : 'Videos'}
                </button>
                <p className="text-sm text-gray-400 mt-4">
                  Supported: {uploadMode === 'image' ? 'JPG, PNG' : 'MP4, AVI, MOV'}
                </p>
              </div>
            )}
          </div>

          {file && !result && !processedVideoUrl && (
            <button
              onClick={handleUpload}
              disabled={loading}
              className={`mt-6 w-full py-4 rounded-lg font-bold text-xl transition-all ${
                loading
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'btn-primary text-white shadow-lg hover:shadow-xl'
              }`}
            >
              {loading 
                ? (uploadMode === 'image' ? 'Analyzing Image...' : 'Processing Video...')
                : (uploadMode === 'image' ? 'Analyze Image' : 'Process Video')}
            </button>
          )}

          {/* No Camouflage Detected Message */}
          {noCamouflageDetected && (
            <div className="mt-6 professional-card rounded-xl p-8 bg-gradient-to-br from-yellow-50 to-orange-50 border-2 border-yellow-300">
              <div className="flex items-center justify-center gap-4 mb-4">
                <svg className="w-16 h-16 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <h3 className="text-2xl font-bold text-yellow-800 mb-2">No Camouflage Soldier Detected</h3>
                  <p className="text-gray-700">
                    The system did not detect any soldiers wearing camouflage patterns in this {uploadMode}.
                  </p>
                </div>
              </div>

              <button
                onClick={handleReset}
                className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all"
              >
                Try Another {uploadMode === 'image' ? 'Image' : 'Video'}
              </button>
            </div>
          )}

          {/* Video Processing Progress */}
          {loading && uploadMode === 'video' && (
            <div className="mt-6 professional-card rounded-xl p-6 bg-gradient-to-br from-blue-50 to-purple-50">
              <div className="space-y-4">
                {/* Progress Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                    <span className="font-bold text-lg text-gray-800">Processing Video Frame by Frame...</span>
                  </div>
                  <span className="text-2xl font-bold text-blue-600">{Math.round(videoProgress)}%</span>
                </div>

                {/* Progress Bar */}
                <div className="relative h-6 bg-gray-200 rounded-full overflow-hidden border-2 border-blue-300">
                  <div 
                    className="absolute top-0 left-0 h-full bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 transition-all duration-1000 ease-out flex items-center justify-end pr-3"
                    style={{ width: `${videoProgress}%` }}
                  >
                    {videoProgress > 10 && (
                      <span className="text-white text-xs font-bold drop-shadow-lg">
                        {Math.round(videoProgress)}%
                      </span>
                    )}
                  </div>
                </div>

                {/* Time Information */}
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div className="bg-white/50 rounded-lg p-3 border border-blue-200">
                    <div className="text-xs text-gray-600 font-semibold mb-1">Elapsed Time</div>
                    <div className="text-xl font-bold text-blue-600">
                      {Math.floor(elapsedTime / 60)}:{String(elapsedTime % 60).padStart(2, '0')}
                    </div>
                  </div>
                  <div className="bg-white/50 rounded-lg p-3 border border-blue-200">
                    <div className="text-xs text-gray-600 font-semibold mb-1">Estimated Total</div>
                    <div className="text-xl font-bold text-blue-600">
                      {Math.floor(estimatedTime / 60)}:{String(estimatedTime % 60).padStart(2, '0')}
                    </div>
                  </div>
                </div>

                {/* Info Message */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-sm text-gray-700 text-center">
                    <span className="font-bold">⚡ Fast Processing:</span> Our optimized AI model analyzes key frames and applies camouflage segmentation overlay.
                    3x faster than standard frame-by-frame processing.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Video Results Section */}
        {processedVideoUrl && (
          <div className="max-w-4xl mx-auto space-y-8">
            <div className="professional-card rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-6 flex items-center gap-3">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>Processed Video</span>
              </h2>

              <div className="space-y-6">
                {/* Video Player */}
                <div className="bg-black rounded-xl overflow-hidden shadow-2xl">
                  <video
                    controls
                    className="w-full"
                    src={processedVideoUrl}
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>

                {/* Download Section */}
                <div className="grid md:grid-cols-2 gap-4">
                  <a
                    href={processedVideoUrl}
                    download="segmented_video.mp4"
                    className="btn-primary py-4 text-center shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    <span>Download Segmented Video</span>
                  </a>
                  <button
                    onClick={handleReset}
                    className="py-4 bg-gray-200 hover:bg-gray-300 rounded-lg font-semibold transition-colors"
                  >
                    Process Another Video
                  </button>
                </div>

                {/* Info Box */}
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6 border-l-4 border-blue-500">
                  <h3 className="font-bold text-lg mb-2 text-blue-800">Video Processing Complete</h3>
                  <p className="text-sm text-gray-700">
                    Your video has been processed with real-time segmentation overlay showing detected camouflaged soldiers.
                    Red overlay indicates detected military personnel throughout the footage.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Image Results Section */}
        {result && result.success && (
          <div className="max-w-4xl mx-auto space-y-8">
            <div className="professional-card rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-6 flex items-center gap-3">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Detection Results</span>
              </h2>
              
              <div className="grid md:grid-cols-2 gap-8">
                {/* Image Display */}
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-gray-800">Detection Overlay</h3>
                  {result.overlay_image ? (
                    <img
                      src={result.overlay_image.startsWith('data:') ? result.overlay_image : `data:image/jpeg;base64,${result.overlay_image}`}
                      alt="Detection Result"
                      className="w-full rounded-lg shadow-lg border-2 border-gray-200"
                      onError={(e) => {
                        console.error('Image failed to load', result.overlay_image.substring(0, 100))
                        e.currentTarget.style.display = 'none'
                        const parent = e.currentTarget.parentElement
                        if (parent) {
                          const errorDiv = document.createElement('div')
                          errorDiv.className = 'w-full h-64 bg-gray-200 rounded-lg flex items-center justify-center text-gray-500'
                          errorDiv.textContent = 'Image failed to load'
                          parent.appendChild(errorDiv)
                        }
                      }}
                    />
                  ) : (
                    <div className="w-full h-64 bg-gray-200 rounded-lg flex items-center justify-center">
                      <p className="text-gray-500">No overlay image available</p>
                    </div>
                  )}
                </div>

                {/* Stats */}
                <div className="space-y-6">
                  <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 border-2 border-green-300">
                    <p className="text-gray-600 text-sm mb-2 font-semibold">Camouflage Soldiers</p>
                    <p className="text-5xl font-bold text-green-600">{result.soldier_count}</p>
                  </div>

                  {/* Show civilian count if present */}
                  {result.civilian_count && result.civilian_count > 0 && (
                    <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 border-2 border-gray-300">
                      <p className="text-gray-600 text-sm mb-2 font-semibold">Civilians Detected</p>
                      <p className="text-3xl font-bold text-gray-600">{result.civilian_count}</p>
                      <p className="text-xs text-gray-500 mt-1">Not included in threat assessment</p>
                    </div>
                  )}

                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 border-2 border-blue-300">
                    <p className="text-gray-600 text-sm mb-2 font-semibold">AI Summary</p>
                    <p className="text-sm text-gray-800 leading-relaxed">
                      {result.report?.analysis?.summary || 'Camouflaged soldiers detected in the image.'}
                    </p>
                  </div>

                  {result.report?.location && (
                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-6 border-2 border-purple-300">
                      <p className="text-gray-600 text-sm mb-2 font-semibold">Location</p>
                      <div className="flex items-center gap-2">
                        <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <p className="text-sm font-mono text-gray-700">
                          {result.report.location.latitude.toFixed(4)}, {result.report.location.longitude.toFixed(4)}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* View Report Button */}
                  {result.report && (
                    <button
                      onClick={() => setShowReport(true)}
                      className="w-full py-4 btn-primary shadow-lg text-lg"
                    >
                      View Complete Report & Download PDF
                    </button>
                  )}
                </div>
              </div>

              <button
                onClick={handleReset}
                className="mt-8 px-8 py-3 bg-gray-200 hover:bg-gray-300 rounded-lg font-semibold transition-colors"
              >
                Analyze Another Image
              </button>
            </div>
          </div>
        )}

        {/* Report Modal */}
        {result?.report && (
          <ReportModal
            isOpen={showReport}
            onClose={() => setShowReport(false)}
            report={result.report}
          />
        )}
      </div>
    </div>
  )
}
