import jsPDF from 'jspdf'

interface Report {
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

export function generatePDF(report: Report) {
  const doc = new jsPDF()
  
  let yPosition = 20

  // Title
  doc.setFontSize(22)
  doc.setTextColor(26, 95, 63) // Primary color
  doc.text('Mirqab Detection Report', 105, yPosition, { align: 'center' })
  
  yPosition += 15

  // Report Info
  doc.setFontSize(10)
  doc.setTextColor(100, 100, 100)
  doc.text(`Report ID: ${report.report_id}`, 20, yPosition)
  yPosition += 6
  doc.text(`Timestamp: ${new Date(report.timestamp).toLocaleString()}`, 20, yPosition)
  yPosition += 6
  
  if (report.location && report.location.latitude !== 0) {
    doc.text(
      `Location: ${report.location.latitude.toFixed(6)}, ${report.location.longitude.toFixed(6)}`,
      20,
      yPosition
    )
    yPosition += 10
  } else {
    yPosition += 4
  }

  // Divider
  doc.setDrawColor(200, 200, 200)
  doc.line(20, yPosition, 190, yPosition)
  yPosition += 10

  // AI Analysis Section
  doc.setFontSize(16)
  doc.setTextColor(0, 0, 0)
  doc.text('AI Analysis', 20, yPosition)
  yPosition += 10

  // Summary
  doc.setFontSize(11)
  doc.setTextColor(50, 50, 50)
  doc.text('Summary:', 20, yPosition)
  yPosition += 6
  doc.setFontSize(10)
  const summaryLines = doc.splitTextToSize(report.analysis.summary, 170)
  doc.text(summaryLines, 20, yPosition)
  yPosition += summaryLines.length * 5 + 5

  // Environment
  doc.setFontSize(11)
  doc.setTextColor(50, 50, 50)
  doc.text('Environment:', 20, yPosition)
  yPosition += 6
  doc.setFontSize(10)
  doc.text(report.analysis.environment, 20, yPosition)
  yPosition += 8

  // Soldier Count
  doc.setFontSize(11)
  doc.setTextColor(50, 50, 50)
  doc.text('Soldier Count:', 20, yPosition)
  yPosition += 6
  doc.setFontSize(10)
  doc.setTextColor(255, 59, 59) // Accent color
  doc.text(report.analysis.soldier_count.toString(), 20, yPosition)
  yPosition += 8

  // Attire
  doc.setFontSize(11)
  doc.setTextColor(50, 50, 50)
  doc.text('Attire & Camouflage:', 20, yPosition)
  yPosition += 6
  doc.setFontSize(10)
  doc.setTextColor(0, 0, 0)
  const attireLines = doc.splitTextToSize(report.analysis.attire_and_camouflage, 170)
  doc.text(attireLines, 20, yPosition)
  yPosition += attireLines.length * 5 + 5

  // Equipment
  doc.setFontSize(11)
  doc.setTextColor(50, 50, 50)
  doc.text('Equipment:', 20, yPosition)
  yPosition += 6
  doc.setFontSize(10)
  doc.setTextColor(0, 0, 0)
  const equipmentLines = doc.splitTextToSize(report.analysis.equipment, 170)
  doc.text(equipmentLines, 20, yPosition)
  yPosition += equipmentLines.length * 5 + 10

  // Check if we need a new page for images
  if (yPosition > 200) {
    doc.addPage()
    yPosition = 20
  }

  // Images Section
  doc.setFontSize(16)
  doc.text('Visual Evidence', 20, yPosition)
  yPosition += 10

  try {
    const imgWidth = 80
    const imgHeight = 60
    let imagesAdded = false
    
    // Original Image
    if (report.images.original_base64 && report.images.original_base64.length > 0) {
      doc.setFontSize(11)
      doc.setTextColor(50, 50, 50)
      doc.text('Original Image:', 20, yPosition)
      yPosition += 5
      
      try {
        // Detect image format from base64 string
        const imageFormat = report.images.original_base64.includes('data:image/png') ? 'PNG' : 'JPEG'
        console.log('Adding original image with format:', imageFormat)
        doc.addImage(report.images.original_base64, imageFormat, 20, yPosition, imgWidth, imgHeight)
        imagesAdded = true
      } catch (imgError) {
        console.error('Error adding original image:', imgError)
        doc.setFontSize(9)
        doc.setTextColor(200, 0, 0)
        doc.text('Original image could not be loaded', 20, yPosition + 30)
      }
    } else {
      doc.setFontSize(11)
      doc.setTextColor(50, 50, 50)
      doc.text('Original Image:', 20, yPosition)
      yPosition += 5
      doc.setFontSize(9)
      doc.setTextColor(150, 150, 150)
      doc.text('(Not available)', 20, yPosition + 30)
    }
    
    // Segmented Image
    if (report.images.masked_base64 && report.images.masked_base64.length > 0) {
      doc.setFontSize(11)
      doc.setTextColor(50, 50, 50)
      doc.text('Segmented Result:', 110, imagesAdded ? yPosition - 5 : yPosition)
      
      try {
        // Detect image format from base64 string
        const imageFormat = report.images.masked_base64.includes('data:image/png') ? 'PNG' : 'JPEG'
        console.log('Adding segmented image with format:', imageFormat)
        doc.addImage(report.images.masked_base64, imageFormat, 110, imagesAdded ? yPosition : yPosition + 5, imgWidth, imgHeight)
        imagesAdded = true
      } catch (imgError) {
        console.error('Error adding segmented image:', imgError)
        doc.setFontSize(9)
        doc.setTextColor(200, 0, 0)
        doc.text('Segmented image could not be loaded', 110, (imagesAdded ? yPosition : yPosition + 5) + 30)
      }
    } else {
      doc.setFontSize(11)
      doc.setTextColor(50, 50, 50)
      doc.text('Segmented Result:', 110, imagesAdded ? yPosition - 5 : yPosition)
      doc.setFontSize(9)
      doc.setTextColor(150, 150, 150)
      doc.text('(Not available)', 110, (imagesAdded ? yPosition : yPosition + 5) + 30)
    }
    
    yPosition += imgHeight + 15
  } catch (error) {
    console.error('Error adding images to PDF:', error)
    doc.setFontSize(10)
    doc.setTextColor(255, 0, 0)
    doc.text('Error: Unable to embed images in PDF', 20, yPosition)
    yPosition += 10
  }

  // Footer
  const pageCount = doc.getNumberOfPages()
  doc.setFontSize(8)
  doc.setTextColor(150, 150, 150)
  
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i)
    doc.text(
      `Page ${i} of ${pageCount} - Mirqab Detection System`,
      105,
      285,
      { align: 'center' }
    )
  }

  // Save PDF
  const filename = `Mirqab_Report_${report.report_id.slice(0, 8)}_${new Date(report.timestamp).toISOString().slice(0, 10)}.pdf`
  doc.save(filename)
}
