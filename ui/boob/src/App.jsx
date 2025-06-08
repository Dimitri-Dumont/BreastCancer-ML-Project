import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      if (file.type.startsWith('image/')) {
        setSelectedFile(file)
        setError(null)
      } else {
        setError('Please select an image file')
      }
    }
  }

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file)
      setError(null)
    } else {
      setError('Please select an image file')
    }
  }

  const uploadImage = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await fetch('http://localhost:8000/upload-image', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(`Upload failed: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const resetUpload = () => {
    setSelectedFile(null)
    setResult(null)
    setError(null)
  }

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Breast Cancer MRI Analysis</h1>
      
      <div className="upload-container">
        <div 
          className={`dropzone ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <p>Drag & drop an MRI image here, or click to select</p>
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            id="fileInput"
          />
          <label htmlFor="fileInput" className="file-select-button">
            Select Image
          </label>
        </div>

        {selectedFile && (
          <div className="file-info">
            <p><strong>Selected:</strong> {selectedFile.name}</p>
            <div className="button-group">
              <button onClick={uploadImage} disabled={loading}>
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </button>
              <button onClick={resetUpload} disabled={loading}>
                Clear
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="error">
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="results">
            <h3>Analysis Results</h3>
            <div className="result-content">
              <p><strong>Prediction:</strong> {result.classification.predicted_class}</p>
              <p><strong>Confidence:</strong> {(result.classification.confidence * 100).toFixed(1)}%</p>
              
              {result.processed_image && (
                <div className="images">
                  <h4>Processed Image with Detection:</h4>
                  <img 
                    src={`data:image/jpeg;base64,${result.processed_image}`} 
                    alt="Processed MRI" 
                    style={{ maxWidth: '400px', height: 'auto' }}
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  )
}

export default App
