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
  
  // Medical parameters state
  const [breastDensity, setBreastDensity] = useState("3")
  const [leftOrRightBreast, setLeftOrRightBreast] = useState('LEFT')
  const [subtlety, setSubtlety] = useState("3")
  const [massMargins, setMassMargins] = useState('SPICULATED')
  const [forceMalignant, setForceMalignant] = useState(false)

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
    formData.append('breast_density', parseInt(breastDensity))
    formData.append('left_or_right_breast', leftOrRightBreast)
    formData.append('subtlety', parseInt(subtlety))
    formData.append('mass_margins', massMargins)
    formData.append('force_malignant', forceMalignant)

    try {
      const response = await fetch('http://localhost:8000/upload-image', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
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

        {/* Medical Parameters Form */}
        <div className="parameters-form">
          <h3>Medical Parameters</h3>
          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="breastDensity">Breast Density:</label>
              <select 
                id="breastDensity"
                value={breastDensity || "3"} 
                onChange={(e) => setBreastDensity(e.target.value)}
              >
                <option value="1">1 - Almost entirely fatty</option>
                <option value="2">2 - Scattered fibroglandular</option>
                <option value="3">3 - Heterogeneously dense</option>
                <option value="4">4 - Extremely dense</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="leftOrRightBreast">Breast Side:</label>
              <select 
                id="leftOrRightBreast"
                value={leftOrRightBreast || "LEFT"} 
                onChange={(e) => setLeftOrRightBreast(e.target.value)}
              >
                <option value="LEFT">Left</option>
                <option value="RIGHT">Right</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="subtlety">Subtlety Rating:</label>
              <select 
                id="subtlety"
                value={subtlety || "3"} 
                onChange={(e) => setSubtlety(e.target.value)}
              >
                <option value="1">1 - Subtle</option>
                <option value="2">2 - Relatively subtle</option>
                <option value="3">3 - Relatively obvious</option>
                <option value="4">4 - Obvious</option>
                <option value="5">5 - Very obvious</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="massMargins">Mass Margins:</label>
              <select 
                id="massMargins"
                value={massMargins || "SPICULATED"} 
                onChange={(e) => setMassMargins(e.target.value)}
              >
                <option value="CIRCUMSCRIBED">Circumscribed</option>
                <option value="ILL_DEFINED">Ill-defined</option>
                <option value="SPICULATED">Spiculated</option>
                <option value="MICROLOBULATED">Microlobulated</option>
                <option value="OBSCURED">Obscured</option>
              </select>
            </div>

            <div className="form-group checkbox-group">
              <label htmlFor="forceMalignant">
                <input 
                  type="checkbox"
                  id="forceMalignant"
                  checked={forceMalignant}
                  onChange={(e) => setForceMalignant(e.target.checked)}
                />
                Force Malignant (Testing Mode)
              </label>
            </div>
          </div>
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
              <div className="classification-summary">
                <div className={`classification-badge ${result.classification.predicted_class.toLowerCase()}`}>
                  {result.classification.predicted_class}
                </div>
                <p><strong>Probability:</strong> {(result.classification.probability * 100).toFixed(1)}%</p>
                <p><strong>Confidence:</strong> {(result.classification.confidence * 100).toFixed(1)}%</p>
                <p><strong>Risk Level:</strong> {result.interpretation.risk_level}</p>
                <p><strong>Recommendation:</strong> {result.interpretation.recommendation}</p>
              </div>

              <div className="detailed-results">
                <h4>Detailed Classification:</h4>
                <div className="probability-bars">
                  <div className="probability-item">
                    <span>Benign:</span>
                    <div className="probability-bar">
                      <div 
                        className="probability-fill benign" 
                        style={{ width: `${result.classification.classes.BENIGN * 100}%` }}
                      ></div>
                      <span className="probability-text">
                        {(result.classification.classes.BENIGN * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="probability-item">
                    <span>Malignant:</span>
                    <div className="probability-bar">
                      <div 
                        className="probability-fill malignant" 
                        style={{ width: `${result.classification.classes.MALIGNANT * 100}%` }}
                      ></div>
                      <span className="probability-text">
                        {(result.classification.classes.MALIGNANT * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="input-parameters">
                <h4>Input Parameters:</h4>
                <div className="parameters-display">
                  <p><strong>Breast Density:</strong> {result.input_parameters.breast_density}</p>
                  <p><strong>Breast Side:</strong> {result.input_parameters.left_or_right_breast}</p>
                  <p><strong>Subtlety:</strong> {result.input_parameters.subtlety}</p>
                  <p><strong>Mass Margins:</strong> {result.input_parameters.mass_margins}</p>
                </div>
              </div>

              {result.result_image && (
                <div className="result-visualization">
                  <h4>Analysis Visualization:</h4>
                  <img 
                    src={result.result_image} 
                    alt="Analysis Results Visualization" 
                    className="result-image"
                  />
                </div>
              )}

              <div className="metadata">
                <p><strong>Analysis Time:</strong> {new Date(result.metadata.timestamp).toLocaleString()}</p>
                <p><strong>Model Version:</strong> {result.metadata.model_version}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  )
}

export default App
