import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Camera, Image, Info, X, Check, Loader, Printer } from 'lucide-react';
import { toast } from 'sonner';
import { cameraService } from '@/services/CameraService';
import { imageProcessingService } from '@/services/ImageProcessingService';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Link } from 'react-router-dom';
import PlantDiseasePredictor from './PlantDiseasePredictor';

interface AnalysisResult {
  leafArea: number;
  greenPixelCount: number;
  redPixelCount: number;
  calibrationArea: number;
  pixelToCmRatio: number;
}

const Home = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isImageDialogOpen, setIsImageDialogOpen] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [analysisProgress, setAnalysisProgress] = useState<number>(0);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [referenceArea, setReferenceArea] = useState<string>('1');
  const [isCalibrated, setIsCalibrated] = useState<boolean>(false);
  const [diseaseResult, setDiseaseResult] = useState<{predicted_class: string, confidence: number} | null>(null);
  const [diseaseLoading, setDiseaseLoading] = useState<boolean>(false);

  const handleCaptureImage = async () => {
    try {
      setIsLoading(true);
      const imageData = await cameraService.captureImage();
      setIsLoading(false);
      
      if (imageData) {
        setSelectedImage(imageData.webPath);
        toast.success("Image captured successfully!");
        setIsImageDialogOpen(true);
      }
    } catch (error) {
      setIsLoading(false);
      toast.error("Failed to capture image. Please try again.");
      console.error("Camera error:", error);
    }
  };

  const handleSelectImage = async () => {
    try {
      setIsLoading(true);
      const imageData = await cameraService.selectImage();
      setIsLoading(false);
      
      if (imageData) {
        setSelectedImage(imageData.webPath);
        toast.success("Image selected successfully!");
        setIsImageDialogOpen(true);
      }
    } catch (error) {
      setIsLoading(false);
      toast.error("Failed to select image. Please try again.");
      console.error("Gallery error:", error);
    }
  };

  const handleCalibrate = async () => {
    if (!selectedImage) return;
    
    try {
      setIsAnalyzing(true);
      setAnalysisProgress(0);
      
      // Simulate progress
      const interval = setInterval(() => {
        setAnalysisProgress(prev => {
          const newProgress = prev + 20;
          if (newProgress >= 100) {
            clearInterval(interval);
            return 100;
          }
          return newProgress;
        });
      }, 200);

      // Set calibration
      await imageProcessingService.setCalibration(
        parseFloat(referenceArea),
        selectedImage
      );
      
      setIsCalibrated(true);
      toast.success("Calibration completed successfully!");
      setIsAnalyzing(false);
    } catch (error) {
      console.error("Calibration error:", error);
      toast.error("Calibration failed. Please try again.");
      setIsAnalyzing(false);
    }
  };

  // Helper to convert dataURL to File (robust)
  const dataURLtoFile = (dataurl: string, filename: string) => {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  // Call FastAPI for disease prediction
  const handleDiseasePrediction = async (imageUrl: string) => {
    setDiseaseLoading(true);
    setDiseaseResult(null);
    try {
      const file = dataURLtoFile(imageUrl, 'leaf.jpg');
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok && !data.error) {
        setDiseaseResult(data);
      } else {
        setDiseaseResult({ predicted_class: 'Error', confidence: 0 });
        console.error('Disease API error:', data.error);
      }
    } catch (err) {
      setDiseaseResult({ predicted_class: 'Error', confidence: 0 });
      console.error('Disease API exception:', err);
    } finally {
      setDiseaseLoading(false);
    }
  };

  const handleAnalyzeLeaf = async () => {
    if (!selectedImage || !isCalibrated) {
      toast.error("Please calibrate first with a reference object.");
      return;
    }
    try {
      setIsAnalyzing(true);
      setAnalysisProgress(0);
      // Simulate progress
      const interval = setInterval(() => {
        setAnalysisProgress(prev => {
          const newProgress = prev + 20;
          if (newProgress >= 100) {
            clearInterval(interval);
            return 100;
          }
          return newProgress;
        });
      }, 200);
      // Measure leaf area
      const result = await imageProcessingService.measureLeafArea(selectedImage);
      setAnalysisResult(result);
      // After leaf area, predict disease
      await handleDiseasePrediction(selectedImage);
      toast.success("Leaf analysis completed successfully!");
      setIsAnalyzing(false);
    } catch (error) {
      console.error("Analysis error:", error);
      toast.error("Analysis failed. Please try again.");
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setAnalysisResult(null);
    setIsImageDialogOpen(false);
    setIsCalibrated(false);
  };

  return (
    <div className="container mx-auto p-4 max-w-md bg-gradient-to-b from-green-50 to-white min-h-screen">
      <header className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-3xl font-bold text-green-800">
            Leaf Area Measurement
          </h1>
          <Link 
            to="/developers" 
            className="text-green-600 hover:text-green-700 text-sm font-medium flex items-center gap-1"
          >
            <span>Meet the Developers</span>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-arrow-right">
              <path d="M5 12h14"/>
              <path d="m12 5 7 7-7 7"/>
            </svg>
          </Link>
        </div>
        <p className="text-center text-green-600 text-sm">
          Precision Agriculture Made Simple
        </p>
      </header>
      
      {selectedImage && (
        <Card className="mb-6 overflow-hidden border-2 border-green-200 shadow-lg">
          <div className="aspect-w-16 aspect-h-9 relative w-full h-56">
            <img 
              src={selectedImage} 
              alt="Selected Leaf" 
              className="absolute inset-0 w-full h-full object-cover"
            />
            {analysisResult && (
              <div className="absolute bottom-0 left-0 right-0 bg-green-700 bg-opacity-80 text-white p-3">
                <div className="font-semibold">Leaf Area: {analysisResult.leafArea} cm²</div>
                <div className="text-sm opacity-90">
                  Green Pixels: {analysisResult.greenPixelCount.toLocaleString()}
                </div>
              </div>
            )}
          </div>
          <CardFooter className="bg-green-100 p-3 flex justify-between">
            <Button 
              variant="outline" 
              size="sm" 
              className="text-green-700 border-green-300"
              onClick={resetAnalysis}
              disabled={isAnalyzing}
            >
              <X className="mr-1 h-4 w-4" />
              Remove
            </Button>
            <Button 
              className="bg-green-700 hover:bg-green-800" 
              size="sm"
              onClick={handleAnalyzeLeaf}
              disabled={isAnalyzing || !isCalibrated}
            >
              {isAnalyzing ? (
                <>
                  <Loader className="mr-1 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : "Analyze Leaf"}
            </Button>
          </CardFooter>
        </Card>
      )}
      
      <Card className="mb-6 bg-white border-2 border-green-200 shadow-lg">
        <CardHeader className="bg-green-100">
          <CardTitle className="text-green-700 text-xl font-bold">Capture Leaf Image</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <p className="text-gray-600 mb-5 text-sm">
            Select a method to analyze a leaf. Make sure to include a red calibration object alongside the leaf.
          </p>
          
          <div className="flex flex-col gap-3">
            <Button 
              className="w-full bg-green-600 hover:bg-green-700 text-lg py-6 rounded-lg shadow-md flex items-center justify-center gap-3"
              onClick={handleCaptureImage}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader className="h-5 w-5 animate-spin" />
              ) : (
                <Camera size={24} />
              )}
              {isLoading ? "Opening Camera..." : "Capture New Image"}
            </Button>
            
            <Button 
              className="w-full bg-green-600 hover:bg-green-700 text-lg py-6 rounded-lg shadow-md flex items-center justify-center gap-3"
              onClick={handleSelectImage}
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader className="h-5 w-5 animate-spin" />
              ) : (
                <Image size={24} />
              )}
              {isLoading ? "Opening Gallery..." : "Select from Gallery"}
            </Button>

            {selectedImage && !isCalibrated && (
              <div className="mt-4 p-4 bg-green-50 rounded-lg">
                <p className="text-sm text-green-800 mb-2">
                  Enter the area of your red reference object (in cm²):
                </p>
                <div className="flex gap-2">
                  <Input
                    type="number"
                    value={referenceArea}
                    onChange={(e) => setReferenceArea(e.target.value)}
                    className="flex-1"
                    min="0.1"
                    step="0.1"
                  />
                  <Button
                    onClick={handleCalibrate}
                    disabled={isAnalyzing}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    {isAnalyzing ? (
                      <Loader className="h-4 w-4 animate-spin" />
                    ) : "Calibrate"}
                  </Button>
                </div>
              </div>
            )}
            
            <Button 
              className="w-full bg-white text-green-700 border-2 border-green-500 hover:bg-green-50 shadow-sm text-lg py-6 rounded-lg flex items-center justify-center gap-3" 
              variant="outline"
              disabled={!analysisResult}
              onClick={() => analysisResult && setIsImageDialogOpen(true)}
            >
              <Info size={24} />
              {analysisResult ? "View Results" : "No Results Yet"}
            </Button>
          </div>
        </CardContent>
      </Card>
      
      <Card className="bg-white border-2 border-green-200 shadow-lg">
        <CardHeader className="bg-green-100">
          <CardTitle className="text-green-700 text-xl font-bold">How it Works</CardTitle>
        </CardHeader>
        <CardContent className="pt-5">
          <ol className="list-decimal pl-5 space-y-3 text-gray-600">
            <li>Capture or select an image containing a leaf and a red calibration object</li>
            <li>Enter the known area of your red calibration object</li>
            <li>Click "Calibrate" to set up the measurement system</li>
            <li>Click "Analyze Leaf" to measure the leaf area</li>
            <li>The leaf area will be calculated and displayed in cm²</li>
          </ol>
        </CardContent>
      </Card>

      <Alert className="mt-8 bg-green-50 border border-green-200">
        <AlertDescription className="text-center text-sm text-green-800">
          Developed by Alok, Sharique, and Arif &copy; 2025
        </AlertDescription>
      </Alert>

      {/* Image Analysis Results Dialog */}
      <Dialog open={isImageDialogOpen} onOpenChange={setIsImageDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{analysisResult ? "Leaf Analysis Results" : "Image Confirmation"}</DialogTitle>
          </DialogHeader>
          
          {selectedImage && (
            <div className="aspect-w-16 aspect-h-9 relative w-full h-48 mb-4">
              <img 
                src={selectedImage} 
                alt="Selected Leaf" 
                className="rounded-md object-cover w-full h-full"
              />
            </div>
          )}
          
          {isAnalyzing && (
            <div className="mb-4">
              <p className="text-gray-700 mb-2 font-medium">Analyzing leaf area...</p>
              <Progress value={analysisProgress} className="h-2" />
              <p className="text-xs text-gray-500 mt-1">{analysisProgress}% complete</p>
            </div>
          )}
          
          {analysisResult && (
            <div className="p-4 bg-green-50 rounded-md mb-4" id="printable-content">
              <div className="flex justify-between items-center mb-3">
                <h3 className="font-bold text-green-800">Analysis Results:</h3>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-green-700 border-green-300"
                  onClick={() => {
                    const printWindow = window.open('', '_blank');
                    if (printWindow) {
                      printWindow.document.write(`
                        <html>
                          <head>
                            <title>Leaf Analysis Results</title>
                            <style>
                              body { font-family: Arial, sans-serif; padding: 20px; }
                              .header { text-align: center; margin-bottom: 20px; }
                              .results { margin-bottom: 20px; }
                              .result-row { display: flex; justify-content: space-between; margin: 5px 0; }
                              .footer { text-align: center; margin-top: 20px; font-size: 12px; color: #666; }
                              @media print {
                                .no-print { display: none; }
                              }
                            </style>
                          </head>
                          <body>
                            <div class="header">
                              <h1>Leaf Analysis Results</h1>
                              <p>Foliage Pixel Probe</p>
                            </div>
                            <div class="results">
                              <div class="result-row">
                                <span>Leaf Area:</span>
                                <strong>${analysisResult.leafArea} cm²</strong>
                              </div>
                              <div class="result-row">
                                <span>Green Pixels:</span>
                                <strong>${analysisResult.greenPixelCount.toLocaleString()}</strong>
                              </div>
                              <div class="result-row">
                                <span>Red Reference Pixels:</span>
                                <strong>${analysisResult.redPixelCount.toLocaleString()}</strong>
                              </div>
                              <div class="result-row">
                                <span>Calibration Area:</span>
                                <strong>${analysisResult.calibrationArea} cm²</strong>
                              </div>
                              <div class="result-row">
                                <span>Pixel to cm² Ratio:</span>
                                <strong>${(analysisResult.pixelToCmRatio * 10000).toFixed(6)} cm²/pixel</strong>
                              </div>
                            </div>
                            <div class="footer">
                              <p>Formula: Leaf Area = Green Pixels × (Calibration Area ÷ Red Pixels)</p>
                              <p>Developed by Alok, Sharique, and Arif &copy; 2025</p>
                            </div>
                            <div class="no-print">
                              <button onclick="window.print()">Print Results</button>
                            </div>
                          </body>
                        </html>
                      `);
                      printWindow.document.close();
                    }
                  }}
                >
                  <Printer className="h-4 w-4 mr-2" />
                  Print Results
                </Button>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Leaf Area:</span>
                  <span className="font-semibold">{analysisResult.leafArea} cm²</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Green Pixels:</span>
                  <span className="font-semibold">{analysisResult.greenPixelCount.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Red Reference Pixels:</span>
                  <span className="font-semibold">{analysisResult.redPixelCount.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Calibration Area:</span>
                  <span className="font-semibold">{analysisResult.calibrationArea} cm²</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Pixel to cm² Ratio:</span>
                  <span className="font-semibold">{(analysisResult.pixelToCmRatio * 10000).toFixed(6)} cm²/pixel</span>
                </div>
              </div>
              <div className="mt-4 pt-3 border-t border-green-200">
                <p className="text-xs text-gray-500">
                  Formula: Leaf Area = Green Pixels × (Calibration Area ÷ Red Pixels)
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  Developed by Alok, Sharique, and Arif &copy; 2025
                </p>
              </div>
            </div>
          )}
          {diseaseResult && (
            <div className="p-4 bg-yellow-50 rounded-md mb-4">
              <h3 className="font-bold text-yellow-800 mb-2">Disease Prediction:</h3>
              {diseaseLoading ? (
                <div className="text-green-700">Predicting disease...</div>
              ) : diseaseResult && (
                diseaseResult.predicted_class === 'Error' ? (
                  <div className="text-red-700">Disease prediction failed. Please try again.</div>
                ) : (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Predicted Disease:</span>
                      <span className="font-semibold">{diseaseResult.predicted_class}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Confidence:</span>
                      <span className="font-semibold">{diseaseResult.confidence ? (diseaseResult.confidence * 100).toFixed(2) : '--'}%</span>
                    </div>
                  </>
                )
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Home;
