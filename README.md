PortraitCameraPro - example CameraX + ML Kit project

This is a starter Android project (Kotlin) that demonstrates:
- CameraX Preview + ImageCapture
- A "portrait" mode that uses ML Kit Selfie Segmentation to blur the background after capture
- A simple "zoom-enhance" method: when zoom ratio >= 2x the app takes multiple captures and averages them to reduce noise, then applies a simple sharpening pass
- UI that mimics a basic "pro" camera: shutter, mode toggle, zoom slider

Important: CPU-based processing is used here for clarity. Replace with GPU/native code for production.
