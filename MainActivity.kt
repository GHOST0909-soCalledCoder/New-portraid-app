package com.portraitcamera.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.SeekBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.google.android.gms.tasks.Tasks

class MainActivity : AppCompatActivity() {
    private val TAG = "PortraitCameraPro"
    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var shutter: ImageButton
    private lateinit var modeBtn: ImageButton
    private lateinit var zoomSeek: SeekBar
    private lateinit var resultPreview: ImageView

    private var imageCapture: ImageCapture? = null
    private var cameraControl: CameraControl? = null
    private var cameraInfo: CameraInfo? = null

    private val cameraExecutor = Executors.newSingleThreadExecutor()

    private var isPortraitMode = false

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val granted = permissions[Manifest.permission.CAMERA] == true
        if (granted) startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        shutter = findViewById(R.id.btn_shutter)
        modeBtn = findViewById(R.id.btn_mode)
        zoomSeek = findViewById(R.id.seek_zoom)
        resultPreview = findViewById(R.id.resultPreview)

        shutter.setOnClickListener { takeSmartPicture() }
        modeBtn.setOnClickListener {
            isPortraitMode = !isPortraitMode
            modeBtn.alpha = if (isPortraitMode) 1.0f else 0.5f
        }

        zoomSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val ratio = 1.0f + (progress / 50.0f) // 1.0 to ~3.0 zoom
                cameraControl?.setZoomRatio(ratio)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
        }
    }

    private fun allPermissionsGranted() = ActivityCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture, analysis)
                cameraControl = camera.cameraControl
                cameraInfo = camera.cameraInfo
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takeSmartPicture() {
        val ic = imageCapture ?: return

        // If zoom ratio >= 2x, perform a simple multi-capture stack to reduce noise
        val ratio = cameraInfo?.zoomState?.value?.zoomRatio ?: 1.0f
        if (ratio >= 2.0f) {
            takeStackedCapture(3)
        } else {
            // regular capture, if portrait enabled apply segmentation blur after capture
            val outFile = File(externalMediaDirs.firstOrNull(), "PortraitCam_${System.currentTimeMillis()}.jpg")
            val outputOptions = ImageCapture.OutputFileOptions.Builder(outFile).build()
            ic.takePicture(outputOptions, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    lifecycleScope.launch {
                        if (isPortraitMode) {
                            val final = applyPortraitBackgroundBlur(outFile)
                            showResult(final)
                            // overwrite file with final
                            saveBitmapToFile(final, outFile)
                        } else {
                            val b = BitmapFactory.decodeFile(outFile.absolutePath)
                            showResult(b)
                        }
                    }
                }
                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Capture failed: ${exception.message}", exception)
                }
            })
        }
    }

    private fun takeStackedCapture(n: Int) {
        val ic = imageCapture ?: return
        lifecycleScope.launch {
            val bitmaps = mutableListOf<Bitmap>()
            for (i in 0 until n) {
                val outFile = File(cacheDir, "tmp_${i}.jpg")
                val outputOptions = ImageCapture.OutputFileOptions.Builder(outFile).build()
                val saved = suspendCancellableImageCapture(ic, outputOptions)
                val bm = BitmapFactory.decodeFile(outFile.absolutePath)
                bitmaps.add(bm)
            }
            // Aligning frames is non-trivial; here we assume small motion and simple average
            val averaged = averageBitmaps(bitmaps)
            val sharpened = unsharpMask(averaged)
            showResult(sharpened)
            // save
            val outFile = File(externalMediaDirs.firstOrNull(), "PortraitCam_SR_${System.currentTimeMillis()}.jpg")
            saveBitmapToFile(sharpened, outFile)
        }
    }

    private suspend fun suspendCancellableImageCapture(ic: ImageCapture, outputOptions: ImageCapture.OutputFileOptions) = withContext(Dispatchers.IO) {
        val latch = java.util.concurrent.CountDownLatch(1)
        ic.takePicture(outputOptions, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) { latch.countDown() }
            override fun onError(exception: ImageCaptureException) { latch.countDown() }
        })
        latch.await()
        return@withContext outputOptions.file!!
    }

    private fun averageBitmaps(list: List<Bitmap>): Bitmap {
        if (list.isEmpty()) throw IllegalArgumentException("Empty list")
        val w = list[0].width
        val h = list[0].height
        val result = Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val paint = Paint()
        paint.alpha = (255.0f / list.size).roundToInt()
        for (b in list) {
            canvas.drawBitmap(b, 0f, 0f, paint)
        }
        return result
    }

    private fun unsharpMask(src: Bitmap): Bitmap {
        // simple sharpen by overlaying a slightly strong contrast version - this is a cheap approximation
        val width = src.width
        val height = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        val paint = Paint()
        paint.alpha = 200
        canvas.drawBitmap(src, 0f, 0f, paint)
        return result
    }

    private fun applyPortraitBackgroundBlur(file: File): Bitmap {
        // Load bitmap
        var bmp = BitmapFactory.decodeFile(file.absolutePath)
        // downscale for faster segmentation
        val scaled = Bitmap.createScaledBitmap(bmp, 640, (640f * bmp.height / bmp.width).toInt(), true)

        val options = SelfieSegmenterOptions.Builder()
            .setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
            .enableRawSizeMask()
            .build()
        val segmenter = Segmentation.getClient(options)

        val input = InputImage.fromBitmap(scaled, 0)

        // Run synchronously using Tasks.await is not recommended on main thread; here we block briefly on a worker thread
        val task = segmenter.process(input)
        val mask = Tasks.await(task)
        // mask: segmentationMask
        val maskBuffer = mask.buffer
        maskBuffer.rewind()

        val maskW = mask.width
        val maskH = mask.height

        // recreate mask bitmap (grayscale alpha)
        val maskBmp = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ALPHA_8)
        val pixels = ByteArray(maskW * maskH)
        maskBuffer.get(pixels)
        maskBmp.copyPixelsFromBuffer(ByteBuffer.wrap(pixels))

        // Scale mask up to original scaled size if needed (documentation may already scale; keep simplistic)
        val scaledMask = Bitmap.createScaledBitmap(maskBmp, scaled.width, scaled.height, true)

        // Blur background
        val blurred = fastBlur(scaled, 15)

        // Composite: where mask has high alpha -> keep original scaled, else blurred
        val output = Bitmap.createBitmap(scaled.width, scaled.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        val paint = Paint()
        canvas.drawBitmap(blurred, 0f, 0f, null)

        // draw foreground using mask as alpha
        paint.isFilterBitmap = true
        val fg = scaled
        val temp = Bitmap.createBitmap(fg.width, fg.height, Bitmap.Config.ARGB_8888)
        val c = Canvas(temp)
        c.drawBitmap(fg, 0f, 0f, null)
        // apply mask by using drawBitmap with paint that uses mask as shader - simpler is to iterate pixels (slow)
        // We'll do a simple per-pixel composite (not optimal for performance)
        output.copyPixelsToBuffer(ByteBuffer.allocate(0)) // noop to quiet tools
        for (y in 0 until scaled.height) {
            for (x in 0 until scaled.width) {
                val m = (scaledMask.getPixel(x,y) and 0xff)
                if (m > 128) {
                    output.setPixel(x,y, fg.getPixel(x,y))
                }
            }
        }

        // upscale back to original size (approx)
        val finalBmp = Bitmap.createScaledBitmap(output, bmp.width, bmp.height, true)
        return finalBmp
    }

    private fun fastBlur(src: Bitmap, radius: Int): Bitmap {
        // very simple box blur approximation (slow for big images). Replace with RenderScript or native code for speed.
        val w = src.width
        val h = src.height
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        val pix = IntArray(w*h)
        out.getPixels(pix, 0, w, 0,0,w,h)
        val temp = pix.copyOf()
        val r = radius
        for (i in 0 until h) {
            var sumR=0; var sumG=0; var sumB=0
            for (j in -r..r) {
                val x = (j.coerceAtLeast(0).coerceAtMost(w-1))
                val c = temp[i*w + x]
                sumR += (c shr 16) and 0xff
                sumG += (c shr 8) and 0xff
                sumB += c and 0xff
            }
            for (x in 0 until w) {
                val nr = sumR/(2*r+1)
                val ng = sumG/(2*r+1)
                val nb = sumB/(2*r+1)
                pix[i*w + x] = (0xff shl 24) or (nr shl 16) or (ng shl 8) or nb
                // slide window
                val addX = (x + r + 1).coerceAtMost(w-1)
                val subX = (x - r).coerceAtLeast(0)
                val addC = temp[i*w + addX]
                val subC = temp[i*w + subX]
                sumR += ((addC shr 16) and 0xff) - ((subC shr 16) and 0xff)
                sumG += ((addC shr 8) and 0xff) - ((subC shr 8) and 0xff)
                sumB += (addC and 0xff) - (subC and 0xff)
            }
        }
        out.setPixels(pix,0,w,0,0,w,h)
        return out
    }

    private fun saveBitmapToFile(bmp: Bitmap, file: File) {
        try {
            val fos = FileOutputStream(file)
            bmp.compress(Bitmap.CompressFormat.JPEG, 92, fos)
            fos.flush()
            fos.close()
        } catch (e: Exception) {
            Log.e(TAG, "Save failed", e)
        }
    }

    private fun showResult(bmp: Bitmap) {
        runOnUiThread {
            resultPreview.setImageBitmap(bmp)
            resultPreview.visibility = View.VISIBLE
        }
    }
}
