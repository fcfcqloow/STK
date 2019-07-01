import java.io.IOException
import java.net.URISyntaxException

import org.bytedeco.opencv.opencv_core.*
import org.bytedeco.opencv.opencv_face.*
import org.bytedeco.opencv.opencv_highgui.*
import org.bytedeco.opencv.opencv_imgproc.*
import org.bytedeco.opencv.opencv_objdetect.*
import org.bytedeco.opencv.global.opencv_core.*
import org.bytedeco.opencv.global.opencv_face.*
import org.bytedeco.opencv.global.opencv_highgui.*
import org.bytedeco.opencv.global.opencv_imgcodecs.*
import org.bytedeco.opencv.global.opencv_imgproc.*
import org.bytedeco.opencv.global.opencv_objdetect.*

class MyFace {
    constructor(){
        // Load Face Detector

    }
    fun init() {
        val faceDetector = CascadeClassifier(this.javaClass.getResource("haarcascade_frontalface_alt.xml").path)
        val facemark = FacemarkKazemi.create()
        facemark.loadModel(this.javaClass.getResource("face_landmark_model.dat").path)
        val img = imread(this.javaClass.getResource("def.jpg").path)


        val gray = Mat()
        cvtColor(img, gray, COLOR_BGR2GRAY)
        equalizeHist(gray, gray)


        val faces = RectVector()
        faceDetector.detectMultiScale(gray, faces)

        println("Faces detected: " + faces.size())
        val landmarks = Point2fVectorVector()
        val success = facemark.fit(img, faces, landmarks)

        if (success) {
            for (i in 0 until landmarks.size()) {
                val v = landmarks.get(i)
                drawFacemarks(img, v, Scalar.YELLOW)
            }
        }
        imshow("Kazemi Facial Landmark", img)
        cvWaitKey(0)
        imwrite("kazemi_landmarks.jpg", img)
        imwrite("kazemi_landmarks.jpg", img)
    }
}