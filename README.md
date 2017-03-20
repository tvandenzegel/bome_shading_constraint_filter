# Shading constraint improves accuracy of time-of-flight measurements
A naïve implementation of the "Shading constraint improves accuracy of time-of-flight measurements" paper by Martin Böhme. [1] [2]  
The goal of this project was just to gain some practical experience of this method.  
The original code used the non-linear conjugate gradient solver from the book "Numerical Recipes". In order to prevent any possible copyright infringements I quickly modified this version so that it uses the non-linear conjugate gradient method of OpenCV. It seems to be a lot slower than the original code...

More details about this implementation can be found at http://tvdz.be/2015/06/improving-time-of-flight-measurements-with-the-shading-constraint/.

<p align="center">
<img src="data/shading_constraint_opencv_example.png" width="500"> 
</p>

## Dependencies
The following dependencies are needed.  
The version numbers are the ones used during development.  
- OpenCV 3.2.0 [http://opencv.org/](http://opencv.org/)
- LibTiff 4.0.7 [http://www.simplesystems.org/libtiff/](http://www.simplesystems.org/libtiff/)

The code was developed on a Windows machine with Visual Studio 2015.  

## References
[1] Böhme, Martin, et al. “Shading constraint improves accuracy of time-of-flight measurements.” Computer Vision and Pattern Recognition Workshops, 2008. CVPRW’08. IEEE Computer Society Conference on. IEEE, 2008.

[2] Böhme, Martin, et al. “Shading constraint improves accuracy of time-of-flight measurements.” Computer vision and image understanding 114.12 (2010): 1329-1335.
