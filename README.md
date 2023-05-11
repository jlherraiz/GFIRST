# GFIRST - GPU-based Fast Iterative Reconstruction Software for Tomography #

<h2> Source Code from [Please cite these two references]: </h2>

**Initial Publication [2011]**: GPU-based fast iterative reconstruction of fully 3-D PET sinograms
Joaquin L. Herraiz, Samuel España, Raúl Cabido, Antonio S. Montemayor, Manuel Desco, Juan Jose Vaquero, Jose Manuel Udias.
IEEE Transactions on Nuclear Science, vol. 58, n. 5, october 2011. Pp. 2257-2263. ISSN: 0018-9499. DOI: 10.1109/TNS.2011.2158113
http://ieeexplore.ieee.org/document/5929498/

**Extended to Double and Triple Reconstruction [2023]**:

<h2> Compilation </h2>

 * Adapt the example provided in the compile.sh file to your specific configuration. 
 * GFIRST is implemented in CUDA, so it is compiled with the NVCC compiler:
 * **nvcc [options] GFIRST.cu -o GFIRST.x**

<h2> Execution </h2>
GFIRST performs different types of reconstructions depending on the number of input parameters:

1) **./GFIRST image_volume.raw**  ➡ Performs the forward projection of the volume creating a sinogram
2) **./GFIRST sinogram.raw normalization.raw**  ➡ Performs the reconstruction of the sinogram with the appropriate normalization.
3) **./GFIRST sinogramD.raw sinogramT.raw normalization.raw**  ➡ Performs the reconstruction of Doubles and Triple Coincidences based on sinograms.
4) **./GFIRST sinogramD.raw VLORs.raw normalization.raw 0** ➡  Performs the reconstruction of Doubles (with sinograms) and Triples (with VLORs).

<br>
Examples that reconstruct simulated data of the Inveon scanner are provided in the folder "examples"
</br>

<h2> List of publications that have used GFIRST </h2>
 * 
