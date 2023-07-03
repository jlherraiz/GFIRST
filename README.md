# GFIRST - GPU-based Fast Iterative Reconstruction Software for Tomography #

<h2> Source Code from [Please cite these two references]: </h2>

**Initial Publication [2011]**: GPU-based fast iterative reconstruction of fully 3-D PET sinograms
Joaquin L. Herraiz, Samuel España, Raúl Cabido, Antonio S. Montemayor, Manuel Desco, Juan Jose Vaquero, Jose Manuel Udias.
IEEE Transactions on Nuclear Science, vol. 58, n. 5, october 2011. Pp. 2257-2263. ISSN: 0018-9499. DOI: 10.1109/TNS.2011.2158113
http://ieeexplore.ieee.org/document/5929498/

**Extended to Double and Triple Reconstruction [2023]**: Simultaneous quantitative imaging of two PET radiotracers via the detection of positron–electron annihilation and prompt gamma emissions - Edwin C. Pratt et al. - Nature Biomedical Engineering, 2023. DOI: 10.1038/s41551-023-01060-y
https://www.nature.com/articles/s41551-023-01060-y

<h2> Compilation </h2>

 * Adapt the example provided in the compile.sh file to your specific configuration. 
 * GFIRST is implemented in CUDA, so it is compiled with the NVCC compiler:
 * **nvcc [options] GFIRST.cu -o GFIRST.x**

<h2> Execution </h2>
GFIRST performs different types of reconstructions depending on the number of input parameters:

1) **./GFIRST.x image_volume.raw**  ➡ Performs the forward projection of the volume creating a sinogram
2) **./GFIRST.x sinogram.raw normalization.raw**  ➡ Performs the reconstruction of the sinogram with the appropriate normalization.
3) **./GFIRST.x sinogramD.raw sinogramT.raw normalization.raw**  ➡ Performs the reconstruction of Doubles and Triple Coincidences based on sinograms.
4) **./GFIRST.x sinogramD.raw sinogramT.raw normalization.raw VLORs.raw** ➡  Performs the reconstruction of Doubles (with sinograms) and Triples (with VLORs).

<h2> Example </h2>
A full example that reconstructs a mouse-like phantom data acquired with 124I and 89Zr with the preclinical Inveon PET/CT scanner can be found here:
<br>
https://colab.research.google.com/drive/1utA6qi9AxpTAIByHb4iIFFe87kM2z2Mw?usp=sharing
</br>
