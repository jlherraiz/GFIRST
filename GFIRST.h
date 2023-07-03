static const float PI=3.1415926f;

// PARAMETERS OF THE DATA (INVEON SCANNER)-----
static const int NTBINS = 1;       // Number of TOF Bins
static const int NRAD = 147;       // Number of radial bins //Check ODD/EVEN
static const int NANG = 168;       // Number of angles
static const int NROWS = 80;       // Number of "rings"
static const int SPAN = 11;        // Span of the sinogram (11 = Differences 5 and 6)
static const int NSINOGS = 1293;    //837;    // Number of sinograms (127 directos)
static const int NSEGMENTS = 15;   // Number of segments 
static const int NDATA = NSINOGS*NANG*NRAD;  // Total number of bins in the sinogram

// PARAMETERS OF THE RECONSTRUCTED IMAGE -----
static const int RES = 128;             // X-Y resolution
static const int NZS = (2*NROWS-1);     // Z number of slices
static const int NVOXELS=RES*RES*NZS;   // Total number of voxels in the image
static const int NPT = RES;
static const int NZSM = NZS/2 - 1;

// PARAMETERS OF THE RECONSTRUCTION
const int NITER = 60;
const int NSUBSETS = 1;     
const int Nav = 1;      // ?

const int NDATA_PART = NZS*NRAD;  // Number of LORs projected simultaneously
float PSF_FW = 1.5;      //3 PSF for forward  (voxel units) Large = More Resolution
float PSF_BW = 1.5;      //3 PSF for backward (voxel units) Large = Smoother / Slower convergence

float scale_D = 3.70e-4*162.0;   // Calibration Factor for Doubles --> uCi
float scale_T = 3.70e-4*162.0;   // Calibration Factor for Triples --> uCi (124I)

//  PARAMETERS OF THE SCANNER ----------------
__device__ __constant__ float pitch = 1.63f;     // (mm)
__device__ __constant__ float DIAM_DET = 165.7758f;  // Distance between detectors (mm) // 
__device__ __constant__ float FOV = 100.0;       // Field of View (mm)
__device__ __constant__ float TOF_FOV = 100.0;   // Field of View of the TOF (mm)

//---MIN/MAX TYPE INDEPENDENT ---
#ifndef _MINMAX_H
#define _MINMAX_H
#define min(x,y) ({ __typeof(x) xx = (x); \
                    __typeof(y) yy = (y); \
                    xx < yy ? xx : yy; })
#define max(x,y) ({ __typeof(x) xx = (x); \
                    __typeof(y) yy = (y); \
                    xx > yy ? xx : yy; })
#endif /* _MINMAX_H */








