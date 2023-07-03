//---------------------------------------------------------------------------
// GFIRST - GPU-based Fast Iterative Reconstruction SofTware
// Joaquin L. Herraiz et al. 
// Nuclear Physics Group and IPARCOS - Complutense University of Madrid (UCM)
// Code available at Github: https://github.com/jlherraiz/GFIRST
// This version of the code allows the reconstruction of Doubles and Triples
//   from the preclinical Inveon scanner (See Reference 2)
//
// Main References:
//  [1] GPU-based fast iterative reconstruction of fully 3-D PET sinograms 
//      Joaquin L. Herraiz et al.
//      IEEE Transactions on Nuclear Science, vol. 58, n. 5, october 2011
//      Pp. 2257-2263. ISSN: 0018-9499. DOI: 10.1109/TNS.2011.2158113 
//      http://ieeexplore.ieee.org/document/5929498/ 
//  [2] Simultaneous quantitative imaging of two PET radiotracers 
//      via the detection of positron–electron annihilation 
//      and prompt gamma emissions - Edwin C. Pratt et al. 
//      Nature Biomedical Engineering, 2023
//      DOI: 10.1038/s41551-023-01060-y
//      https://www.nature.com/articles/s41551-023-01060-y
//
// recGPU.cu --> MAIN FILE
//---------------------------------------------------------------------------

#include <stdlib.h>
#include <stdlib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string>
#include <cuda_runtime.h>

//--- CONFIGURATION FILE ---------------
#include <GFIRST.h>

//--- CONFIGURATION FILE ---------------
#include <GFIRST_CPU_Kernels.cpp>

//--- FILE WITH GPU KERNELS ------------
#include <GFIRST_GPU_Kernels.cu>

void osem3d(char* filename, char* filename2, char* filenameCal, char* filenameV, float ft,int par){
	
  // ----------------------------------------------------------------
  // [STEP 0 ] ------ INITIAL SETUP ---------------------------------
  // ----------------------------------------------------------------
 
  //TIME CPU -->	
	time_t start_total,end_total;	
	time (&start_total);
	double dif;
	
	char* name;
	
	crearTexturas();	
	int iTOF = 0;
	
  //-------------- ALLOCATION OF MEMORY SPACE GPU---------------------	
	// GPU -->
	float* d_PROJ;
	cudaMalloc((void**)&d_PROJ,NDATA_PART*sizeof(float));
	float* d_CALPROJ;
	cudaMalloc((void**)&d_CALPROJ,NDATA_PART*sizeof(float));	
	float* d_NUMER;
	cudaMalloc((void**)&d_NUMER,NVOXELS*sizeof(float));	
	float* d_DENOM;
	cudaMalloc((void**)&d_DENOM,NVOXELS*sizeof(float));
	float* d_CORR;
	cudaMalloc((void**)&d_CORR,NVOXELS*sizeof(float));
	float* d_NUMERF;
	cudaMalloc((void**)&d_NUMERF,NVOXELS*sizeof(float));	
	float* d_DENOMF;
	cudaMalloc((void**)&d_DENOMF,NVOXELS*sizeof(float));
	float* d_SINOG;
	cudaMalloc((void**)&d_SINOG,NDATA*sizeof(float));
	float* d_INPUT;
	cudaMalloc((void**)&d_INPUT,NVOXELS*sizeof(float));	
	float* d_REF;
	cudaMalloc((void**)&d_REF,NVOXELS*sizeof(float));	
	float* d_OUTPUT;
	cudaMalloc((void**)&d_OUTPUT,NVOXELS*sizeof(float));		
	
//----------------ALLOCATION OF MEMORY SPACE CPU--------------	
	float* PROJ;		
	PROJ = (float*)malloc(NDATA_PART*sizeof(float));
	float* CORR_PART;		
	CORR_PART = (float*)malloc(NDATA_PART*sizeof(float));
	float* CALIB_PART;		
	CALIB_PART = (float*)malloc(NDATA_PART*sizeof(float));
	float* CORR;		
	CORR = (float*)malloc(NDATA*sizeof(float));
	float* SINOG_PROJ;		
	SINOG_PROJ = (float*)malloc(NDATA*sizeof(float));		
	float* SINOG_PROJ_SCALE;		
	SINOG_PROJ_SCALE = (float*)malloc(NDATA*sizeof(float));		
	float* PROJ_BLANK;		
	PROJ_BLANK = (float*)malloc(NDATA*sizeof(float));				
	float* imgFactores;
	imgFactores = (float*)malloc(NVOXELS*sizeof(float));
	float* imgFactores2;
	imgFactores2 = (float*)malloc(NVOXELS*sizeof(float));
	float* FACTOR;
	FACTOR = (float*)malloc(NVOXELS*sizeof(float));
	float* imgEst;
	float* imgEst2;
	float* imgEst2f;
	float* imgEst_D;	
	float* imgEst_T;		
	
	float* datos_d;
	float* datos_t;
	float* datos_v;
	float* datos_v0;
	float* BLANK;
        			
	datos_v = (float*)malloc(NDATA*sizeof(float));	
	datos_v0 = (float*)malloc(NDATA*sizeof(float));	

	float* BLANKT;
	BLANKT = (float*)malloc(NDATA*sizeof(float));

//--------------------------------------------
	chequearError("Error allocating memmory");

	CleanMemory(d_NUMER,NVOXELS);
	CleanMemory(d_DENOM,NVOXELS);

	printf("Starting............ \n");
	
  // ----------------------- OBLIQUE SINOGRAMS ----------------------
	int nz_acum[NSEGMENTS+1];  
	int zmin,zmax,idz;
	nz_acum[0]=0; 
	for (int idz_ind=0;idz_ind<NSEGMENTS;idz_ind++){	  // 7 SEGMENTS = 0,1,-1,2,-2,3,-3 --> 3 INCLINATIONS & 2 SIGNS	 
	 idz = (idz_ind+1)/2;	 
	 if (idz_ind==0){zmin=0;}else{zmin = (abs(idz)-1)*SPAN+(SPAN/2+1);}		
	 nz_acum[idz_ind+1]= nz_acum[idz_ind] + (NZS - 2*zmin) ;
	}	
	
	int ilor=0;		
	int lor=0;
	float calib;

  // ----------------------------------------------------------------
  // [STEP 1 ] ------ INITIAL PROJECTION  ---------------------------
  // ----------------------------------------------------------------

	// ------------- CREATING PROJECTION OF CYLINDER [OPTIONAL] ------------------	
	float val = 1.0f;
	if (par==0) {
 	  printf("Reading Input Image for Projection ... \n");
    imgEst = ReadImage(filename);
   }else{
	  printf("Creating a Uniform Cylinder (FOV size) for Normalization ... \n");
	  imgEst = CreateInitialCylinder(val);	
	}
	UpdateImage(imgEst);  // --> Send to GPU
  chequearError("Error updating the Image in GPU memmory");
	
	for (int iTOF=0;iTOF<NTBINS;iTOF++){   // In case there is TOF information
	for (int idz_ind=0;idz_ind<NSEGMENTS;idz_ind++){	  // 7 SEGMENTS = 0,1,-1,2,-2 --> 3 INCLINATIONS & 2 SIGNS
   int isigno = 1 - (idz_ind%2)*2;
	 int idz = ((idz_ind+1)/2)*isigno;	
	 if (idz==0){zmin=0;}else{zmin = (abs(idz)-1)*SPAN+(SPAN/2+1);}
	 int zmax = NZS - zmin;	 
	 int nza = nz_acum[idz_ind]; 
 	 for (int itheta=0;itheta<NANG;itheta++){	    
	  FORWPROJ(d_PROJ,itheta,idz,iTOF); 
	  chequearError("Error projecting initial cylinder");
	  cudaMemcpy(PROJ,d_PROJ,NDATA_PART*sizeof(float),cudaMemcpyDeviceToHost);
	 for (int iz0=zmin;iz0<zmax;iz0++){
   for (int ird=0;ird<NRAD;ird++){	
	  ilor = iz0*NRAD + ird;	 	  
	  lor = (nza+(iz0-zmin))*(NRAD*NANG)+itheta*NRAD+ird;
	  PROJ_BLANK[lor] = PROJ[ilor]; //*BLANK[lor];	
   } // R
   } // Z
	} // THETA
	} // INCLINATION		
	} // TOF
	name="proj_ref.raw";
 	WriteImage(PROJ_BLANK,NDATA,name);	
	
  if (par==0){   // PROJECTION OF IMAGE
	  printf("Finish After Projection \n");
	  return;
  }

  // ----------------------------------------------------------------
  // [STEP 2 ] ------ DATA LOADING  ---------------------------------
  // ----------------------------------------------------------------

	int NTYPES;			 
  printf("\n");
	if (par==1){    //VLORS
	  printf("Mode = 1 --> Reconstruction of Doubles Sinogram \n");
	} else if(par==2){
    printf("Mode = 2 --> Reconstruction of Doubles and Triples Sinogram \n");
  } else if (par==3){
    printf("Mode = 3 --> Reconstruction of Doubles and Triples VLORs\n");
  }
  printf("\n");
	printf("Loading Data \n");

	//----------------- LOAD BLANK CALIBRATION ------------------------	
	int tipod = 1;  // 0=INT, 1=FLOAT, 2=ONES    
	BLANK = ReadSinogram_Type(filenameCal,tipod);	
	printf("Normalization Sinogram loaded \n");

	//----------------- LOAD DATA FILES ------------------------
  const int SVLOR = 2;
	int VLOR[SVLOR];
	float s1,s2,s,s_inv;
	int i1,i2;		

  FILE * ficheroVLOR;
  FILE* ficheroC;  //output file

	if (par==2 || par==3){   // READING DOUBLES + TRIPLES
	 NTYPES=2;           // Number of different isotopes in FOV		
	 datos_d = ReadSinogram_Type(filename,tipod);	
	 for(int i=0;i<2*NDATA;i++){datos_d[i]=RES*datos_d[i];}		 	 
	 datos_t = ReadSinogram_Type(filename2,tipod);	     
	 for(int i=0;i<2*NDATA;i++){datos_t[i]=RES*datos_t[i];}     
  }

	if (par==3){	      // READING VLORS
    printf("Reading VLOR file \n");
    for(int i=0;i<NDATA;i++){datos_v0[i]=0.;}	
    ficheroVLOR=fopen(filenameV,"rb");		 	 	 
	  while (!feof(ficheroVLOR)) {   
	   fread(VLOR,sizeof(int),SVLOR,ficheroVLOR);	
	   i1 = VLOR[0]-1;
	   i2 = VLOR[1]-1;	  	  
	   if (i1<=0 || i2 <= 0 || i1>=NDATA || i2>=NDATA) { continue; }
	   datos_v0[i1]++;   
	   datos_v0[i2]++;
	  }
    fclose(ficheroVLOR);
	
    // -- CALIBRATION (TRIPLES VS VLORS)  TO "EXTRACT" THE CALIBRATION FROM THE TRIPLES SINOGRAM
    for (int i=0;i<NDATA;i++){ 
      val = datos_v0[i]; 
      if (val>0.01) {BLANKT[i]= datos_t[i]/float(RES*val);} else {BLANKT[i]=0.;}  
    }

    // For debug purposes -----
	  //ficheroC=fopen("datos_v0.raw","wb");
	  //fwrite(datos_v0,sizeof(float),NDATA,ficheroC);
	  //fclose(ficheroC);
	  //ficheroC=fopen("BLANKT.raw","wb");
	  //fwrite(BLANKT,sizeof(float),NDATA,ficheroC);
	  //fclose(ficheroC);
 
	 }else if (par==1){   // DOUBLES ONLY
	  NTYPES=1;
	  datos_d = ReadSinogram_Type(filename,tipod);	
	  for(int i=0;i<2*NDATA;i++){datos_d[i]=RES*datos_d[i];}
   }else if (par==0){   // PROJECTION OF IMAGE
	  printf("Finish After Projection \n");
	  return; 	  			 	 	  
	 }

  // ----------- BASIC STATISTICS OF THE DATASETS ----------------	
	float sumd=0.;
	float sumdbg=0.;
  for(int i=0;i<NDATA;i++){sumd+=datos_d[i]; sumdbg+=datos_d[NDATA+i];}	
	printf("  Doubles Coincidences: Prompts= %.1f Million, Background= %.1f Million\n",(sumd/RES)/1.0e6, (sumdbg/RES)/1.0e6);

	if (par>1){
	 float sumt=0.;
	 float sumtbg=0.;
   for(int i=0;i<NDATA;i++){sumt+=datos_t[i]; sumtbg+=datos_t[NDATA+i];}	
	 printf("  Triples Coincidences: Prompts = %.1f Million, Background= %.1f Million\n",(sumt/RES)/1.0e6,(sumtbg/RES)/1.0e6);
	}

  // ----------------------------------------------------------------
  // [STEP 3 ] ------ INITIAL IMAGE ---------------------------------
  // ----------------------------------------------------------------
	val = 1.0f;
	imgEst = CreateInitialCylinder(val);	
	imgEst_D = CreateInitialCylinder(val);	
	if (par==1) {val=0.;}  // Par == 1 --> No Triples Reconstruction
	imgEst2 = CreateInitialCylinder(val);	
	imgEst2f = CreateInitialCylinder(val);
	imgEst_T = CreateInitialCylinder(val);	
		
  // ----------------------------------------------------------------
  // [STEP 4 ] -------------- ITERATIONS ----------------------------
  // ----------------------------------------------------------------
  chequearError("Error before starting iterations");
  printf("\n");
	printf("Starting Iterations \n");
	
	srand( time( NULL ) );
	iTOF = 0;
  int isigno = 0;
	int nza=0;
  FILE* ficheroV;
	
  for (int iter=1;iter<=NITER;iter++){        // ITERATIONS LOOP
   printf("ITER= %d  \n",iter);	
	 for (int itype=0;itype<NTYPES;itype++){    // 2 LOOPS FOR DOUBLE/TRIPLE

   if (itype==0) {
    printf("  Doubles Reconstruction \n");	
   } else {
    printf("  Triples Reconstruction \n");	
   }
   
	 float sum_b = 0.; 
   float sum_p = 0.; 
	 float sum_d = 0.;
	 for(int i=0;i<NDATA;i++){ SINOG_PROJ[i]=0.;}
	 for(int i=0;i<NDATA;i++){ CORR[i]=0.;}
	 time (&start_total);   // TIME COUNTER FOR EACH SUBSET  

  // ----------------------------------------------------------------
  // [STEP 4A ] ------ CONVOLUTION WITH PSF BEFORE PROJECTION -------
  // ----------------------------------------------------------------

	 if (itype==0) {  // DOUBLES
 	  for(int i=0;i<NVOXELS;i++){ imgEst_D[i]=imgEst[i];}		   
	   // ------------ PSF CONVOLUTION BEFORE PROJECTION  ----------
	   calcGaussianCoefficients(PSF_FW);    // GAUSSIAN FILTER
	   cudaMemcpy(d_NUMER,imgEst_D,NVOXELS*sizeof(float),cudaMemcpyHostToDevice);	
	   convolutionXY(d_NUMER,d_CORR); 
	   cudaMemcpy(imgEst_D,d_CORR,NVOXELS*sizeof(float),cudaMemcpyDeviceToHost);	  
	   UpdateImage(imgEst_D);		             
	 }else{	         // TRIPLES
	  for(int i=0;i<NVOXELS;i++){ imgEst_T[i]= imgEst2[i];}  
	  // ------------  PSF CONVOLUTION BEFORE PROJECTION ----------
	  calcGaussianCoefficients(1.0*PSF_FW);  // GAUSSIAN FILTER
	  cudaMemcpy(d_NUMER,imgEst_T,NVOXELS*sizeof(float),cudaMemcpyHostToDevice);	
	  convolutionXY(d_NUMER,d_CORR); 
	  cudaMemcpy(imgEst_T,d_CORR,NVOXELS*sizeof(float),cudaMemcpyDeviceToHost);    
	  UpdateImage(imgEst_T);	 	  
	 }
	  
  // ----------------------------------------------------------------
  // [STEP 4B ] -------------- PROJECTION ---------------------------
  // ----------------------------------------------------------------

	 for (int idz_ind=0;idz_ind<NSEGMENTS;idz_ind++){	  
	  isigno = 1 - (idz_ind%2)*2;
	  idz = ((idz_ind+1)/2)*isigno;
	 for (int itheta=0;itheta<NANG;itheta++){
	  FORWPROJ(d_PROJ,itheta,idz,iTOF); 	 
	  cudaMemcpy(PROJ,d_PROJ,NDATA_PART*sizeof(float),cudaMemcpyDeviceToHost);
	  chequearError("Error during projection");	  
	  // Storing projection into 3D sinogram
 	  if (idz==0){zmin=0;}else{zmin = (abs(idz)-1)*SPAN+(SPAN/2+1);}		 
	  zmax = NZS - zmin;
	  nza = nz_acum[idz_ind]; 	 	 
	  for (int iz0=zmin;iz0<zmax;iz0++){
     for (int ird=0;ird<NRAD;ird++){	
	    ilor = iz0*NRAD + ird;	 	  
	    lor = ((iz0-zmin)+nza)*(NRAD*NANG)+itheta*NRAD+ird;	
	    SINOG_PROJ[lor] = PROJ[ilor];	
     }
	  }
	 }   // Loop Angles
	 }   // Loop Sinogram Segments
	 
	//name="proj.raw";   
 	//WriteImage(SINOG_PROJ,NDATA,name);	
		
	//----------------------------------------------------------
  //----------------- SUBDIVIDING VLORS ----------------------  	  	 
	
  if (par==3 & itype==1) {  // VLOR TRIPLES RECONST
	for(int i=0;i<NDATA;i++){datos_v[i]=0.;}	
	 ficheroV=fopen(filenameV,"rb");		 	 	 
	 while (!feof(ficheroV)) {   
	  fread(VLOR,sizeof(int),SVLOR,ficheroV);	
	  i1 = VLOR[0]-1;
	  i2 = VLOR[1]-1;	  	  
	  if (i1<=0 || i2 <= 0 || i1>=NDATA || i2>=NDATA) {  continue; }
    float valor = BLANKT[i1]+BLANKT[i2];
	  s1 = SINOG_PROJ[i1];    // Weights are obtained from the reference sinogram
    s2 = SINOG_PROJ[i2];  
	  s = s1 + s2;
	  if (s>0.01f) {s_inv=1.0/s;} else {s_inv=0.f;}
	  datos_v[i1]+=s1*s_inv*valor;   
	  datos_v[i2]+=s2*s_inv*valor;
	 }
  fclose(ficheroV);
	
	// WRITING VLOR BEFORE CALIBRATION
	 ficheroC=fopen("SINOG_VLOR.raw","wb");
	 fwrite(datos_v,sizeof(float),NDATA,ficheroC);
	 fclose(ficheroC);
  			
	// --- SCALING VLORS (SIMILAR TO DOUBLES AND TRIPLES)---
   for(int i=0;i<NDATA;i++){ datos_v[i]*=RES;}
	 float sumv=0.;
	 float sumvbg=0.;
	 float descal;
   for(int i=0;i<NDATA;i++){
      descal = 1.0/BLANKT[i];
	  if (BLANKT[i]<0.1) {descal = 0.;}
      sumv+= datos_v[i]*descal; 
      sumvbg+=datos_t[NDATA+i]*descal;
   }	
   //printf("    Triples VLORs: Prompts = %.0f Million, Background= %.0f Million\n",sumv/1.0e6,sumvbg/1.0e6);  	 	 
	}  // VLOR RECONST

  // ----------------------------------------------------------------
  // [STEP 4C ] ------ CORRECTIONS FROM DATA vs PROJECTIONS --------- 
  // ----------------------------------------------------------------
	float val_proy,val_dato, val_bg;
  sum_d = 0.; sum_p = 0.; sum_b= 0.;
	for(int i=0;i<NDATA;i++){	  	         
	  val_proy = SINOG_PROJ[i];
	  if (itype==0){   // Doubles
	    val_dato = datos_d[i];             val_bg = datos_d[NDATA+i];
	  }else{          // Triples or  VLOR
      if (par==2){val_dato = datos_t[i]; val_bg = datos_t[NDATA+i];}    // Triples 
      if (par==3){val_dato = datos_v[i]; val_bg = datos_t[NDATA+i];}    // VLOR 
	  }
	  if (val_bg<0.) val_bg=0.;
    sum_d+=val_dato;   sum_b+=val_bg;   sum_p+=val_proy; 
	  if (val_proy+val_bg>=0.01f) {	  
	   CORR[i]=val_dato/(val_proy+val_bg);	   
	  } else{
	   CORR[i]=-1000.;
	  }
	 }	 
	  
  // ----------------------------------------------------------------
  // [STEP 4D ] ------ BACKPROJECTION ------------------------------- 
  // ----------------------------------------------------------------
 
	  CleanMemory(d_NUMER,NVOXELS);
	  CleanMemory(d_DENOM,NVOXELS);
	 
	  for (int idz_ind=0;idz_ind<NSEGMENTS;idz_ind++){
	   isigno = 1 - (idz_ind%2)*2;
	   idz = ((idz_ind+1)/2)*isigno;
	   if (idz==0){zmin=0;}else{zmin = (abs(idz)-1)*SPAN+(SPAN/2+1);}		 
	   zmax = NZS - zmin;
	   nza = nz_acum[idz_ind]; 
	  for (int itheta=0;itheta<NANG;itheta++){	 	   
	   // Recovering corrections in each projection 
	   for (int i=0;i<NDATA_PART;i++){CORR_PART[i]=-1000.; CALIB_PART[i]=-1000.;}
	   for (int iz0=zmin;iz0<zmax;iz0++){
      for (int ird=0;ird<NRAD;ird++){	
	     ilor = iz0*NRAD + ird;	 	  
	     lor = ((iz0-zmin)+nza)*(NRAD*NANG)+itheta*NRAD+ird;	
	     CORR_PART[ilor] = CORR[lor];	
	     if (PROJ_BLANK[lor]>0.01 && BLANK[lor]>0.01) {
        calib = 1.0;  //BLANK[lor]/PROJ_BLANK[lor];
       }else{
        calib=-1000.;
       }
	     CALIB_PART[ilor] = calib;
      } // rho
	   } // z	 
		
	   actualizarCorr(CORR_PART);  	   
	   actualizarCalib(CALIB_PART);  	   
   
	   BACKWPROJ(d_NUMER,d_DENOM,itheta,idz,iTOF);
	   chequearError("Error in Backprojection");		   
		
 	  } // Angle
	 }  // Segments	  	 
	
	cudaMemcpy(imgFactores,d_NUMER,NVOXELS*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(imgFactores,d_DENOM,NVOXELS*sizeof(float),cudaMemcpyDeviceToHost);
  
  // For debug purposes
  //name="img_SENS.raw";
 	//WriteImage(imgFactores,NVOXELS,name);
	//printf("    Measured Prompts = %.1f Million ; Estimated Prompts = %.1f Million \n",(sum_d/RES)/1.0e6,((sum_p+sum_b)/RES)/1.0e6);
  
  // ----------------------------------------------------------------
  // [STEP 4E ] ------ IMAGE UPDATE ------------------------------- 
  // ----------------------------------------------------------------
	dividirVxVGPU(d_NUMER,d_DENOM,NVOXELS,d_CORR);
  calcGaussianCoefficients(PSF_BW);  
	convolutionXY(d_CORR,d_NUMERF); // FILTERING BACW en xy
	chequearError("Error in Ratio computation");	
	
  cudaMemcpy(FACTOR,d_NUMERF,NVOXELS*sizeof(float),cudaMemcpyDeviceToHost);
  gaussian_filterZ(FACTOR,imgFactores);

	// Doubles and Triples Update
 	 if (itype==0) {	//DOUBLES
    for(int i=0;i<NVOXELS;i++){ imgEst[i]*=imgFactores[i];}
    for (int i=0;i<RES*RES;i++){	
  	  imgEst[i] = imgEst[i+RES*RES];	 	  
  	  imgEst[i+(NZS-1)*RES*RES] = imgEst[i+(NZS-2)*RES*RES];	 	
    }  
   } else {         //TRIPLES
	  for(int i=0;i<NVOXELS;i++){ imgEst2[i]*=imgFactores[i];}	
    for (int i=0;i<RES*RES;i++){	
 	   imgEst2[i] = imgEst2[i+RES*RES];	 	  
 	   imgEst2[i+(NZS-1)*RES*RES] = imgEst2[i+(NZS-2)*RES*RES];	 	
    }  

   // ---- BASIC STATISTICS ----
   float sum_INP=0.; float sum_REF=0.;
	 for(int i=0;i<NVOXELS;i++){ sum_REF+=imgEst[i];}  
	 for(int i=0;i<NVOXELS;i++){ sum_INP+=imgEst2[i];}  
	 //printf("sumREF (DOBLES)(M)= %.0f  sum TRIPLES(M)= %.0f \n",sum_REF/1.0e6,sum_INP/1.0e6);

  // -----  JOINT BILATERAL GUIDED_FILTER OF TRIPLES -------------------
    // ------------ FILTER PARAMETERS ----- 
	  int ir = 4;
	  float eps = 1.0e-8; 
    float filt_coef = 0.1;
    float max_ref = 0.;
	  float sigma3D = 0.75;
	  float sigmaI = 0.025;
  
    cudaMemcpy(d_INPUT,imgEst2,NVOXELS*sizeof(float),cudaMemcpyHostToDevice);	
	  cudaMemcpy(d_REF,imgEst,NVOXELS*sizeof(float),cudaMemcpyHostToDevice);	    
	  for(int i=0;i<NVOXELS;i++){ if (imgEst[i]>max_ref) max_ref = imgEst[i]; }
	  jointbilateralfilter3D(d_INPUT, d_REF, max_ref, ir, sigma3D, sigmaI, d_OUTPUT);
	  cudaMemcpy(imgEst2f,d_OUTPUT,NVOXELS*sizeof(float),cudaMemcpyDeviceToHost);
	  for(int i=0;i<NVOXELS;i++){imgEst2[i] = (1.0-filt_coef)*imgEst2[i] + filt_coef*imgEst2f[i];}
	  cudaMemcpy(d_INPUT,imgEst_T,NVOXELS*sizeof(float),cudaMemcpyHostToDevice);

	 } // UPDATE DOBLE/TRIPLE
	} // TYPE LOOP (DOUBLE/TRIPLE)

  //--------------------------	
	// ---  IMAGE FOR WRITING OUTPUT
  for(int i=0;i<NVOXELS;i++){
     imgEst_D[i]=imgEst[i]*scale_D;  // Using a global scale factor
     imgEst_T[i]=imgEst2[i]*scale_T; // Using a global scale factor
  } 

  int RESM = RES/2;
	float dist = 1.0;
  	for(int z=0;z<NZS;z++){
  	 for(int y=0;y<RES;y++){
  	  for(int x=0;x<RES;x++){
  	   dist = ((x-RESM)*(x-RESM)+(y-RESM)*(y-RESM)-(RESM-1)*(RESM-1));
   	   if( dist>0.) { 
	     imgEst_D[RES*RES*z+y*RES+x]= 0.f;     
	     imgEst_T[RES*RES*z+y*RES+x]= 0.f;     
	    }	   	  
	  }
   }
  }	
   
  // ---  WRITING IMAGE EVERY 20 ITERATIONS ---
  char c[255]; 
  if (iter%20 == 0) {   
  if (par>=1 && par<=3){	 	  
	  snprintf(c, sizeof(c), "img_GPU_D_ITER%d.raw", iter);
 	  WriteImage(imgEst_D,NVOXELS,c);
	if (par>1){
	  snprintf(c, sizeof(c), "img_GPU_T_ITER%d.raw", iter);
 	  WriteImage(imgEst_T,NVOXELS,c);
	 }
	}
	} // EVERY 20 ITERATIONS

  }  // LOOP OF ITERATIONS
   	    	  
	//name="img_GPU_D.raw";
 	//WriteImage(imgEst_D,NVOXELS,name);   
	//name="img_GPU_T.raw";
 	//WriteImage(imgEst_T,NVOXELS,name);  
		
  // TIME
	time (&end_total);  dif = difftime (end_total,start_total);
	printf("\nTotal Time: %f (s) \n", dif);
	return;
}

int main(int argc, char** argv){

	//TestMemoria();   //This may be useful to get info of the GPU
	
	char filename[1024];
	char filename2[1024];	
	char filenameCal[1024];
  char filenameV[1024];
	int par;
	float ft;
	
	 if (argc==1 || argc>5){
	  printf("1 Input  --> Projection [Input = 3D Image] \n");
	  printf("2 Inputs --> Reconstruction [Input = Sinog_Doubles + Sinog_Normalization]\n");
	  printf("3 Inputs --> Reconstruction Doubles & Triples [Input = Sinog_Doubles + Sinog_Triples + Sinog_Normalization] \n");
	  printf("4 Inputs --> VLOR reconstruction [Input = Sinog_Doubles + Sinog_Triples + Sinog_Normalization + Vlor_file] \n"); 
	  return 0;
	 } else {
	  strcpy(filename,argv[1]);
	 }

	 ft = 1.0f;
	 if (argc==2) {
	  ft=0.f;
	  par=0;
	 }else if (argc==3) {
	  ft=0.f;
	  strcpy(filenameCal,argv[2]);	  
	  par=1;
	 }else if (argc==4){
	  strcpy(filename2,argv[2]);
	  ft=1.;
	  strcpy(filenameCal,argv[3]);
	  par=2;
	 }else if (argc==5) {
    strcpy(filename2,argv[2]);
    ft=1.;
    strcpy(filenameCal,argv[3]);
    strcpy(filenameV,argv[4]);
    par=3;
   }
	 
	osem3d(filename,filename2,filenameCal,filenameV,ft,par);
	return 0;
}
