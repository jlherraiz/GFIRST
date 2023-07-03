//---------------------------------------------------------------------------
// GFIRST - GPU-based Fast Iterative Reconstruction SofTware
// Joaquin L. Herraiz et al. 
// Nuclear Physics Group and IPARCOS - Complutense University of Madrid (UCM)
// Code available at Github: https://github.com/jlherraiz/GFIRST
// This version of the code allows the reconstruction of Doubles and Triples
//   from the preclinical Inveon scanner (See Reference 2)
//
// Main References (Please Cite this):
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
// GPU_Kernels.cu --> GPU KERNELS in CUDA
//---------------------------------------------------------------------------

#define BLOCK_W 8
#define BLOCK_H 8

const int BSX=256;
__constant__ float d_G1D[9];

texture<float,3> imgEstTex; // Name of the Texture in the Estimated Image
cudaArray *array3dImg = 0;  // Array of the Estimated Image in GPU

texture<float,3> CorrTex;    // Name of the Texture in the Correction
cudaArray *array3dCorr = 0;  // Array of the Correction in GPU

texture<float,3> CalibTex;    // Name of the Calibration in the Correction
cudaArray *array3dCalib = 0;  // Array of the Calibration in GPU

__global__ void FORWPROJ_K(float* d_PROJ, int itheta, int idz, int iTOF){
	
	int ix0 = threadIdx.x; // Radial bin
	int iz0 = blockIdx.x;  // Axial bin
	int zmin,idz_signo=0;

	if (ix0 < NRAD && iz0 < NZS){

	float x0;
	float x,y,z;
	float xc,yc,zc;
	float proj;
	float dx,dy,dz;	
	float ang_th,sin_th,cos_th;	
	if (idz==0){zmin=0;}else{zmin = (abs(idz)-1)*SPAN+(SPAN/2+1); idz_signo=idz/abs(idz);}
		
  int ilor = iz0*NRAD+ix0;

	float vx = FOV/ RES;               // Voxel size (mm)
  float px = 0.5f*pitch;             // Detector Pixel size (mm)
	float fx = px/vx; 	               // Convert Detector pixel xy to Voxel

  float tan_dz = float(idz_signo*zmin*(pitch*0.5f)/DIAM_DET);  
	float fz = (FOV/NPT) / (0.5f*pitch);      // Convert points to pixel z (z = x*dx/dz)

	float fac = float(RES)/float(NPT);
	float TOF_size = float(NPT)/float(NTBINS);
	
	float RADM = (NRAD-1)*0.5f;  // ODD NUMBER OF BINS
	float RESM = (RES-1)*0.5f;   // ODD NUMBER OF BINS
	
  proj = 0.f;
		 
	if (iz0>= zmin && iz0<NZS-zmin){  // Z range
    
	 ang_th = -float(itheta)/float(NANG)*PI + PI; // - 0.0f*PI;	
	 sin_th = sinf(ang_th);
	 cos_th = cosf(ang_th);
	    		
	 x0 = fx*float(ix0 - RADM);	  // Coordinates before rotation (voxel units)
	
	 dx =  fac*sin_th;   //*cos_dz;    //fac includes FOV/DISTANCE DETECTORS. Only the FOV is projected (in voxels)
	 dy = -fac*cos_th;   //*cos_dz;
	 dz = (FOV/NPT)*float(idz_signo*zmin)/(0.5f*DIAM_DET);  // Convert voxels to detector pixel z	 

	 // IN-PLANE ROTATION 
	 xc = RESM + (x0*cos_th) + 0.5f;     // Voxel units
	 yc = RESM + (x0*sin_th) + 0.5f;     // Voxel units
	 zc = float(iz0) + 0.5f;	     // Pixel size in Z = Voxel Size in Z	

	 int init_point = floor((iTOF-0.5f)*TOF_size);
	 int end_point = floor((iTOF+0.5f)*TOF_size);
 
	 for (int lambda=init_point;lambda<=end_point;lambda++){        	
	  x = xc + lambda*dx;                // Position in mm converted into voxels	 
	  y = yc + lambda*dy;                // --
	  z = zc + lambda*dz;                // Position in mm converted into z voxels	
	  proj += tex3D(imgEstTex,x,y,z);  //*fact_edge;
	  }    
	} // if Z 

	d_PROJ[ilor]=proj;
}
}

void FORWPROJ(float* d_PROJ, int itheta, int idz, int iTOF){
  FORWPROJ_K<<<NZS,NRAD>>>(d_PROJ, itheta, idz, iTOF);
}

__global__ void BACKWPROJ_K(float* d_NUMER,float* d_DENOM,int itheta, int idz, int iTOF) {	        
		
	int iz = blockIdx.y;
	int iy = blockIdx.x;
 	int ix = threadIdx.x;	
	int iV = iz*RES*RES+iy*RES+ix;	

	float corr=0.f;
	float cal=0.f;
	int zmin,idz_signo;

	if (ix<RES && iy<RES && iz<NZS){

	float x,z;						
	float sin_th,cos_th;	
	float ang_th;
	float xrot,yrot;	

	float dx = FOV/ RES;         // Voxel size (mm)
  float px = pitch / 2.0;      // Detector Pixel size (mm)
	float fx = dx/px; 		       // Convert voxels to detector pixel xy
	float RADM = (NRAD-1)*0.5f;  // EVEN NUMBER OF BINS
	float RESM = (RES-1)*0.5f;   //- 0.5f;

	if (idz==0){zmin=0;}else{zmin = (abs(idz)-1)*SPAN+(SPAN/2+1); idz_signo=idz/abs(idz);}	
	float fz = (dx)*float(idz_signo*zmin)/(DIAM_DET/2.0);  // Convert voxels to detector pixel z
		
	float xv = float(ix-RESM);    // voxel position (voxel units)
	float yv = float(iy-RESM);    // voxel position (voxel units)

   ang_th = -float(itheta)/float(NANG)*PI + PI; // - 0.0f*PI;	
	 sin_th = sinf(ang_th);
	 cos_th = cosf(ang_th);    		
	 	 
	 // In-plane invert rotation (voxel units)
	 xrot =  xv*cos_th + yv*sin_th;
	 yrot = -xv*sin_th + yv*cos_th;

	 x = (RADM) + xrot*fx + 0.5f;   // Now x is radial position (pixel units)	  	 
	 z = float(iz) + yrot*fz + 0.5f; // + 0.5f;	// Position (>=0) in Z at detector
	 
	 if ( z>= zmin && z<NZS-zmin){  // Z range	 	 
	  corr=tex3D(CorrTex,x,z,0.5f);        // Only 1 angle now
	  cal=tex3D(CalibTex,x,z,0.5f); 
         }else{
          corr=0.;
          cal=0.;
         }
	 if (cal<0.f || corr<0.f) {corr=0.f; cal=0.f;}   // GAPS	 
     
	 d_NUMER[iV]+=corr;
	 d_DENOM[iV]+=cal;
     
	 }  // if iV
	
}

void BACKWPROJ(float* d_NUMER,float* d_DENOM, int itheta, int idz, int iTOF){	
	
	dim3 block(RES,1);
	dim3 grid(RES,NZS);
	BACKWPROJ_K<<<grid,block>>>(d_NUMER,d_DENOM,itheta,idz,iTOF);
}

__global__ void divVxVGPU(float* vA, float* vB,int els,float* vR){
	float s_A,s_B,s_R;
	int index = blockIdx.y*RES*RES+blockIdx.x*RES+threadIdx.x;
	if(index<els){  // Shared memory
		s_A=vA[index];
		s_B=vB[index];
		if(s_B<0.001f){s_R=1.0f;}else{s_R=s_A/s_B;}
		vR[index]=s_R;
	}
}

void dividirVxVGPU(float* d_vA,float* d_vB,int els,float* d_vR){
	dim3 block(RES,1);
	dim3 grid(RES,NZS);
	divVxVGPU<<<grid,block>>>(d_vA,d_vB,els,d_vR);
}

__global__ void calcVxVGPU(float* vA, float* vB, float* vC, float fact, int els,float* vR){
	float s_A,s_B,s_R;
	int index = blockIdx.y*RES*RES+blockIdx.x*RES+threadIdx.x;
	if(index<els){
		s_A=vA[index];
		s_B=vB[index]-fact*vC[index];	    
		if(s_B<0.0){s_R=1.0f;}else{s_R=s_A/s_B;}
		vR[index]=s_R;
	}
}

void calcularVxVGPU(float* d_vA,float* d_vB,float* d_vC,float fact,int els,float* d_vR){
	dim3 block(RES,1);
	dim3 grid(RES,NZS);
	calcVxVGPU<<<grid,block>>>(d_vA,d_vB,d_vC,fact,els,d_vR);
}

__global__ void CleanMemoryK(float* mem,int size){
	int index = blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){  		// Set to 0
		mem[index]=0;
	}
}

void CleanMemory(float* mem,int size){
	int bloques = (int)ceil(size/(float)BSX);
	dim3 block(BSX,1);
	dim3 grid(bloques,1);
	CleanMemoryK<<<grid,block>>>(mem,size);
}

void UpdateImage(float* h_imagen){
    // Move volume from CPU to GPU
    cudaExtent dims = make_cudaExtent(RES, RES, NZS);   
    cudaMemcpy3DParms params={0};
    params.srcPtr= make_cudaPitchedPtr((void*)h_imagen, dims.width*sizeof(float), dims.width, dims.height);
    params.dstArray=array3dImg;
    params.extent=dims;
    params.kind=cudaMemcpyHostToDevice;
    cudaMemcpy3D(&params);
}

void actualizarCorr(float* h_imagen){
    // Move volume of corrections from CPU to GPU 
    cudaExtent dims = make_cudaExtent(NRAD,NZS,1);   
    cudaMemcpy3DParms params={0};
    params.srcPtr= make_cudaPitchedPtr((void*)h_imagen, dims.width*sizeof(float), dims.width, dims.height);
    params.dstArray=array3dCorr;
    params.extent=dims;
    params.kind=cudaMemcpyHostToDevice;
    cudaMemcpy3D(&params);
}

void actualizarCalib(float* h_imagen){
    // Move volume of Normalization from CPU to GPU
    cudaExtent dims = make_cudaExtent(NRAD,NZS,1);   
    cudaMemcpy3DParms params={0};
    params.srcPtr= make_cudaPitchedPtr((void*)h_imagen, dims.width*sizeof(float), dims.width, dims.height);
	  params.dstArray=array3dCalib;
    params.extent=dims;
    params.kind=cudaMemcpyHostToDevice;
    cudaMemcpy3D(&params);
}

void crearTexturas(){
	// Create array 3D 
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaExtent dims = make_cudaExtent(RES, RES, NZS);
	cudaMalloc3DArray(&array3dImg,&desc,dims);

	imgEstTex.normalized = 0;	
	imgEstTex.addressMode[0] = cudaAddressModeClamp; 
	imgEstTex.filterMode = cudaFilterModeLinear;     
	cudaBindTextureToArray(imgEstTex,array3dImg,desc);	

	//-------------- CORRECTIONS -------------------------
	cudaChannelFormatDesc desc2 = cudaCreateChannelDesc<float>();
	cudaExtent dims2 = make_cudaExtent(NRAD,NZS,1);
	cudaMalloc3DArray(&array3dCorr,&desc2,dims2);

	CorrTex.normalized = 0;	
	CorrTex.addressMode[0] = cudaAddressModeClamp;   
	CorrTex.filterMode = cudaFilterModeLinear;       
	cudaBindTextureToArray(CorrTex,array3dCorr,desc2);	

	//-------------- CALIBRATIONS ---------------------------
	cudaMalloc3DArray(&array3dCalib,&desc2,dims2);
	CalibTex.normalized = 0;	
	CalibTex.addressMode[0] = cudaAddressModeClamp;  
	CalibTex.filterMode = cudaFilterModeLinear;          
	cudaBindTextureToArray(CalibTex,array3dCalib,desc2);	
}

void TestMemoria(){
cudaDeviceProp dP;
cudaGetDeviceProperties(&dP,0);
printf("Max threads per block: %d \n",dP.maxThreadsPerBlock);
printf("Max Threads DIM: %d x %d x %d \n",dP.maxThreadsDim[0],dP.maxThreadsDim[1],dP.maxThreadsDim[2]);
printf("Max Grid Size: %d x %d x %d \n",dP.maxGridSize[0],dP.maxGridSize[1],dP.maxGridSize[2]);
printf("totalGlobalMem: %d \n",dP.totalGlobalMem);
printf("sharedMemPerBlock: %d \n",dP.sharedMemPerBlock);
printf("regsPerBlock: %d \n",dP.sharedMemPerBlock);
printf("warpSize: %d \n",dP.warpSize);
printf("memPitch: %d \n",dP.memPitch);
printf("totalConstMem: %d \n",dP.memPitch);
printf("clockRate (kHz): %d \n",dP.clockRate);
printf("textureAlignment: %d \n",dP.textureAlignment);
printf("deviceOverlap: %d \n",dP.deviceOverlap);
printf("multiProcessorCount: %d \n",dP.multiProcessorCount);
printf("kernelExecTimeoutEnabled: %d \n",dP.kernelExecTimeoutEnabled);
printf("integrated: %d \n",dP.integrated);
printf("canMapHostMemory: %d \n",dP.canMapHostMemory);
printf("computeMode: %d \n",dP.computeMode);
}

void chequearError(const char *errorStr){
	cudaError error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf(errorStr);
		printf(": %s.\n",cudaGetErrorString(error));
		getchar();
		exit(-1);
	}
	error = cudaThreadSynchronize();
	if(error!=cudaSuccess){
		printf(errorStr);
		printf(": %s.\n",cudaGetErrorString(error));
		getchar();
		exit(-1);
	}
}

void calcGaussianCoefficients(float PSF){
	//create the 1D kernel co-efficients  // 7x7 kernel
	float G1D[9];	
	float total=0.f, factor;
	int g1dindex=0;
	for (int z=-4;z<=4;++z)	{total+= exp(-((z*1.662f/(PSF+0.1f))*(z*1.662f/(PSF+0.1f))));} // Factor Normalizacion
	for (int z=-4;z<=4;++z)	{factor= exp(-((z*1.662f/(PSF+0.1f))*(z*1.662f/(PSF+0.1f))))/total; G1D[g1dindex] = factor; g1dindex++;}
	cudaMemcpyToSymbol(d_G1D, G1D, sizeof(G1D));	
}

__global__ void convolutionXY_GPU(float* d_volume,float* d_tempVolume){
	int plane = blockIdx.x;       //the plane
  int col = threadIdx.x;        //the column //  each thread does a full column
  int maximo = NZS*RES*RES-1;
	__shared__ float yConvol[RES];  //RES
  float valCache[9];  //should be stored in reg and not lmem - but always good idea to check ptx / cubin output  // MAX 7x7 CONVOLUTION KERNEL
  valCache[0]=0; valCache[1]=0; valCache[2]=0; valCache[3]=0; valCache[4]=0; valCache[5]=0; valCache[6]=0; valCache[7]=0; valCache[8]=0;
  if (plane<NZS && col <RES) {
  for (int c=-4;c<RES;++c)	{  
    int position = plane*RES*RES+(c+4)*RES+col;
	if (position>maximo) position=maximo;
	float inputVal = d_volume[position];  	 
    if (c>=RES-4) inputVal = 0.f;    
    valCache[0]=valCache[1]; valCache[1]=valCache[2]; valCache[2]=valCache[3]; valCache[3]=valCache[4]; valCache[4]=valCache[5]; valCache[5]=valCache[6]; valCache[6]=valCache[7];valCache[7]=valCache[8]; valCache[8]=inputVal;
    if (c>=0) { 
     float outputVal=valCache[0]*d_G1D[0]+valCache[1]*d_G1D[1]+valCache[2]*d_G1D[2]+valCache[3]*d_G1D[3]+valCache[4]*d_G1D[4]+valCache[5]*d_G1D[5]+valCache[6]*d_G1D[6]+valCache[7]*d_G1D[7]+valCache[8]*d_G1D[8];
     yConvol[col] = outputVal;
     __syncthreads(); //dangerous to put this in a loop/conditional but in this case should be fine - can probably do without it as 1/2 warp will be sync'			
     float xConvol = valCache[4]; //now calc the x convol
     //     if ((col>1)&&(col<RES-1)){ xConvol  = yConvol[col-2]*d_G1D[0]+yConvol[col-1]*d_G1D[1]+yConvol[col]*d_G1D[2]+yConvol[col+1]*d_G1D[3]+yConvol[col+2]*d_G1D[4];}
     if ((col>3)&&(col<RES-4)){ xConvol  = yConvol[col-4]*d_G1D[0]+yConvol[col-3]*d_G1D[1]+yConvol[col-2]*d_G1D[2]+yConvol[col-1]*d_G1D[3]+yConvol[col]*d_G1D[4]+yConvol[col+1]*d_G1D[5]+yConvol[col+2]*d_G1D[6]+yConvol[col+3]*d_G1D[7]+yConvol[col+4]*d_G1D[8];}   
     __syncthreads();
	   d_tempVolume[plane*RES*RES+c*RES+col] = xConvol;  //coalesced    
    }    	       
   }
  }	
}

void convolutionXY(float *d_volume, float *d_tempVolume ){    
  dim3 threads(RES,1);  
  dim3 blocks(NZS,1);
  convolutionXY_GPU<<<blocks, threads>>>(d_volume,d_tempVolume);  
}

__global__ void convolution_sinog_GPU(float* d_volume){
	
  int Z = blockIdx.x;       //the plane
  int Y = threadIdx.x;      //the column //  each thread does a full column  
  int init_pos = Z*NANG*NRAD+Y*NRAD;
  int position=0;
  float inputVal=0.;
  float outputVal=0.;
  float valCache[9];  //should be stored in reg and not lmem - but always good idea to check ptx / cubin output  // MAX 7x7 CONVOLUTION KERNEL
  valCache[0]=0; valCache[1]=0; valCache[2]=0; valCache[3]=0; valCache[4]=0; valCache[5]=0; valCache[6]=0; valCache[7]=0; valCache[8]=0;  
  float RConvol[NRAD];  
  if (Z<NSINOGS && Y < NANG) {
   for (int c=-4;c<NRAD;++c)	{  
    position = init_pos + (c+4);
	  inputVal = d_volume[position];  	 
    if (c>=NRAD-4) inputVal = 0.f;    
    valCache[0]=valCache[1]; valCache[1]=valCache[2]; valCache[2]=valCache[3]; valCache[3]=valCache[4]; valCache[4]=valCache[5]; valCache[5]=valCache[6]; valCache[6]=valCache[7];valCache[7]=valCache[8]; valCache[8]=inputVal;
    if (c>=0) { 
     outputVal=valCache[0]*d_G1D[0]+valCache[1]*d_G1D[1]+valCache[2]*d_G1D[2]+valCache[3]*d_G1D[3]+valCache[4]*d_G1D[4]+valCache[5]*d_G1D[5]+valCache[6]*d_G1D[6]+valCache[7]*d_G1D[7]+valCache[8]*d_G1D[8];
     RConvol[c] = outputVal;
    }    	       
   }
   __syncthreads();
   for (int c=0;c<NRAD;++c){
    d_volume[init_pos+c] = RConvol[c]; 
   }  
  }	
}

void convolution_sinog(float *d_volume){    
  dim3 threads(NANG,1);  
  dim3 blocks(NSINOGS,1);
  convolution_sinog_GPU<<<blocks, threads>>>(d_volume);  
}

__global__ void median_filterXY_K( float *input, float *output){
    __shared__ float window[BLOCK_W*BLOCK_H][9];
    unsigned int x=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int y=blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int tid=threadIdx.y*blockDim.y+threadIdx.x;
    float initial; 
    if(x>=RES || y>=RES)        return;
    
    for (unsigned int z=0;z<NZS;++z)	{		
	   
    initial = input[z*RES*RES+y*RES + x];
	   
    window[tid][0]= (y==0||x==0)?0.0f:input[z*RES*RES+(y-1)*RES+x-1];
    window[tid][1]= (y==0)?0.0f:input[z*RES*RES+(y-1)*RES+x];
    window[tid][2]= (y==0||x==RES-1)?0.0f:input[z*RES*RES+(y-1)*RES+x+1];
    window[tid][3]= (x==0)?0.0f:input[z*RES*RES+y*RES+x-1];
    window[tid][4]=input[z*RES*RES+y*RES+x];
    window[tid][5]= (x==RES-1)?0.0f:input[z*RES*RES+y*RES+x+1];
    window[tid][6]= (y==RES-1||x==0)?0.0f:input[z*RES*RES+(y+1)*RES+x-1];
    window[tid][7]= (y==RES-1)?0.0f:input[z*RES*RES+(y+1)*RES+x];
    window[tid][8]= (y==RES-1||x==RES-1)?0.0f:input[z*RES*RES+(y+1)*RES+x+1];
    __syncthreads();
 
    // Order elements (only half of them)
    for (unsigned int j=0; j<5; ++j)
    {
        // Find position of minimum element
        int min=j;
        for (unsigned int l=j+1; l<9; ++l)
            if (window[tid][l] < window[tid][min])
                min=l; 
        // Put found minimum element in its place
        const float temp=window[tid][j];
        window[tid][j]=window[tid][min];
        window[tid][min]=temp;
        __syncthreads();
    }
    output[z*RES*RES+y*RES + x]=0.5*initial + 0.5*window[tid][4];	
	}
}

void median_filterXY(float *input, float *output){
   dim3 threads(RES/BLOCK_W, RES/BLOCK_H);
   dim3 blocks(BLOCK_W,BLOCK_H);
   median_filterXY_K<<<threads,blocks>>>(input,output);  
}

__global__ void median_filter3D_K( float *input, float *output){
    __shared__ float window[BLOCK_W][27];
    unsigned int ip=threadIdx.x;
    unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int iy=blockIdx.y;
    unsigned int iz=blockIdx.z;
    unsigned int id = iz*RES*RES + iy*RES + ix;
    unsigned int iV;
    unsigned int RES2 = RES*RES;
    if(ip<BLOCK_W && ix<RES && iy<RES && iz<NZS) {
    unsigned int iw = 0;
    unsigned int ic = 0;
    float media = 0.;
    for (unsigned int k=0; k<3; k++){
    for (unsigned int j=0; j<3; j++){    
    for (unsigned int i=0; i<3; i++){
     iV = id + (k-1)*RES2 + (j-1)*RES + (i-1);  
     if (iV>=0 && iV<RES*RES*NZS) { 
      window[ip][iw] = input[iV]; 
      media += input[iV]; 
      if (input[iV]>1.0e-8) ic++;
     } else {
      window[ip][iw]=0.;
     }        
     iw++;     
    }
    }
    }
    media = media/27.0;
    __syncthreads();
        
    // Order elements (only half of them)
    for (unsigned int j=0; j<15; j++)    {
     // Find position of minimum element
     float minim = window[ip][j];
     unsigned int imin=j;  
     for (unsigned int k=j+1; k<27; k++) {if (window[ip][k] < minim) minim=window[ip][k]; imin=k;  }
     // Put found minimum element in its place
     window[ip][imin]= window[ip][j];
     window[ip][j]= minim;
     __syncthreads();
    }

    float median = window[ip][14];
    if (ic==27) {
     output[id] = median;   // Avoid set to zero the borders of the image
    } else {
     output[id] = input[id];
    }
    
   }
   __syncthreads();
}

void median_filter3D(float *input, float *output){
   dim3 threads(BLOCK_W,1);
   dim3 blocks(RES/BLOCK_W,RES,NZS);
   median_filter3D_K<<<blocks,threads>>>(input,output);  
}

__global__ void jointbilateralfilter3D_K(float *input, float *reference, float max_ref, int r, float sigma3D, float sigmaI, float *output){
// JOINT BILATERAL FILTER

    unsigned int ix=threadIdx.x;
    unsigned int iy=blockIdx.x;
    unsigned int iz=blockIdx.y;
    
    if(ix>=0 && ix<RES && iy>=0 && iy<RES && iz>=0 && iz<NZS) {
     unsigned int iV= iz*RES*RES + iy*RES + ix;    
     float ref_value = reference[iV];
     float norm = 0.;
     float sum = 0.;
     for (unsigned int izz=iz-r;izz<=iz+r;izz++) {
     for (unsigned int iyy=iy-r;iyy<=iy+r;iyy++) {	
     for (unsigned int ixx=ix-r;ixx<=ix+r;ixx++) {		
      if(ixx>=0 && ixx<RES && iyy>=0 && iyy<RES && izz>=0 && izz<NZS) {
        unsigned int iW = izz*RES*RES + iyy*RES + ixx;
        float dist_intensity2 = ((reference[iW] - ref_value)/max_ref)*((reference[iW] - ref_value)/max_ref);  //REF      
	float dist_spatial2 = float((ixx-ix)*(ixx-ix) + (iyy-iy)*(iyy-iy) + (izz-iz)*(izz-iz));
	float H = exp(-dist_intensity2/((2.0*sigmaI)*(2.0*sigmaI)));
	float G = exp(-dist_spatial2/((2.0*sigma3D)*(2.0*sigma3D)));                
	norm += H*G;
	sum += H*G*input[iW];
      }
     }
     }
     }
     if (norm>1e-8){ output[iV] = sum/norm; }else{output[iV] = 0.; }    
     
    }
    __syncthreads();
}
void jointbilateralfilter3D(float *input, float *reference, float max_ref, int r, float sigma3D, float sigmaI, float *output){
   dim3 blocks(RES,NZS);
   dim3 threads(RES,1);
   jointbilateralfilter3D_K<<<blocks,threads>>>(input,reference,max_ref,r,sigma3D, sigmaI,output);   
}