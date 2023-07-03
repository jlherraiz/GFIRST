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
// CPU_KERNELS.cpp --> FUNCTIONS FOR INPUT, OUTPUT & OTHER AUXILIARY FUNCTIONS
//---------------------------------------------------------------------------

void WriteImage(float* imagen,int numvoxels, char* name){
	// This function writes a volume into a file
	FILE* fichero;
	fichero=fopen(name,"wb");
	if(fichero){
		fwrite(imagen,sizeof(float),numvoxels,fichero);
		fclose(fichero);
	}
}

float* CreateInitialCylinder(float val){
	// Create a Initial Cylinder (Diameter=RES) with provided value
	float* cuboPtr;
	float valor;
	size_t memSize = RES*RES*NZS*sizeof(float);
  cuboPtr=(float*)malloc(memSize);
  int iV=0;
	for(int z=0;z<NZS;z++){
	 for(int y=0;y<RES;y++){
	  for(int x=0;x<RES;x++){
     if (pow(float(x-RES/2),2)+pow(float(y-RES/2),2)<=(RES/2)*(RES/2)){	 	    
      valor = val;
	   }else{
      valor = 0.0f;
	   }
     cuboPtr[iV] = valor;
	   iV++;
		}
	 }
	}
	return cuboPtr;
}

float* ReadImage( char* name){
	FILE* fichero;
  float* Objeto;	
  Objeto=(float*)malloc(NVOXELS*sizeof(float));
	fichero=fopen(name,"rb");		
	fread(Objeto,sizeof(float),NVOXELS,fichero);
	fclose(fichero);
	return Objeto;
}

float* ReadImageInt( char* name){
	FILE* fichero;
	int* datos;		
	datos = (int*)malloc(NVOXELS*sizeof(int));
  float* Objeto;	
  Objeto=(float*)malloc(NVOXELS*sizeof(float));
	fichero=fopen(name,"rb");		
	fread(datos,sizeof(int),NVOXELS,fichero);
	fclose(fichero);
	for (int i=0;i<NVOXELS;i++){Objeto[i]=datos[i];} //valor_cal = datos_cal[offset+i];
	return Objeto;
}

float* ReadSinogram( char* name){
	FILE* fichero;
	short* datos;	
	float* sinog;		
	datos = (short*)malloc(NDATA*sizeof(short));
	sinog = (float*)malloc(NDATA*sizeof(float));	
	fichero=fopen(name,"rb");		
	fread(datos,sizeof(short),NDATA,fichero);
	fclose(fichero);
	for (int i=0;i<NDATA;i++){sinog[i]=datos[i];} //valor_cal = datos_cal[offset+i];
	return sinog;
}

float* ReadSinogramProj( char* name){
	FILE* fichero;	
	float* sinog;		
	sinog = (float*)malloc(NDATA*sizeof(float));	
	fichero=fopen(name,"rb");		
	fread(sinog,sizeof(float),NDATA,fichero);
	fclose(fichero);		
	return sinog;
}

float* ReadSinogram_Type( char* name, int &tipod){  //11Feb2020 - 2*NDATA size to include background
	FILE* fichero;	
	int* datos;	
	float* sinog;				
	datos = (int*)malloc(2*NDATA*sizeof(int));	
	sinog = (float*)malloc(2*NDATA*sizeof(float));	
	for (int i=0;i<2*NDATA;i++){datos[i]=0; sinog[i]=0.;}  //All initial values are zero
  // calculating the size of the file 	
  fichero=fopen(name,"rb");
  if (fichero == NULL) { printf("Sinogram Not Found!\n"); return sinog; }   
  fseek(fichero, 0L, SEEK_END);        
  long int sizefile = ftell(fichero)/sizeof(int); 
  fclose(fichero); 

	fichero=fopen(name,"rb");		
	fread(datos,sizeof(int),sizefile,fichero);
	fclose(fichero);		              
	
	float max = 0.f;
	for (int i=0;i<2*NDATA;i++){sinog[i]=datos[i]; if (sinog[i]>max) {max=sinog[i];}} 
	tipod = 0;  // Integer - Normal file
	
	if (max>1.0e6) {
	 //printf("---------------------------Reading a sinogram with float values \n");
	 fichero=fopen(name,"rb");		
	 fread(sinog,sizeof(float),sizefile,fichero);
	 fclose(fichero);	
	 tipod = 1;  // Float projected file
	}

  float sum=0.;
	float sumbg=0.;	
  for (int i=0;i<NDATA;i++){ 
    sum+=sinog[i]; 
    sumbg+=sinog[NDATA+i];
  }
  //printf(" SUM SINOG = %f  SUM BG = %f \n",sum,sumbg);	
	return sinog;
}

  float* crearObjeto(){
	float* Objeto;	
	size_t memSize = RES*RES*NZS*sizeof(float);
  Objeto=(float*)malloc(memSize);
  float valor;
  int iv=0;
  for (int izz=0;izz<NZS;izz++){
   for (int iyy=0;iyy<RES;iyy++){
    for (int ixx=0;ixx<RES;ixx++){
  	 valor = 0.0f;
 		 if (pow(float(ixx-RES/4),2)+pow(float(iyy-RES/2),2)<=900.0f) valor = 0.1f; 
		 if (pow(float(ixx-RES/4),2)+pow(float(iyy-RES/2),2)<=10.0f && abs(izz-NZS/4)<=4) valor = 1.0f;
     Objeto[iv]=valor;
     iv++;
    }
   }
  }

	return Objeto;
}

void gaussian_filterZ(float* imagen,float* imagen_filtrada){

 int iV=0;
 int RES2=RES*RES;
 float g1D[5];	
 float total=0.f, factor;
 int g1dindex=0;
 for (int z=-2;z<=2;++z)	{total+= expf(-((z*1.0f)*(z*1.0f)));} // Factor Normalizacion
 for (int z=-2;z<=2;++z)	{factor= expf(-((z*1.0f)*(z*1.0f)))/total; g1D[g1dindex] = factor; g1dindex++;}
 for (int K=0;K<NZS;K++){
  for (int J=0;J<RES;J++){
   for (int I=0;I<RES;I++){

    iV++;	
    if (K==0) imagen_filtrada[iV]=0.5f*imagen[iV]+0.5f*imagen[iV+RES2];
    if (K==NZS-1) imagen_filtrada[iV]=0.5f*imagen[iV-RES2]+0.5f*imagen[iV];
    if (K==1 || K==NZS-2) imagen_filtrada[iV]=0.333f*imagen[iV-RES2]+0.333f*imagen[iV]+0.333f*imagen[iV+RES2];
    if (K>1 && K<NZS-2)imagen_filtrada[iV]=g1D[0]*imagen[iV-2*RES2]+g1D[1]*imagen[iV-RES2]+g1D[2]*imagen[iV]+g1D[3]*imagen[iV+RES2]+g1D[4]*imagen[iV+2*RES2];
   }
  }
 }

}

float* cargarCalib(int tipod){
	int* intdatos;
	float* datos;	
	int offset = 0;	
	int elem=NDATA+offset;
	intdatos = (int*)malloc(elem*sizeof(int));
	datos = (float*)malloc(elem*sizeof(float));	
	FILE* fichero;     
	if (tipod==0) {   // Normal
	printf("Reading Integer Calibration File \n");
    fichero=fopen("calibration.raw","rb");	// CALIB	
	fread(intdatos,sizeof(int),elem,fichero);
	fclose(fichero);		
	for (int i=1;i<NDATA;i++){ datos[i]=intdatos[i]; }
	}else if (tipod==1){
	 printf("Reading Float Calibration File \n");
	 fichero=fopen("calibration.raw","rb");	// CALIB	
	 fread(datos,sizeof(float),elem,fichero);
	 fclose(fichero);	
	}else{
     printf("Using Ones as Calibration File \n");
	 for (int i=1;i<NDATA;i++){ datos[i]=100.0f; }
	}

	return datos;
}

