/* Updated 9/19/2012 by Matt Smith to 64-bit support on Windows and Mac */
/* This removes 32-bit support from this function (use readNEV_32bit)   */

#include <math.h>
#include "mex.h"
#include <stdio.h>
#include <stdint.h>
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  char * filename, * wd;
 FILE * fid;
 int buflen;

 int bytesPerSample = 2;
 unsigned int numSamples;

 char fileType[8];
 unsigned char version[2];
 char fileFormatAdditional[2];
 #ifdef _WIN64 /* check whether we're on windows */
    _int64 fileBytes;
 #else /* if not, use Mac/Linux syntax */
    off_t fileBytes; 
 #endif
         
 unsigned int headerSize;
 unsigned int timeResTimeStamps;
 unsigned int timeResSamples;
 unsigned int packetSize;
 unsigned short timeOrigin[8];
 char application[32];
 unsigned char comment[256];
 int extendedHeaderNumber;
 int64_t i;
 int64_t spikeCount;
 double* nevData;
 
 unsigned char junk[500];

 unsigned int time;
 unsigned short unit;
 short packetID;
 
 /* Get filename */
 buflen = (mxGetM(prhs[0]) * mxGetN(prhs[0]) * sizeof(mxChar))+1;
 filename = mxCalloc(buflen, sizeof(char));
 mxGetString(prhs[0], filename, buflen); 
 
 if ((fid = fopen(filename,"rb")) == NULL)
 {
   wd = (char *)mxCalloc(100,1);
   getcwd(wd,100);
   mexPrintf("File %s/%s does not exist\n",wd,filename);
   plhs[0] = mxCreateNumericMatrix(0,0,mxDOUBLE_CLASS,mxREAL);
   return;
 }
 
 #ifdef _WIN64
    _fseeki64(fid,0,SEEK_SET);
 #else
    fseeko(fid,0,SEEK_SET);
 #endif
 
 /* Read the header */
 fread(fileType, 1, 8, fid);
 fread(version, 1, 2, fid);
 fread(fileFormatAdditional, 1, 2, fid);
 fread(&headerSize, 4, 1, fid);
 fread(&packetSize, 4, 1, fid);
 fread(&timeResTimeStamps, 4, 1, fid);
 fread(&timeResSamples, 4, 1, fid);
 fread(timeOrigin, 2, 8, fid);
 fread(application, 1, 32, fid);
 fread(comment, 1, 256, fid);
 fread(&extendedHeaderNumber,4,1,fid);
 numSamples = (packetSize-8)/bytesPerSample;

 #ifdef _WIN64
    _fseeki64(fid,0,SEEK_END);
 #else
    fseeko(fid,0,SEEK_END);
 #endif
 
 #ifdef _WIN64
    fileBytes = _ftelli64(fid);
 #else
    fileBytes = ftello(fid);
 #endif
         
 spikeCount = ((int64_t)fileBytes - (int64_t)headerSize)/(int64_t)packetSize;
 
/* mexPrintf("nlhs params: %li\n",nlhs); */
 
 /* if there are 2 params on the left side */
 /* create a matrix to return the waverosm and fill it up below */
/*  plhs[1] = mxCreateNumericMatrix(spikeCount,numSamples,mxDOUBLE_CLASS,mxREAL); */
 /* waveData = mxGetPr(plhs[1]); */
 
 mexPrintf("Reading %s, %llu events...\n",filename,spikeCount);
 mexPrintf("File size %llu, Header size %li, Packet size %li\n",fileBytes,headerSize,packetSize);
 
 fseek(fid,headerSize,SEEK_SET);
 
 plhs[0] = mxCreateNumericMatrix(spikeCount,3,mxDOUBLE_CLASS,mxREAL);
 nevData = mxGetPr(plhs[0]);
 
 for (i = 0; i < spikeCount; i++)
 {
   fread(&time,4,1,fid);
   fread(&packetID,2,1,fid);
   unit = 0;
   fread(&unit,1,1,fid);
   
   if (packetID == 0)  /* digital code */
     {
       fread(junk,1,1,fid);
       fread(&unit,2,1,fid);
       fread(junk,packetSize-10,1,fid);   
     }
   else
     {
       fread(junk,packetSize-7,1,fid); /* this is the waveform */
       /* right now it's thrown out, but if there are two params on the */
       /* left hand side then we should fill up the matrix */
       /* *(waveData+i*numSamples) = waveformdata */
       
     }
   
   *(nevData+i+spikeCount*2) = ((double)time)/timeResSamples;
   *(nevData+i) = packetID;
   *(nevData+i+spikeCount) = unit;     
   
   if (i % (spikeCount/10) == 0)
     {
       mexPrintf("  %lld/%lld\n",i,spikeCount);
       mexEvalString("drawnow;");
     }
 }
 
 fclose(fid);
}

