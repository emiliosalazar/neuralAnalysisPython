/* Updated 9/19/2012 by Matt Smith to 64-bit support on Windows and Mac */
/* This removes 32-bit support from this function (use readNEV_32bit)   */

#include <math.h>
#include "mex.h"
#include <stdio.h>
#include <stdint.h>
#include "unistd.h"
#define GetCurrentDir getcwd
#include <string.h>
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#define MAXCHAN 5121 /* NEV 2.07 spec max of 5120 recording electrodes + 1 for zero-indexing*/

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
    __int64 fileBytes;
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
    
    int64_t j;
    char Identifier[8];
    char newIdentifier[9];
    unsigned int ElecID;
    unsigned char temp0[24];
    
    unsigned char PhysConnect[1];
    unsigned char PhysConnectPin[1];
    unsigned int nVperBit;
    unsigned int nVperBitVec[MAXCHAN]; /* hard-coded max # of recording electrodes */
    unsigned int EnergyThresh;
    unsigned int HighThresh;
    unsigned int LowThresh;
    unsigned char SortedUnits[1];
    unsigned char BytesPerSample[1];
    unsigned char temp1[10];
    
    int64_t i;
    int64_t spikeCount;
    int64_t n;
    double* nevData;
    double* waveData;
    int16_t tempwave;
    unsigned char junk[500];
    int16_t* nvb;
    
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
    fread(fileType, 1, 8, fid);//8
    fread(version, 1, 2, fid);//2
    fread(fileFormatAdditional, 1, 2, fid);//2
    fread(&headerSize, 4, 1, fid);//4
    fread(&packetSize, 4, 1, fid);//4
    fread(&timeResTimeStamps, 4, 1, fid);//4
    fread(&timeResSamples, 4, 1, fid);//4
    fread(timeOrigin, 2, 8, fid);//16
    fread(application, 1, 32, fid);//32
    fread(comment, 1, 256, fid);//256
    fread(&extendedHeaderNumber,4,1,fid);//4
    numSamples = (packetSize-8)/bytesPerSample;
    
    mexPrintf("NumExtendedHeader:%d\n",extendedHeaderNumber);
    
    for(j=1;j<=extendedHeaderNumber;j++){
        fread(Identifier, 1, 8, fid);
        memcpy(newIdentifier,&Identifier[0],8);
        newIdentifier[8] = '\0';
        if(strcmp(newIdentifier,"NEUEVWAV")==0){
            fread(&ElecID,2,1,fid);
            fread(PhysConnect,1,1,fid);
            fread(PhysConnectPin,1,1,fid);
            fread(&nVperBit,2,1,fid);
            if ((uint16_t)ElecID>0 && (uint16_t)ElecID<=MAXCHAN) {
                nVperBitVec[(uint16_t)ElecID]=(uint16_t)nVperBit;
            }
            fread(&EnergyThresh,2,1,fid);
            fread(&HighThresh,2,1,fid);
            fread(&LowThresh,2,1,fid);
            fread(SortedUnits,1,1,fid);
            fread(BytesPerSample,1,1,fid);
            fread(temp1,1,10,fid);
        }
        else{
            fread(temp0,1,24,fid);
        }
    }
    
    mexPrintf("Done reading ext headers\n");
    
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
    
    
    mexPrintf("Reading %s, %llu events...\n",filename,spikeCount);
    mexPrintf("File size %llu, Header size %li, Packet size %li\n",fileBytes,headerSize,packetSize);
    
    fseek(fid,headerSize,SEEK_SET);
    
    plhs[0] = mxCreateNumericMatrix(spikeCount,3,mxDOUBLE_CLASS,mxREAL);
    nevData = mxGetPr(plhs[0]);
    
    /* if there are 2 params on the left side */
    /* create a matrix to return the waverosm and fill it up below */
    if (nlhs>=2){
        plhs[1] = mxCreateNumericMatrix(numSamples,spikeCount,mxDOUBLE_CLASS,mxREAL);
        waveData = mxGetPr(plhs[1]);
    }
    
    
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
            fread(junk,1,1,fid);
            /* this is the waveform */
            /* right now it's thrown out, but if there are two params on the */
            /* left hand side then we should fill up the matrix */
            /* *(waveData+i*numSamples) = waveformdata *nVperBitVec[(uint16_t)packetID] */
            for(n=0;n<numSamples;n++){
                fread(&tempwave,2,1,fid);
                if (nlhs>=2){
                    *(waveData+i*numSamples+n) =(double)tempwave*nVperBitVec[(uint16_t)packetID]*0.001;
                }
            }
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
