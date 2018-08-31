// This file contains some old, unused functions in pfock.c, fock_buf.c and taskq.c

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <ga.h>
#include <macdecls.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>
#include <assert.h>
#include <math.h>

#include "pfock.h"
#include "config.h"
#include "fock_task.h"
#include "fock_buf.h"
#include "taskq.h"
#include "screening.h"
#include "one_electron.h"

#include "Buzz_Matrix.h"
#include "utils.h"

PFockStatus_t PFock_putDenMat(int rowstart, int rowend,
                              int colstart, int colend,
                              int stride, double *dmat,
                              int index, PFock_t pfock)
{
    int lo[2];
    int hi[2];
    int ld[1];

    if (pfock->committed == 1) {
        PFOCK_PRINTF (1, "Can't change density matrix"
                      " after PFock_commitDenMats() is called.\n");
        return PFOCK_STATUS_EXECUTION_FAILED;
    }
    if (index < 0 || index >= pfock->num_dmat) {
        PFOCK_PRINTF (1, "Invalid index\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
    
    lo[0] = rowstart;
    hi[0] = rowend;    
    lo[1] = colstart;
    hi[1] = colend;
    ld[0] = stride;
    
    NGA_Put(pfock->ga_D[index], lo, hi, (void *)dmat, ld);
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_putDenMatGA(int ga, int index, PFock_t pfock)
{
    GA_Copy(ga, pfock->ga_D[index]);
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_fillDenMat(double value, int index,
                               PFock_t pfock)
{
    if (pfock->committed == 1) {
        PFOCK_PRINTF (1, "Can't change density matrix"
                      " after PFock_commitDenMats() is called.\n");
        return PFOCK_STATUS_EXECUTION_FAILED;
    }
    if (index < 0 || index >= pfock->num_dmat) {
        PFOCK_PRINTF (1, "Invalid index\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
    
    GA_Fill(pfock->ga_D[index], &value);
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_commitDenMats(PFock_t pfock)
{
    GA_Sync();
    pfock->committed = 1;
    if (pfock->nosymm == 1) {
        for (int i = 0; i < pfock->num_dmat; i++) {
            GA_Transpose(pfock->ga_D[i], pfock->ga_D[i + pfock->num_dmat]);
        }
    }

    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_sync(PFock_t pfock)
{
    GA_Sync();
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getMat(PFock_t pfock, PFockMatType_t type,
                           int index,
                           int rowstart, int rowend,
                           int colstart, int colend,
                           int stride, double *mat)
{
    int lo[2];
    int hi[2];
    int ld[1];
    int *ga;
    if (index < 0 || index >= pfock->max_numdmat) {
        PFOCK_PRINTF (1, "Invalid index\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
#ifdef _SCF_
    if (PFOCK_MAT_TYPE_J == type || PFOCK_MAT_TYPE_K == type) {
        PFOCK_PRINTF (1, "Invalid matrix type\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
#endif    

    lo[0] = rowstart;
    hi[0] = rowend;    
    lo[1] = colstart;
    hi[1] = colend;
    ld[0] = stride;
    
    ga = pfock->gatable[type];
    NGA_Get(ga[index], lo, hi, mat, ld);

#ifndef __SCF__
    if (PFOCK_MAT_TYPE_F == type) {
        int sizerow = rowend - rowstart + 1;
        int sizecol = colend - colstart + 1;
        double *K = (double *)PFOCK_MALLOC(sizerow * sizecol * sizeof(double));
        if (NULL == K) {
            PFOCK_PRINTF(1, "Failed to allocate memory: %lld\n",
                sizerow * sizecol * sizeof(double));
            return PFOCK_STATUS_ALLOC_FAILED;
        }
        int ga_K = pfock->ga_K[index];
        NGA_Get(ga_K, lo, hi, K, &stride);
        #pragma omp parallel for
        for (int i = 0; i < sizerow; i++) {
            #pragma simd
            for (int j = 0; j < sizecol; j++) {
                mat[i * stride + j] += K[i * sizecol + j];
            }
        }
        PFOCK_FREE(K);
    }    
#endif

    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_getMatGA(PFock_t pfock, PFockMatType_t type,
                             int index, int ga)
{
    int *my_ga = pfock->gatable[type];
    GA_Copy(my_ga[index], ga);
    
 #ifndef __SCF__
    if (PFOCK_MAT_TYPE_F == type) {
        int ga_K = pfock->ga_K[index];
        double fone = 1.0;
        double fzero = 0.0;
        GA_Add(&fone, ga_K, &fzero, ga, ga);
    }    
#endif

    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getLocalMatInds(PFock_t pfock,
                                    int *rowstart, int *rowend,
                                    int *colstart, int *colend)
{
    int lo[2];
    int hi[2];
    int myrank;  
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    NGA_Distribution(pfock->ga_D[0], myrank, lo, hi);
    *rowstart = lo[0];
    *rowend = hi[0];
    *colstart = lo[1];
    *colend = hi[1];

    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getLocalMatPtr(PFock_t pfock,
                                   PFockMatType_t type, int index,
                                   int *rowstart, int *rowend,
                                   int *colstart, int *colend,
                                   int *stride, double **mat)
{
    int lo[2];
    int hi[2];
    int myrank;
    int *ga;

    if (index < 0 || index >= pfock->max_numdmat)
    {
        PFOCK_PRINTF (1, "Invalid index\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
    
    ga = pfock->gatable[type];
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    NGA_Distribution (ga[index], myrank, lo, hi);
    NGA_Access (ga[index], lo, hi, mat, stride);
    *rowstart = lo[0];
    *rowend = hi[0];
    *colstart = lo[1];
    *colend = hi[1];
    
    return PFOCK_STATUS_SUCCESS;  
}


PFockStatus_t PFock_getMatGAHandle(PFock_t pfock,
                                   PFockMatType_t type, int index,
                                   int *ga)
{
    if (index < 0 || index >= pfock->max_numdmat)
    {
        PFOCK_PRINTF (1, "Invalid index\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
    
    int *g = pfock->gatable[type];
    *ga = g[index];
    return PFOCK_STATUS_SUCCESS;
}

void load_local_bufD(PFock_t pfock)
{
    int lo[2];
    int hi[2];
	/*
    int *loadrow = pfock->loadrow;
    int *loadcol = pfock->loadcol;
    int sizerow = pfock->sizeloadrow;
    int sizecol = pfock->sizeloadcol;

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int ldD;    
    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    for (int i = 0; i < pfock->num_dmat2; i++) {
    #ifdef GA_NB
        ga_nbhdl_t nbnb;
    #endif
        // load local buffers
        double *D1;
        double *D2;
        double *D3;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Access(pfock->ga_D1[i], lo, hi, &D1, &ldD);
        hi[1] = pfock->sizeX2 - 1;
        NGA_Access(pfock->ga_D2[i], lo, hi, &D2, &ldD);
        hi[1] = pfock->sizeX3 - 1;
        NGA_Access(pfock->ga_D3[i], lo, hi, &D3, &ldD);
        int ldD1 = pfock->ldX1;
        int ldD2 = pfock->ldX2;
        int ldD3 = pfock->ldX3;     
        // update D1
        lo[0] = pfock->sfunc_row;
        hi[0] = pfock->efunc_row;
        for (int A = 0; A < sizerow; A++) {
            lo[1] = loadrow[PLEN * A + P_LO];
            hi[1] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
        #ifdef GA_NB
            NGA_NbGet(pfock->ga_D[i], lo, hi, &(D1[posrow]), &ldD1, &nbnb);
        #else
            NGA_Get(pfock->ga_D[i], lo, hi, &(D1[posrow]), &ldD1);
        #endif
        }
        // update D2
        lo[0] = pfock->sfunc_col;
        hi[0] = pfock->efunc_col;
        for (int B = 0; B < sizecol; B++) {
            lo[1] = loadcol[PLEN * B + P_LO];
            hi[1] = loadcol[PLEN * B + P_HI];
            int poscol = loadcol[PLEN * B + P_W];
        #ifdef GA_NB    
            NGA_NbGet(pfock->ga_D[i], lo, hi, &(D2[poscol]), &ldD2, &nbnb);
        #else
            NGA_Get(pfock->ga_D[i], lo, hi, &(D2[poscol]), &ldD2);
        #endif
        }
        // update D3
        for (int A = 0; A < sizerow; A++) {
            lo[0] = loadrow[PLEN * A + P_LO];
            hi[0] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
            for (int B = 0; B < sizecol; B++) {
                lo[1] = loadcol[PLEN * B + P_LO];
                hi[1] = loadcol[PLEN * B + P_HI];
                int poscol = loadcol[PLEN * B + P_W];
            #ifdef GA_NB
                NGA_NbGet(pfock->ga_D[i], lo, hi,
                          &(D3[posrow * ldD3 + poscol]), &ldD3, &nbnb);
            #else
                NGA_Get(pfock->ga_D[i], lo, hi,
                        &(D3[posrow * ldD3 + poscol]), &ldD3);        
            #endif
            }
        }
    #ifdef GA_NB
        NGA_NbWait (&nbnb);
    #endif
        // release update
        lo[0] = myrank;
        hi[0] = myrank;
        lo[1] = 0;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Release_update(pfock->ga_D1[i], lo, hi);
        hi[1] = pfock->sizeX2 - 1;
        NGA_Release_update(pfock->ga_D2[i], lo, hi);
        hi[1] = pfock->sizeX3 - 1;
        NGA_Release_update(pfock->ga_D3[i], lo, hi);
    }
    */
    
    // Load full density matrix, num_dmat2 should be 1
    int nbf = pfock->nbf;
    double *D_mat = pfock->D_mat;
    lo[0] = 0;
    lo[1] = 0;
    hi[0] = nbf - 1;
    hi[1] = nbf - 1;
    #ifdef GA_NB
    ga_nbhdl_t nbnb;
    NGA_NbGet(pfock->ga_D[0], lo, hi, D_mat, &nbf, &nbnb);
    NGA_NbWait (&nbnb);
    #else
    NGA_Get(pfock->ga_D[0], lo, hi, D_mat, &nbf);
    #endif
}

void store_local_bufF(PFock_t pfock)
{
    int *loadrow = pfock->loadrow;
    int *loadcol = pfock->loadcol;
    int sizerow = pfock->sizeloadrow;
    int sizecol = pfock->sizeloadcol;
    int myrank;
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    int lo[2];
    int hi[2];
    int ldF;
	
    int *ga_J = pfock->ga_F;
#ifdef __SCF__
    int *ga_K = pfock->ga_F;
#else
    int *ga_K = pfock->ga_K;
#endif
    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    for (int i = 0; i < pfock->num_dmat2; i++) {
    #ifdef GA_NB    
        ga_nbhdl_t nbnb;
    #endif
        // local buffers
        double *F1;
        double *F2;
        double *F3;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Access(pfock->ga_F1[i], lo, hi, &F1, &ldF);
        lo[1] = 0;
        hi[1] = pfock->sizeX2 - 1;
        NGA_Access(pfock->ga_F2[i], lo, hi, &F2, &ldF);
        lo[1] = 0;
        hi[1] = pfock->sizeX3 - 1;
        NGA_Access(pfock->ga_F3[i], lo, hi, &F3, &ldF);
        int ldF1 = pfock->ldX1;
        int ldF2 = pfock->ldX2;
        int ldF3 = pfock->ldX3;    

        // update F1
        double done = 1.0;
        lo[0] = pfock->sfunc_row;
        hi[0] = pfock->efunc_row;
        for (int A = 0; A < sizerow; A++) {
            lo[1] = loadrow[PLEN * A + P_LO];
            hi[1] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
        #ifdef GA_NB
            NGA_NbAcc(ga_J[i], lo, hi, &(F1[posrow]),
                      &ldF1, &done, &nbnb);    
        #else
            NGA_Acc(ga_J[i], lo, hi, &(F1[posrow]), &ldF1, &done);
        #endif
        }

        // update F2
        lo[0] = pfock->sfunc_col;
        hi[0] = pfock->efunc_col;
        for (int B = 0; B < sizecol; B++) {
            lo[1] = loadcol[PLEN * B + P_LO];
            hi[1] = loadcol[PLEN * B + P_HI];
            int poscol = loadcol[PLEN * B + P_W];
        #ifdef GA_NB
            NGA_NbAcc(ga_J[i], lo, hi, &(F2[poscol]),
                      &ldF2, &done, &nbnb);
        #else
            NGA_Acc(ga_J[i], lo, hi, &(F2[poscol]), &ldF2, &done);
        #endif
        }

        // update F3
        for (int A = 0; A < sizerow; A++) {
            lo[0] = loadrow[PLEN * A + P_LO];
            hi[0] = loadrow[PLEN * A + P_HI];
            int posrow = loadrow[PLEN * A + P_W];
            for (int B = 0; B < sizecol; B++) {
                lo[1] = loadcol[PLEN * B + P_LO];
                hi[1] = loadcol[PLEN * B + P_HI];
                int poscol = loadcol[PLEN * B + P_W];
            #ifdef GA_NB
                NGA_NbAcc(ga_K[i], lo, hi, 
                          &(F3[posrow * ldF3 + poscol]), &ldF3, &done, &nbnb);
            #else
                NGA_Acc(ga_K[i], lo, hi, 
                        &(F3[posrow * ldF3 + poscol]), &ldF3, &done);        
            #endif
            }
        }
    #ifdef GA_NB
        NGA_NbWait(&nbnb);
    #endif
        // update release
        lo[0] = myrank;
        hi[0] = myrank;
        lo[1] = 0;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Release(pfock->ga_F1[i], lo, hi);
        lo[1] = 0;
        hi[1] = pfock->sizeX2 - 1;
        NGA_Release(pfock->ga_F2[i], lo, hi);
        lo[1] = 0;
        hi[1] = pfock->sizeX3 - 1;
        NGA_Release(pfock->ga_F3[i], lo, hi);
    }
    GA_Sync();
    MPI_Barrier(MPI_COMM_WORLD);
}

int init_taskq(PFock_t pfock)
{
    int dims[2];
    int block[2];
    
    // create GA for dynamic scheduler
    int nprow = pfock->nprow;
    int npcol = pfock->npcol;
    dims[0] = nprow;
    dims[1] = npcol;
    block[0] = nprow;
    block[1] = npcol;    
    int *map = (int *)PFOCK_MALLOC(sizeof(int) * (nprow + npcol));
    if (NULL == map) {
        return -1;
    }    
    for (int i = 0; i < pfock->nprow; i++) {
        map[i] = i;
    }
    for (int i = 0; i < npcol; i++) {
        map[i + nprow] = i;
    }
    pfock->ga_taskid =
        NGA_Create_irreg(C_INT, 2, dims, "array taskid", block, map);
    if (0 == pfock->ga_taskid) {
        return -1;
    }
    PFOCK_FREE(map);
    
    return 0;
}


void clean_taskq(PFock_t pfock)
{
    GA_Destroy(pfock->ga_taskid);
}


void reset_taskq(PFock_t pfock)
{
    int izero = 0;    
    GA_Fill(pfock->ga_taskid, &izero);
}


int taskq_next(PFock_t pfock, int myrow, int mycol, int ntasks)
{
    int idx[2];

    idx[0] = myrow;
    idx[1] = mycol;
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);    
    int nxtask = NGA_Read_inc(pfock->ga_taskid, idx, ntasks);   
    gettimeofday(&tv2, NULL);    
    pfock->timenexttask += (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
    return nxtask;
}
