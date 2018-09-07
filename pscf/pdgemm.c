#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <mkl.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

#include "pdgemm.h"

static void copyMat(int m, int n, double *From, int ldfrom, double *To, int ldto)
{
    #pragma omp parallel for
    for (int r = 0; r < m; r++)
        memcpy(To + r * ldto, From + r * ldfrom, sizeof(double) * n);
}


void ReduceTo2D (int myrow, int mycol, int mygrd,
                 int nrows, int ncols, double *S, int ghost, MPI_Comm comm_3D)
{
    if (mygrd != ghost)
    {
        if (mycol == mygrd)
        {
            int coords[3] = { myrow, mycol, 0 }, to;
            MPI_Cart_rank (comm_3D, coords, &to);
            MPI_Send (&S[0], nrows * ncols, MPI_DOUBLE, to, 0, comm_3D);
        }
    }
    else
    {
        if (mycol)
        {
            MPI_Status status;
            int coords[3] = { myrow, mycol, mycol }, from;
            MPI_Cart_rank (comm_3D, coords, &from);
            MPI_Recv (S, nrows * ncols, MPI_DOUBLE, from, 0, comm_3D, &status);
        }
    }

}

#define N_DUP 4
MPI_Comm    comm_rows[N_DUP];
MPI_Status  status[N_DUP];
MPI_Request reqs[N_DUP];
int spos[N_DUP + 1], blklen[N_DUP];
int comm_dupped = 0;

static int block_low(int n, int bid, int nb)
{
    long long k = bid;
    k *= (long long) n;
    k /= nb;
    return (int)(k);
}

static void dup_comm_row(MPI_Comm comm_row, int blksize)
{
    if (comm_dupped == 1) return;
    comm_dupped = 1;
    spos[0] = 0;
    for (int i = 0; i < N_DUP; i++)
    {
        MPI_Comm_dup(comm_row, &comm_rows[i]);
        spos[i + 1] = block_low(blksize, i + 1, N_DUP);
        blklen[i]   = spos[i + 1] - spos[i];
    }
}

// Used by McWeeny purification only, input D, output D^2 and D^3
// All MPI_Comms used in this function are fixed, so we can duplicate 
// comm_row to further utilized the network bandwidth
int pdgemm3D(int myrow, int mycol, int mygrd,
             MPI_Comm comm_row, MPI_Comm comm_col,
             MPI_Comm comm_grd, MPI_Comm comm_3D,
             int *nr, int *nc,
             int nrows, int ncols,
             double *D_, double *D2_, double *D3_,
             tmpbuf_t *tmpbuf, double *dgemm_time)
{
    struct timeval tv1, tv2;
    int ncols0 = nc[0], nrows0 = nr[0];
    if (dgemm_time != NULL) *dgemm_time = 0.0;
    assert(nrows == nr[myrow] && ncols == nc[mycol]);
    assert(ncols0 >= ncols && nrows0 >= nrows);
    
    dup_comm_row(comm_row, ncols0 * nrows0);

    double *A = tmpbuf->A;
    double *S = tmpbuf->S;
    double *C = tmpbuf->C;

    memset(A, 0, sizeof(double) * nrows0 * ncols0);

    copyMat(nrows, ncols, D_, ncols, A, ncols0);

    {
        // new number of rows and columns
        int nrows = nrows0;
        int ncols = ncols0;

        double *A_i = tmpbuf->A_i;
        double *S_i = tmpbuf->S_i;
        double *C_i = tmpbuf->C_i;

        // 1. Replicate D matrix on all planes
        MPI_Bcast(A, nrows * ncols, MPI_DOUBLE, 0, comm_grd);

        // 2.1. Broadcast A_i row
        if (myrow == mygrd) copyMat(nrows, ncols, A, ncols, &A_i[0], ncols);
        MPI_Bcast(&A_i[0], nrows * ncols, MPI_DOUBLE, mygrd, comm_col);

        // 2.2. Do local dgemm
        gettimeofday(&tv1, NULL);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ncols, ncols,
                    ncols, 1.0, A, ncols, &A_i[0], ncols, 0.0, &S_i[0], ncols);
        gettimeofday(&tv2, NULL);
        if (dgemm_time != NULL) {
            *dgemm_time += (tv2.tv_sec - tv1.tv_sec) +
                           (tv2.tv_usec - tv1.tv_usec) / 1000.0 / 1000.0;
        }

        // 2.3. reduce S_i into a column i on plane i
        //MPI_Reduce(&S_i[0], S, nrows * ncols, MPI_DOUBLE, MPI_SUM, mygrd, comm_row);
        for (int i = 0; i < N_DUP; i++)
            MPI_Ireduce(&S_i[spos[i]], S + spos[i], blklen[i], MPI_DOUBLE, MPI_SUM, mygrd, comm_rows[i], &reqs[i]);
        MPI_Waitall(N_DUP, &reqs[0], &status[0]);

        // 2.4. Reduce S to plane 0
        ReduceTo2D(myrow, mycol, mygrd, nrows, ncols, S, 0, comm_3D);

        // 3.1 Transpose S_i on each plane to prepare for computing C
        copyMat(nrows, ncols, &S[0], ncols, &S_i[0], ncols);
        if (mycol == mygrd && myrow != mycol) {
            int coords[3] = { mycol, myrow, mygrd }, to;
            MPI_Cart_rank(comm_3D, coords, &to);
            // printf("(%d %d %d) sending to (%d %d %d)\n"
            // myrow, mycol, mygrd, mycol, myrow, mygrd);
            MPI_Send(&S_i[0], nrows * ncols, MPI_DOUBLE, to, 0, comm_3D);
        }
        if (myrow == mygrd && myrow != mycol) {
            MPI_Status status;
            int coords[3] = { mycol, myrow, mygrd }, from;
            MPI_Cart_rank (comm_3D, coords, &from);
            // printf("(%d %d %d) receiving from (%d %d %d)\n",
            // myrow, mycol, mygrd, mycol, myrow, mygrd);
            MPI_Recv(&S_i[0], nrows * ncols, MPI_DOUBLE,
                     from, 0, comm_3D, &status);
        }

        // 3.2. Broadcast S_i
        MPI_Bcast(&S_i[0], nrows * ncols, MPI_DOUBLE, mygrd, comm_col);

        // 3.3. C_i=A*S_i
        gettimeofday(&tv1, NULL);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ncols, ncols,
                    ncols, 1.0, A, ncols, &S_i[0], ncols, 0.0, &C_i[0], ncols);
        gettimeofday(&tv2, NULL);
        if (dgemm_time != NULL) {
            *dgemm_time += (tv2.tv_sec - tv1.tv_sec) +
                           (tv2.tv_usec - tv1.tv_usec) / 1000.0 / 1000.0;
        }
        
        // 3.4. Reduce C_i into a column on plane i
        //MPI_Reduce(&C_i[0], C, nrows * ncols, MPI_DOUBLE, MPI_SUM, mygrd, comm_row);
        for (int i = 0; i < N_DUP; i++)
            MPI_Ireduce(&C_i[spos[i]], C + spos[i], blklen[i], MPI_DOUBLE, MPI_SUM, mygrd, comm_rows[i], &reqs[i]);
        MPI_Waitall(N_DUP, &reqs[0], &status[0]);

        // 3.5. Reduce C to plane 0
        ReduceTo2D(myrow, mycol, mygrd, nrows, ncols, C, 0, comm_3D);
    }

    copyMat(nrows, ncols, S, ncols0, D2_, ncols);
    copyMat(nrows, ncols, C, ncols0, D3_, ncols);

    return 0;
}

// True parallel dgemm, input A, B return C := A * B
void pdgemm3D_2(int myrow, int mycol, int mygrd,
                MPI_Comm comm_row, MPI_Comm comm_col,
                MPI_Comm comm_grd, MPI_Comm comm_3D,
                int *nr, int *nc, int nrows, int ncols,
                double *A_block_, double *B_block_,
                double *C_block_, tmpbuf_t *tmpbuf, double *dgemm_time)
{
    struct timeval tv1, tv2;
    int ncols0 = nc[0], nrows0 = nr[0];
    if (dgemm_time != NULL) *dgemm_time = 0.0;
    assert(nrows == nr[myrow] && ncols == nc[mycol]);
    assert(ncols0 >= ncols && nrows0 >= nrows);

    double *A_block = tmpbuf->A;
    double *B_block = tmpbuf->C;
    double *C_block = tmpbuf->S;
    
    memset(A_block, 0, sizeof (double) * nrows0 * ncols0);
    memset(B_block, 0, sizeof (double) * nrows0 * ncols0);
    copyMat(nrows, ncols, A_block_, ncols, A_block, ncols0);
    copyMat(nrows, ncols, B_block_, ncols, B_block, ncols0);

    {
        int nrows = nrows0;
        int ncols = ncols0;
        double *B_block_copy = tmpbuf->A_i;
        double *C_i = tmpbuf->C_i;

        MPI_Bcast(A_block, nrows * ncols, MPI_DOUBLE, 0, comm_grd);

        // for matrix B at grid 0, send row i (except row 0) to grid i
        if (mygrd == 0 && myrow != 0) {
            int coords[3] = { myrow, mycol, myrow }, to;
            MPI_Cart_rank(comm_3D, coords, &to);
            MPI_Send(&B_block[0], nrows * ncols, MPI_DOUBLE, to, 0, comm_3D);
        }
        if (mygrd && myrow == mygrd) {
            MPI_Status status;
            int coords[3] = { myrow, mycol, 0 }, from;
            MPI_Cart_rank(comm_3D, coords, &from);
            MPI_Recv(&B_block[0], nrows * ncols, MPI_DOUBLE, from, 0,
                     comm_3D, &status);
        }
        // spread / bcast the row block of B_block on each grid 
        copyMat(nrows, ncols, &B_block[0], ncols, &B_block_copy[0], ncols);
        MPI_Bcast(&B_block_copy[0], nrows * ncols, MPI_DOUBLE, mygrd, comm_col);

        // do local dgemm
        gettimeofday(&tv1, NULL);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ncols, ncols,
                    ncols, 1.0, A_block, ncols, &B_block_copy[0], ncols, 0.0,
                    &C_i[0], ncols);
        gettimeofday(&tv2, NULL);
        if (dgemm_time != NULL) {
            *dgemm_time += (tv2.tv_sec - tv1.tv_sec) +
                           (tv2.tv_usec - tv1.tv_usec) / 1000.0 / 1000.0;
        }

        MPI_Reduce(&C_i[0], C_block, nrows * ncols, MPI_DOUBLE, MPI_SUM, mygrd, comm_row);

        ReduceTo2D(myrow, mycol, mygrd, nrows, ncols, C_block, 0, comm_3D);
    }

    copyMat(nrows, ncols, C_block, ncols0, C_block_, ncols);
}


void allocate_tmpbuf (int nrows, int ncols, int *nr, int *nc,
                      tmpbuf_t * tmpbuf)
{
    int ncols0 = nc[0], nrows0 = nr[0];
    assert (ncols0 >= ncols && nrows0 >= nrows);

    int block_size_align64b = (nrows0 * ncols0 + 7) / 8 * 8;
    tmpbuf->A   = (double *) _mm_malloc(sizeof(double) * block_size_align64b * 6, 64);
    tmpbuf->S   = tmpbuf->A   + block_size_align64b;
    tmpbuf->C   = tmpbuf->S   + block_size_align64b;
    tmpbuf->A_i = tmpbuf->C   + block_size_align64b;
    tmpbuf->S_i = tmpbuf->A_i + block_size_align64b;
    tmpbuf->C_i = tmpbuf->S_i + block_size_align64b;

    #pragma omp parallel for schedule(static)
    #pragma simd
    for (int i = 0; i < nrows0 * ncols0; i++)
    {
        tmpbuf->A[i] = 0;
        tmpbuf->C[i] = 0;
        tmpbuf->S[i] = 0;
        tmpbuf->A_i[i] = 0;
        tmpbuf->C_i[i] = 0;
        tmpbuf->S_i[i] = 0;
    }
}


void dealloc_tmpbuf (tmpbuf_t * tmpbuf)
{
    _mm_free(tmpbuf->A);
}
