#ifndef __FOCK_TASK_H__
#define __FOCK_TASK_H__


#include "CInt.h"

void init_block_buf(int _nbf, int _nshells, int *f_startind, int num_dmat, BasisSet_t basis, int _maxcolfuncs);

void fock_task(BasisSet_t basis, SIMINT_t simint, int ncpu_f, int num_dmat,
               int *shellptr, double *shellvalue,
               int *shellid, int *shellrid, int *f_startind,
               int *rowpos, int *colpos, int *rowptr, int *colptr,
               double tolscr2, int startrow, int startcol,
               int startM, int endM, int startP, int endP,
               double *D_mat, 
               double *F1, double *F2, double *F3,
               int ldX1, int ldX2, int ldX3,
               int ldX4, int ldX5, int ldX6,
               int sizeX1, int sizeX2, int sizeX3,
               int sizeX4, int sizeX5, int sizeX6,
               double *nitl, double *nsq, 
               int _nbf, int _nshells, int repack_D);

void reset_F(int numF, int num_dmat, double *F1, double *F2, double *F3, int sizeX1, int sizeX2, int sizeX3);

void reduce_F(double *F1, double *F2, double *F3, int maxrowsize, int maxcolsize, int ldX3, int ldX4, int ldX5, int ldX6);


#endif /* #define __FOCK_TASK_H__ */
