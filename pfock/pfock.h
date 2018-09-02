#ifndef __PFOCK_H__
#define __PFOCK_H__


#include <omp.h>
#include "CInt.h"

#include "Buzz_Matrix.h"
#include "Buzz_Task_Queue.h"

/** 
 * @struct  PFock
 * @brief   PFock computing engine.
 */
struct PFock {
    BasisSet_t basis;

    int input_type;
    int nthreads;
    int max_numdmat;
    int num_dmat;
    int max_numdmat2;
    int num_dmat2;
    int nosymm;
    
    // screening
    int nnz;
    double *shellvalue;
    int *shellptr;
    int *shellid;
    int *shellrid;
    double maxvalue;
    double tolscr;
    double tolscr2;

    // problem parameters
    int nbf;
    int nshells;
    int natoms;
    int maxnfuncs;

    int nprocs;
    int nprow; // np per row    
    int npcol; // np per col
	
    // task size
    int nbp_p;
    int nbp_row;
    int nbp_col;
	// local number of taskes
    int ntasks;
	
    // pointers to shells
    int *blkrowptr_sh;
    int *blkcolptr_sh;
    
    // pointers to blks, shells and functions
    int *rowptr_blk;
    int *colptr_blk;
    int *rowptr_sh;
    int *colptr_sh;
    int *rowptr_f;
    int *colptr_f;
	
    // blks
    int sblk_row;
    int eblk_row;
    int sblk_col;
    int eblk_col;
    int nblks_row;
    int nblks_col;
	
    // shells
    int sshell_row;
    int sshell_col;
    int eshell_row;
    int eshell_col;
    int nshells_row;
    int nshells_col;
	
    // functions
    int sfunc_row;
    int sfunc_col;
    int efunc_row;
    int efunc_col;
    int nfuncs_row;
    int nfuncs_col;
    int sizemyrow;
    int sizemycol;

    // integrals
    SIMINT_t simint;
    int *f_startind;
    int *s_startind;

    // arrays and buffers
    int *rowptr;
    int *colptr;
    int *rowpos;
    int *colpos;
    int *rowsize;
    int *colsize;
    int *loadrow;
    int *loadcol;
	int sizeloadrow;
    int sizeloadcol;
    
    double *D_mat;

    // global arrays
    int *ga_D;
    int ga_X;

    double *FT_block;

    // buf D and F
    int maxrowfuncs;
    int maxcolfuncs;
    int maxrowsize;
    int maxcolsize;
    int sizeX1;
    int sizeX2;
    int sizeX3;
    int sizeX4;
    int sizeX5;
    int sizeX6;
    int ldX1;
    int ldX2;
    int ldX3;
    int ldX4;
    int ldX5;
    int ldX6;
    double *F1;
    double *F2;
    double *F3;
    int numF;
    int ncpu_f;

    int getFockMatBufSize;
    double *getFockMatBuf;
	Buzz_Matrix_t bm_Hmat;
	Buzz_Matrix_t bm_Xmat;
	Buzz_Matrix_t bm_Smat;
	Buzz_Matrix_t bm_tmp1;
	Buzz_Matrix_t bm_tmp2;
    Buzz_Matrix_t bm_Dmat;      // Global density matrix
    Buzz_Matrix_t bm_Fmat;      // Global Coulomb matrix & Fock matrix
    Buzz_Matrix_t bm_Kmat;      // Global exchange matrix
    Buzz_Matrix_t bm_F1;        // Each process's buffer for its J_{MN}
    Buzz_Matrix_t bm_F2;        // Each process's buffer for its J_{PQ}
    Buzz_Matrix_t bm_F3;        // Each process's buffer for its K_{MP, NP, MQ, NQ}
	Buzz_Matrix_t bm_screening; // Screening values
    
    Buzz_Task_Queue_t task_queue;

    // statistics
    double mem_cpu;
	double timepass;
	double timereduce;
	double timeinit;
	double timegather;
	double timescatter;
	double timecomp;
	double usq;
	double uitl;
	double steals;
	double stealfrom;
	double ngacalls;
	double volumega;
	double timenexttask;
    double *mpi_timepass;
    double *mpi_timereduce;
    double *mpi_timeinit;
    double *mpi_timegather;
    double *mpi_timescatter;    
    double *mpi_timecomp;
    double *mpi_usq;
    double *mpi_uitl;
    double *mpi_steals;
    double *mpi_stealfrom;
    double *mpi_ngacalls;
    double *mpi_volumega;
    double *mpi_timenexttask;
};


typedef struct PFock *PFock_t;


/** 
 * @enum   PFockStatus_t
 * @brief  Defines the return status of GTFock functions.
 */
typedef enum
{
    /// Successful
    PFOCK_STATUS_SUCCESS = 0,
    /// Initialization errors
    PFOCK_STATUS_INIT_FAILED = 1,
    /// Memory allocation errors
    PFOCK_STATUS_ALLOC_FAILED = 2,
    /// Invalid values
    PFOCK_STATUS_INVALID_VALUE = 3,   
    /// Runtime errors
    PFOCK_STATUS_EXECUTION_FAILED = 4,
    /// Internal errors
    PFOCK_STATUS_INTERNAL_ERROR = 5
} PFockStatus_t;


/** 
 * @enum   PFockMatType_t
 * @brief  Defines the matrix types.
 */
typedef enum
{
    /// Densitry matrix
    PFOCK_MAT_TYPE_D = 0,
    /// Fock matrix   
    PFOCK_MAT_TYPE_F = 1,
    /// Coulomb matrix
    PFOCK_MAT_TYPE_J = 2,
    /// Exchange matrix
    PFOCK_MAT_TYPE_K = 3   
} PFockMatType_t;


/** 
 * @brief  Create a PFock compute engine. 
 *
 * This function must be called by all processes "loosely synchronously".
 *
 * @param[in] basis        the pointer to the BasisSet_t
 * @param[in] nprow        the number of processes per row 
 *                         (process topology is nprow by npcol)
 * @param[in] npcol        the number of processes per column
 *                         (process topology is nprow by npcol)
 * @param[in] ntasks       the number of tasks
 *                         (ntasks x ntasks tasks per process)
 * @param[in] tolscr       the screening threshold
 * @param[in] max_numdmat  the maximum number of density matrices
 * @param[in] symm         whether the density matrices are symmetric or not  
 * @param[in] pfock        the pointer to the PFock_t compute engine returned
 *
 * @return    the function return status
 */
PFockStatus_t PFock_create(BasisSet_t basis, int nprow, int npcol, int ntasks,
                           double tolscr, int max_numdmat, int symm,
                           PFock_t *_pfock);

/** 
 * @brief  Destroy the PFock compute engine. 
 *
 * @param[in] pfock  the pointer to the PFock_t compute engine
 *
 * @return    the function return status
 */
PFockStatus_t PFock_destroy(PFock_t pfock);


/**
 * @brief  Get the data from a specific global density matrix.
 *
 * @param[in] pfock     the pointer to the PFock_t compute engine
 * @param[in] type      the type of the matrix
 * @param[in] index     the index of the density matrix
 * @param[in] rowstart  the starting row index of the
 *                      global density matrix section
 * @param[in] rowend    the ending row index of the
 *                      global density matrix section
 * @param[in] colstart  the starting column index of the
 *                      global density matrix section
 * @param[in] colend    the ending column index of the
 *                      global density matrix section
 * @param[in] stride    the leading dimension of the local data
 * @param[out] mat      the pointer to the local data
 *
 * @return    the function return status  
 */
PFockStatus_t PFock_getMat(PFock_t pfock,
                           PFockMatType_t type,
                           int index,
                           int rowstart,
                           int rowend,
                           int colstart,
                           int colend,
                           int stride,
                           double *mat);

/**
 * @brief  Computes all J and K matrices
 *
 * This function is called loosely synchronously by all processes
 * to compute all J and K matrices
 *
 * @param[in] basis  the pointer to the BasisSet_t
 * @param[in] pfock  the pointer to the PFock_t compute engine
 *
 * @return    the function return status
 */
PFockStatus_t PFock_computeFock(BasisSet_t basis,
                                PFock_t pfock);

/**
 * @brief  Creates a Core Hamilton matrix
 *
 * This function is called loosely synchronously by all processes
 * to compute all J and K matrices
 *
 * @param[in] pfock  the pointer to the PFock_t compute engine
 * @param[in] basis  the pointer to the BasisSet_t
 *
 * @return    the function return status 
 */
PFockStatus_t PFock_createCoreHMat(PFock_t pfock, BasisSet_t basis);

/**
 * @brief  Destory the Core Hamilton matrix
 *
 * This function is called loosely synchronously by all processes.
 *
 * @param[in] pfock  the pointer to the PFock_t compute engine
 *
 * @return    the function return status
 */    
PFockStatus_t PFock_destroyCoreHMat(PFock_t pfock);

/**
 * @brief  Returns a block of the Core Hamilton matrix
 *
 * @param[in] pfock      the pointer to the PFock_t compute engine
 * @param[out] rowstart  the pointer to the starting row index of the local data
 * @param[out] rowend    the pointer to the ending row index of the local data
 * @param[out] colstart  the pointer to the starting column index
 *                       of the local data
 * @param[out] colend    the pointer to the ending column index
 *                       of the local data
 * @param[out] stride    the pointer to the leading dimension of the local data
 * @param[out] mat       the pointer to the local data
 *
 * @return    the function return status   
 */
PFockStatus_t PFock_getCoreHMat(PFock_t pfock, int rowstart, int rowend,
                                int colstart, int colend,
                                int stride, double *mat);

/**
 * @brief  Creates a Overlap matrix and its square root
 *
 * This function is called loosely synchronously by all processes.
 *
 * @param[in] pfock  the pointer to the PFock_t compute engine
 * @param[in] basis  the pointer to the BasisSet_t
 *
 * @return    the function return status 
 */
PFockStatus_t PFock_createOvlMat(PFock_t pfock, BasisSet_t basis);

/**
 * @brief  Destory the Overlap matrix
 *
 * This function is called loosely synchronously by all processes.
 *
 * @param[in] pfock  the pointer to the PFock_t compute engine
 *
 * @return    the function return status
 */  
PFockStatus_t PFock_destroyOvlMat(PFock_t pfock);

/**
 * @brief  Returns a block of the Overlap matrix
 *
 * @param[in] pfock      the pointer to the PFock_t compute engine
 * @param[out] rowstart  the pointer to the starting row index of the local data
 * @param[out] rowend    the pointer to the ending row index of the local data
 * @param[out] colstart  the pointer to the starting column index
 *                       of the local data
 * @param[out] colend    the pointer to the ending column index
 *                       of the local data
 * @param[out] stride    the pointer to the leading dimension of the local data
 * @param[out] mat       the pointer to the local data
 *
 * @return    the function return status   
 */                                      
PFockStatus_t PFock_getOvlMat(PFock_t pfock, int rowstart, int rowend,
                              int colstart, int colend,
                              int stride, double *mat);

/**
 * @brief  Returns a block of the square root (X) of the Overlap matrix
 *
 * @param[in] pfock      the pointer to the PFock_t compute engine
 * @param[out] rowstart  the pointer to the starting row index of the local data
 * @param[out] rowend    the pointer to the ending row index of the local data
 * @param[out] colstart  the pointer to the starting column index
 *                       of the local data
 * @param[out] colend    the pointer to the ending column index
 *                       of the local data
 * @param[out] stride    the pointer to the leading dimension of the local data
 * @param[out] mat       the pointer to the local data
 *
 * @return    the function return status   
 */  
PFockStatus_t PFock_getOvlMat2(PFock_t pfock, int rowstart, int rowend,
                               int colstart, int colend,
                               int stride, double *mat);

/**
 * @brief  Returns the memory usage of the PFock computing engine
 *
 * @param[in] pfock     the pointer to the PFock_t compute engine
 * @param[out] mem_cpu  the pointer to the memory usage
 *
 * @return    the function return status
 */ 
PFockStatus_t PFock_getMemorySize(PFock_t pfock, double *mem_cpu);

/**
 * @brief  Prints the performance results of the PFock computing engine
 *
 * @param[in] pfock     the pointer to the PFock_t compute engine
 *
 * @return    the function return status
 */
PFockStatus_t PFock_getStatistics(PFock_t pfock);

/**
 * @brief  Get a block from global Fock (Coulomb, exchange) matrix.
 *
 * @param[in] pfock     the pointer to the PFock_t compute engine
 * @param[in] rowstart  the starting row index of the
 *                      global density matrix section
 * @param[in] rowend    the ending row index of the
 *                      global density matrix section
 * @param[in] colstart  the starting column index of the
 *                      global density matrix section
 * @param[in] colend    the ending column index of the
 *                      global density matrix section
 * @param[in] stride    the leading dimension of the local data
 * @param[out] mat      the pointer to the local data
 */
void PFock_Buzz_getFockMat(
    PFock_t pfock,
    int rowstart, int rowend,
    int colstart, int colend,
    int stride,   double *mat
);

#endif /* #ifndef __PFOCK_H__ */
