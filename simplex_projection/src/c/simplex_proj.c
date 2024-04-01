#include <stddef.h> //gives us size_t
#include <stdlib.h> //gives us qsort
#include <string.h> //gives us memcpy
#include <omp.h> //gives us multithreading

inline static int compare(const void* p, const void* q){
    //this function should return < 0 if p* goes before q*
    //> 0 if q* goes before p* and 0 if they're equal
    //to sort in decreasing order return 1 if p* < q*
    const double a = *(const double*)p;
    const double b = *(const double*)q;
    if (a > b) return -1; 
    return (a == b) ? 0: 1;
}

void simplex_proj(double* restrict arr,
    double* restrict u,
    double* restrict cumsum,
    const size_t m, const size_t n){
    //arr, restrict, and cssvs are arrays of size (m, n)
    //arr is the only one that needs populated; the
    //rest are workspaced
    //we want to project each row of arr
    //onto the probability simplex
    //first we declare variables and assign them
    size_t idx_cond; //the greatest idx satisfying
    //u - cssv/idx > 0
    
    //fill u with values of arr
    memcpy(u, arr, m*n*sizeof(double));

    //if PARALLEL include this pragma
    #ifdef PARALLEL
    #   pragma omp parallel for private(idx_cond)
    #endif
    for(size_t i=0; i<m; i++){ //perform the operation across rows
        const size_t row_shift = i*n; //go this deep into array based on row i
        //sort values of u in decreasing order
        qsort(u + row_shift, n, sizeof(double), compare);
        cumsum[row_shift+0] = u[row_shift+0];
        idx_cond = 0;
        for(size_t j=1; j<n; j++){
            cumsum[row_shift+j] = cumsum[row_shift+j-1] + u[row_shift+j]; //form the cumsum
            //the 1.0 is b/c we project onto ||x|| \leq 1.0
            //the j+1 is b/c we must convert the 0-indexed
            //j to 1-indexed
            //We don't have to check this case for j=0
            //because it is true (see our init of idx_cond)
            if ((u[row_shift+j] - (cumsum[row_shift+j] - 1.0)/((double)(j+1))) > 0){
                idx_cond = j;
            }
        }
        //the thresholding value. again here's the 1.0 b/c ||x|| \leq 1.0
        const double theta = (cumsum[row_shift+idx_cond] - 1.0)/((double)(idx_cond + 1));
        //make one final pass to do the thresholding
        for(size_t j = 0; j<n; j++){
            arr[row_shift+j] = (arr[row_shift+j] > theta) ? arr[row_shift+j] - theta : 0.0; 
        }
    }
}


