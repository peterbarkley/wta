#include <stddef.h> //gives us size_t
#include <stdlib.h> //gives us qsort
#include <string.h> //gives us memcpy

inline static int compare(const void* p, const void* q){
    const double a = *(const double*)p;
    const double b = *(const double*)q;
    if (a == b) return 0;
    return (a < b) ? 1: -1;
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
    double theta; //the thresholding value
    
    //fill u with values of arr
    memcpy(u, arr, m*n*sizeof(double));
  
    //we'll do this in serial for now 
    for(size_t i=0; i<m; i++){ //perform the operation across rows
        //sort values of u in decreasing order
        qsort(u + i*n, n, sizeof(double), compare);
        cumsum[n*i+0] = u[n*i+0];
        idx_cond = 0;
        for(size_t j=1; j<n; j++){
            cumsum[n*i+j] = cumsum[n*i+j-1] + u[n*i+j]; //form the cumsum
            //the 1.0 is b/c we project onto ||x|| \leq 1.0
            //the j+1 is b/c we must convert the 0-indexed
            //j to 1-indexed
            //We don't have to check this case for j=0
            //because it is true (see our init of idx_cond)
            if ((u[n*i+j] - (cumsum[n*i+j] - 1.0)/((double)(j+1))) > 0){
                idx_cond = j;
            }
        }
        //again here's the 1.0 b/c ||x|| \leq 1.0
        theta = (cumsum[n*i+idx_cond] - 1.0)/((double)(idx_cond + 1));
        //make one final pass to do the thresholding
        for(size_t j = 0; j<n; j++){
            arr[n*i+j] = (arr[n*i+j] > theta) ? arr[n*i+j] - theta : 0.0; 
        }
    }
}


