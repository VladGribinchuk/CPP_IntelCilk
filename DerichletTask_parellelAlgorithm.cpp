//// Example of numerical solution of the Dirichlet problem 
//// for the Poisson equation using iterative method
////
//// As a domain of the jobs D some function U will be used 
//// by the unit square 
////	D = (x in [0..1], y in [0..1])
////
//// Boundary conditions:
////	U(0,y)=0 U(1,y)=y^2
////	U(x,0)=0 U(x,1)=x^2
////
//// Follow code implement parallel algorithm using Intel Cilk Plus

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <cilk/cilk_api.h>

// Lenght side of a square D will be A
const int A = 1;

double** initMatrix(int ny, int nx)
{
	// matrix will be consist of ny+1 rows and nx+1 cols
	double** matr = new double*[ny + 1];
	for (int i = 0; i <= ny; ++i)
		matr[i] = new double[nx + 1];

	// h - a step along the x and y axis
	double hx = A / (double)nx;
	double hy = A / (double)ny;
	double x = 0;
	double y = 0;
	for (int i = 0; i <= ny; ++i){
		y = i*hy;
		for (int j = 0; j <= nx; ++j){
			x = j*hx;
			matr[i][j] = 0.;
			if (i == ny)
				matr[i][j] = x*x;
			if (j == nx)
				matr[i][j] = y*y;
		}
	}
	return matr;
}

void cleanupMatrix(double** matrix, int ny)
{
	// Remove and clear all the allocated memory
	for (int i = 0; i <= ny; ++i){
		delete[]matrix[i];
		matrix[i] = NULL;
	}
	delete matrix;
	matrix = NULL;
}

void outputMatrix(double** matrix, int ny, int nx)
{
	for (int i = 0; i <= ny; ++i){
		for (int j = 0; j <= nx; ++j)
			printf("%.3f ", matrix[i][j]);
		printf("\n");
	}
}

int main()
{
	//
	// Graininess of x and y axis
	int N = 500;

	//
	// Initialization of two computing matrix,
	//	that will be consist of actual and previous 
	//	function values per each iterations
	double** matrix = initMatrix(N, N);
	double** matrix2 = initMatrix(N, N);

	double dmax;
	double e = .001;
	int it = 0;

	//
	// signal of event out of limit precision
	//	if dm=0 required precision achieved
	cilk::reducer_opadd<int> dm(0);

	printf("count workers=%d\n", __cilkrts_get_nworkers());
	clock_t beginTime = clock();
	do
	{
		it++;
		dm.set_value(0);

		cilk_for(int i = 1; i < N; ++i){
			for (int j = 1; j < N; ++j){
				matrix2[i][j] = 0.25*(matrix[i][j - 1] + matrix[i][j + 1] + matrix[i - 1][j] + matrix[i + 1][j]);
				double diff = fabs(mat2[i][j] - matrix[i][j]);
				if (e < diff) dm += 1;
			}
			
			for (int j = 1; j < N; ++j)
				matrix[i][j] = matrix2[i][j];
			
		}
	} while (dm.get_value() > 0);
	clock_t endTime = clock();

	printf("itaretion=%d\ntime: %dms\n", it, endTime - beginTime);
	//printf("result:\n");
	//outputMatrix(matrix, Ny, Nx);

	cleanupMatrix(matrix, N);

	getchar();
	return EXIT_SUCCESS;
}