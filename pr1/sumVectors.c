// Suma de vectores secuencial

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef int *vector;


// Function for generating random values for a vector
void LoadStartValuesIntoVectorRand(vector V, unsigned int n)
{
   unsigned int i;

   for (i=0;i<n;i++) 
     V[i] = (int)(random()%9);
}


// Function for printing a vector
void PrintVector(vector V, unsigned int n)
{
   unsigned int i;

   for (i=0;i<n;i++)
      printf("%3d",V[i]);
   printf("\n");
}

// Suma vectores C = A + B
void SumVectorSeq(vector A, vector B, vector C, unsigned int n)
{
   unsigned int k;
   
   for (k = 0; k <n ; k++)
      C[k] = A[k] + B[k];
}


// ------------------------
// MAIN function
// ------------------------
int main(int argc, char **argv)
{
   struct timeval start, stop;
   float timet;
   unsigned int n, size;

   if (argc == 2)
      n = atoi(argv[1]);
   else
     {
       printf ("Sintaxis: <ejecutable> <tamaño de los vectores>\n");
       exit(0);
     }

   srandom(12345);
   size = n * sizeof(int);

   // Define vectors at host
   vector A, B, C;

   gettimeofday(&start,0);

   // Load values into A
   A = (int *)malloc(size);
   LoadStartValuesIntoVectorRand(A,n);
   //printf("\nPrinting Vector A  %d\n",n);
   //PrintVector(A,n);

   // Load values 
   B = (int *)malloc(size);
   LoadStartValuesIntoVectorRand(B,n);
   //printf("\nPrinting Vector B  %d\n",n);
   //PrintVector(B,n);

   C = (int *)malloc(size);


   // execute the subprogram
   SumVectorSeq(A,B,C,n);

   //printf("\nPrinting vector C  %d\n",n);
   //PrintVector(C,n);

   // Free vectors
   free(A);
   free(B);
   free(C);

   gettimeofday(&stop,0);
   timet = (stop.tv_sec + stop.tv_usec * 1e-6)-(start.tv_sec + start.tv_usec * 1e-6);
   printf("Time = %f s\n",timet);

   printf("%f Gops/s \n",n/timet/1000000000);
   return 0;
}
