#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// #define NROWS
// #define NCOLS
#define MASTER 0

int calc_nrows_from_rank(int rank, int size, int nx);

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);

int calc_nrows_from_rank(int rank, int size, int nx)
{
  int nrows = nx / size;       /* integer division */
  if ((nx % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      nrows += nx % size;  /* add remainder to last rank */
  }
  return nrows;
}
int main(int argc, char* argv[])
{
  int ii,jj;             /* row and column indices for the grid */
  int rankings;                /* index for looping over ranks */
  int rank;              /* the rank of this process */
  int up;              /* the rank of the process to the left */
  int down;             /* the rank of the process to the right */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */
  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  up = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  down = (rank + 1) % size;
  if(rank== MASTER){
    up = MPI_PROC_NULL;
  }
  if(rank== size -1){
    down = MPI_PROC_NULL;
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;

  local_nrows = calc_nrows_from_rank(rank, size,nx);
  local_ncols = ny;
  if (local_nrows < 1) {
    fprintf(stderr,"Error: too many processes:- local_ncols < 1\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }


  // Allocate the image
  float* image = malloc(sizeof(double) * width * height);
  float* tmp_image = malloc(sizeof(double) * width * height);

  // double* sendbuf  = malloc(sizeof(double) * local_ncols);
  // double* recvbuf  = malloc(sizeof(double) * local_ncols);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);

  float* section = malloc(sizeof(double) * (local_nrows+2) * (local_ncols+2));
  float* tmp_section = malloc(sizeof(double) * (local_nrows+2) * (local_ncols+2));

  int part = nx/size;
  for(int ii=0;ii<local_nrows + 2;ii++) {
   for(int jj=0; jj<local_ncols + 2; jj++) {
     if (jj > 0 && jj < (local_ncols+1) && ii > 0 && ii < (local_nrows+1)){
          section[jj + ii * (local_ncols+2) ] = image[(jj + ii* width  + (rank*part*width))];
          tmp_section[jj + ii * (local_ncols+2)] = image[(jj + ii *width + (rank*part*width))];
          }
     else {
            section[jj + ii * (local_ncols+2) ] = 0;
            tmp_section[jj + ii * (local_ncols+2) ] = 0;
            }
 }
}
  MPI_Barrier(MPI_COMM_WORLD);
  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    MPI_Sendrecv(&section[(local_ncols+3)], local_ncols, MPI_FLOAT, up, tag, &section[(local_nrows+1) * (local_ncols+2) + 1], local_ncols, MPI_FLOAT, down, tag, MPI_COMM_WORLD, &status);
    MPI_Sendrecv(&section[(local_nrows) * (local_ncols+2) + 1], local_ncols, MPI_FLOAT, down, tag, &section[1], local_ncols, MPI_FLOAT, up, tag, MPI_COMM_WORLD, &status);
    stencil(local_nrows, local_ncols, width, height, section, tmp_section);
    MPI_Sendrecv(&tmp_section[(local_ncols+3)], local_ncols, MPI_FLOAT, up, tag, &tmp_section[(local_nrows+1) * (local_ncols+2) + 1], local_ncols, MPI_FLOAT, down, tag, MPI_COMM_WORLD, &status);
    MPI_Sendrecv(&tmp_section[(local_nrows) * (local_ncols+2) + 1], local_ncols, MPI_FLOAT, down, tag, &tmp_section[1], local_ncols, MPI_FLOAT, up, tag, MPI_COMM_WORLD, &status);
    stencil(local_nrows, local_ncols, width, height, tmp_section, section);
  }
  double toc = wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == MASTER){
    for(int i=1;i<local_nrows+1;i++){
      for(int j=1;j<local_ncols+1;j++){
        image[j + (i*width)] = section[j + i *(local_ncols + 2) ];
      }
    }
    for(int kk = 1;kk<size;kk++){
      int nrows = calc_nrows_from_rank(kk,size,nx);
      for(int i =1;i<nrows+1;i++){
        MPI_Recv(&image[ ((kk * local_nrows) + i) * width + 1], local_ncols, MPI_FLOAT, kk, tag, MPI_COMM_WORLD, &status);
      }
    }

  }
  else{
  for(int i =1;i<local_nrows + 1;i++){
        MPI_Send(&section[i * (local_ncols + 2) + 1], local_ncols, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
      }
  }
  // Output
if(rank == MASTER ){
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc - tic);
  printf("------------------------------------\n");
  output_image(OUTPUT_FILE, nx, ny, width, height, image);
  }
  MPI_Finalize();
  free(image);
  free(tmp_image);
  free(section);
  free(tmp_section);
}


void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  for (int i = 1; i < nx + 1; ++i) {
    for (int j = 1; j < ny + 1; ++j) {
	int x = j+i*(ny + 2);
	tmp_image[x]=(image[x]*0.6f)+((image[x-1]+image[x+1]+image[x+(ny + 2)]+image[x-(ny + 2)])*0.1f);
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

