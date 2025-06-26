// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"

// Exchange the boundary values
void init_exchange(Field& field, const ParallelData parallel)
{
    // To up, from down
    double* sbuf_up = field.temperature.data(1, 0);
    double* rbuf_down  = field.temperature.data(field.nx + 1, 0);
    MPI_Isend(sbuf_up, field.ny + 2, MPI_DOUBLE,
	      parallel.nup, 11, MPI_COMM_WORLD, &parallel.requests[0]);
    MPI_Irecv(rbuf_down, field.ny + 2, MPI_DOUBLE,
	      parallel.ndown, 11, MPI_COMM_WORLD, &parallel.requests[1]);

    // Send to down, receive from up
    double* sbuf_down = field.temperature.data(field.nx, 0);
    double* rbuf_up = field.temperature.data();
    MPI_Isend(sbuf_down, field.ny + 2, MPI_DOUBLE,
              parallel.ndown, 12, MPI_COMM_WORLD, &parallel.requests[2]);
    MPI_Irecv(rbuf_up, field.ny + 2, MPI_DOUBLE,
              parallel.nup, 12, MPI_COMM_WORLD, &parallel.requests[3]);
}

void finalize_exchange(ParallelData& parallel)
{
    MPI_Waitall(4, parallel.requests, MPI_STATUSES_IGNORE);
}

// Update the temperature values using five-point stencil 
// for those not depending on ghost layers
void evolve_interior(Field& curr, const Field& prev, const double a, const double dt)
{
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Decrease the number of iterations in the outer loop
  // to avoid the ghost layers.
  for (int i = 2; i < curr.nx; i++) {
    for (int j = 1; j < curr.ny + 1; j++) {
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }
}

// Update the temperature on the boundaries
void evolve_boundaries(Field& curr, const Field& prev, const double a, const double dt)
{
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Update the temperature values in the upper boundary
  for (int j = 1; j < curr.ny + 1; j++) {
    int i = 1;
    curr(i, j) = prev(i, j) + a * dt * (
      ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
      ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
    );
  }
  // Lower boundary
  for (int j = 1; j < curr.ny + 1; j++) {
    int i = curr.nx;
    curr(i, j) = prev(i, j) + a * dt * (
      ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
      ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
    );
  }
}