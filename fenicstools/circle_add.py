from dolfin import *
import numpy as np
from mpi4py import MPI


mesh = UnitSquareMesh(20, 20)
comm = mesh.mpi_comm().tompi4py()

nparticles = 200
particles = 1.2*np.random.rand(nparticles, 2)

tree = mesh.bounding_box_tree()
lim = mesh.topology().size_global(2)

found = []
not_found_local = []
for p in particles:
    index = tree.compute_first_collision(Point(*p))
    if index < lim: 
        not_found_local.append(p)
    else:
        found.append(p)
my_count = len(found)

count = len(not_found_local)
count_global = comm.allgather(count)

world_size = comm.size
world_rank = comm.rank

next_rank = (world_rank + 1) % world_size
prev_rank = (world_rank + world_size - 1) % world_size

loop = 0
while max(count_global) > 0 and loop < world_size:
    loop += 1

    # Send to next
    comm.Send(np.array(not_found_local).flatten(), next_rank, world_rank)

    # Receive particles from previous
    received = np.zeros(count_global[prev_rank]*2, dtype=float)
    comm.Recv(received, prev_rank, prev_rank)
    

    # Work with received
    received = received.reshape((-1, 2))
    not_found_local = []
    for p in received:
        index = tree.compute_first_collision(Point(*p))
        if index < lim: 
            not_found_local.append(p)
        else:
            found.append(p)

    count = len(not_found_local)
    count_global = comm.allgather(count)

dist = comm.allgather(len(found))
total_found = sum(dist)
print total_found, nparticles*world_size
assert total_found == nparticles*world_size

if world_rank == 0:
    print dist
    # print found
