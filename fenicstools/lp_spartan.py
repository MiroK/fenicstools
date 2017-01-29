import dolfin as df
import numpy as np


def counter():
    c = 0
    while True: 
        yield c
        c += 1

class CellWithParticles(df.Cell):
    '''TODO'''
    def __init__(self, lp_collection, cell_id, particle=None):
        '''TODO'''
        mesh = lp_collection.mesh

        df.Cell.__init__(self, mesh, cell_id)

        self.particles = set([])
        if particle is not None: self + particle

        tdim = lp_collection.dim
        neighbors = sum((vertex.entities(tdim).tolist() for vertex in df.vertices(self)), [])
        neighbors = set(neighbors) - set([cell_id])   # Remove self
        self.neighbors = map(lambda neighbor_index: df.Cell(mesh, neighbor_index), neighbors)

    def __add__(self, particle): self.particles.add(particle)

    def __len__(self): return len(self.particles)

    def __bool__(self): return len(self) > 0


class LPCollection(object):
    '''TODO'''
    def __init__(self, V, property_layout):
        '''TODO'''
        mesh = V.mesh()
        self.dim = mesh.geometry().dim()
        assert self.dim == mesh.topology().dim()
        self.mesh = mesh

        self.tree = mesh.bounding_box_tree()
        self.lim = mesh.topology().size_global(self.dim)

        # NOTE: property layout here is the map which maps property name to
        # length of a vector that represents property value
        assert all(v > 0 for v in property_layout.values())
        offsets = [0, self.mesh.geometry().dim()] + [property_layout[k] for k in property_layout]
        self.psize = sum(offsets)
        self.offsets = np.cumsum(offsets)
        self.keys = ['x'] + property_layout.keys()

        self.particles = {}
        self.cells = {}
        self.ticket = counter()

        comm = mesh.mpi_comm().tompi4py()
        assert comm.size == 1 or comm.size % 2 == 0
        self.next_rank = (comm.rank + 1) % comm.size
        self.prev_rank = (comm.rank + comm.size - 1) % comm.size
        self.comm = comm

    def locate(self, x):
        point = df.Point(*x)
        c = self.tree.compute_first_entity_collision(point)
        return c if c < self.lim else -1

    def add_particles(self, particles):
        not_found = []
        for p in particles:
            x = p[:self.dim]

            c = self.locate(x)
            if c > -1: 
                key = next(self.ticket)
                self.particles[key] = p
                if c in self.cells:
                    self.cells[c] + key
                else:
                    self.cells[c] = CellWithParticles(self, c, key)
                key += 1
            else:
                not_found.append(p)
        count = len(not_found)
        count_global = self.comm.allgather(count)
        # if self.comm.rank == 0: info('>> %s' % count_global)

        loop = 1
        while max(count_global) > 0 and loop < self.comm.size:
            loop += 1
            received = np.zeros(count_global[self.prev_rank]*self.psize, dtype=float)
            if self.comm.rank % 2:
                # Send to next
                self.comm.Send(np.array(not_found).flatten(), self.next_rank, self.comm.rank)
                # Receive particles from previous
                self.comm.Recv(received, self.prev_rank, self.prev_rank)
            else:
                # Receive particles from previous
                self.comm.Recv(received, self.prev_rank, self.prev_rank)
                # Send to next
                self.comm.Send(np.array(not_found).flatten(), self.next_rank, self.comm.rank)
            # Work with received
            received = received.reshape((-1, self.psize))
            not_found = []
            for p in received:
                x = p[:self.dim]

                c = self.locate(x)
                if c > -1: 
                    key = next(self.ticket)
                    self.particles[key] = p
                    if c in self.cells:
                        self.cells[c] + key
                    else:
                        self.cells[c] = CellWithParticles(self, c, key)
                    key += 1
                else:
                    not_found.append(p)
            count = len(not_found)
            count_global = self.comm.allgather(count)

            # if self.comm.rank == 0: info('%s' % count_global)
        self.comm.barrier()
   #  def add_particles(self, particles):
   #      for i, particle in enumerate(list_of_particles):
   #          c = self.locate(particle)
   #          if not (c == -1 or c == __UINT32_MAX__):
   #              my_found[i] = True
   #              if not has_properties:
   #                  pmap += self.mesh, c, particle
   #              else:
   #                  # Get values of properties for this particle
   #                  for key in properties:
   #                      particle_properties[key] = properties_d[key][i]
   #                  pmap += self.mesh, c, particle, particle_properties
   #      # All particles must be found on some process
   #      comm.Reduce(my_found, all_found, root=0)

   #      if self.myrank == 0:
   #          missing = np.where(all_found == 0)[0]
   #          n_missing = len(missing)

   #          assert n_missing == 0,\
   #              '%d particles are not located in mesh' % n_missing

   #          # Print particle info
   #          if self.__debug:
   #              for i in missing:
   #                  print 'Missing', list_of_particles[i].position

   #              n_duplicit = len(np.where(all_found > 1)[0])
   #              print 'There are %d duplicit particles' % n_duplicit

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, VectorFunctionSpace, info, Timer
    import sys

    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, 'CG', 1)

    property_layout = {}#{'q': 1}
    lpc = LPCollection(V, property_layout)
    size = lpc.comm.size

    nparticles = int(sys.argv[1])
    
    t = Timer('foo')
    for i in range(100):
        particles_x = 0.8*np.random.rand(nparticles/size, 2)
        particles_q = np.random.rand(nparticles/size)
        particles = np.c_[particles_x, particles_q]
        lpc.add_particles(particles_x)
    dt = t.stop()
    info('%g' % dt)

    # if lpc.comm.rank == 0:
    #     print lpc.particles
    #     for i, c in lpc.cells.iteritems():
    #         print i, c.particles

    
