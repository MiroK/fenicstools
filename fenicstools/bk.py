from collections import defaultdict
from mpi4py import MPI as pyMPI
from itertools import count, izip
import dolfin as df
import numpy as np


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
    def __init__(self, V, property_layout=None):
        '''TODO'''
        mesh = V.mesh()
        self.dim = mesh.geometry().dim()
        assert self.dim == mesh.topology().dim()
        mesh.init(0, self.dim)
        self.mesh = mesh

        self.tree = mesh.bounding_box_tree()
        self.lim = mesh.topology().size_global(self.dim)
            
        # NOTE: property layout here is the map which maps property name to
        # length of a vector that represents property value
        if property_layout is None: property_layout=[]
        assert all(p[1]> 0 for p in property_layout)
        offsets = [0, self.mesh.geometry().dim()] + [p[1] for p in property_layout]
        self.psize = sum(offsets)
        self.offsets = np.cumsum(offsets)
        self.keys = ['x'] + [p[0] for p in property_layout]

        self.particles = {}
        self.cells = {}
        self.ticket = count()

        comm = mesh.mpi_comm().tompi4py()
        assert comm.size == 1 or comm.size % 2 == 0
        self.next_rank = (comm.rank + 1) % comm.size
        self.prev_rank = (comm.rank + comm.size - 1) % comm.size
        self.comm = comm

#        x  = mesh.coordinates()
#        # Local
#        local_min = np.min(x, axis=0)
#        local_max = np.max(x, axis=0)
#        # Alloc global
#        global_min = np.tile(local_min, (comm.size, 1))
#        global_max = np.tile(local_max, (comm.size, 1))
#        # Communicate
#        comm.Allgather([local_min, pyMPI.DOUBLE], [global_min, pyMPI.DOUBLE])
#        comm.Allgather([local_max, pyMPI.DOUBLE], [global_max, pyMPI.DOUBLE])
#        # 'Reduce'
#        bounding_min = np.min(global_min, axis=0)
#        bounding_max = np.max(global_max, axis=0)
#
#        info('%s %s' % (local_min, local_max))
#        info('%s %s' % (global_min, global_max))
#        info('%s %s' % (bounding_min, bounding_max))
#
#        self.is_inside_local_bbox = \
#                lambda x: all(df.between(xi, (lmin_xi, lmax_xi))
#                              for (xi, lmin_xi, lmax_xi) in izip(x, local_min, local_max))
#
#        self.is_inside_global_bbox = \
#                lambda x: all(df.between(xi, (lmin_xi, lmax_xi))
#                              for (xi, lmin_xi, lmax_xi) in izip(x, bounding_min, bounding_max))
#
        element = V.dolfin_element()
        
        num_tensor_entries = 1
        for i in range(element.value_rank()): num_tensor_entries *= element.value_dimension(i)

        self.coefficients = np.zeros(element.space_dimension())
        self.basis_matrix = np.zeros((element.space_dimension(), num_tensor_entries))
        self.element = element
        self.num_tensor_entries = num_tensor_entries

    def counts(self):
        local_count = np.array([len(self.cells), len(self.particles)])
        global_count = self.comm.allreduce(local_count)
        return local_count, global_count

    def get_x(self, particle): return self.particles[particle][:self.dim]

    def get_property(self, particle, prop):
        if isinstance(particle, int):
            index = self.keys.index(prop)
            return self.particles[particle][self.offsets[index]:self.offsets[index+1]]
        else:
            return [self.get_property(p, prop) for p in particle]

    def locate(self, x):
        #if self.is_inside_global_bbox(x):
        #    if self.is_inside_local_bbox(x):
        point = df.Point(*x)
        c = self.tree.compute_first_entity_collision(point)
        return c if c < self.lim else -1
        #    else:
        #        return -1
        #else:
        #    return -2


    def __add_particles_local(self, particles):
        not_found = []
        for p in particles:
            x = p[:self.dim]

            c = self.locate(x)
            print c
            if c > -1: 
                key = next(self.ticket)
                self.particles[key] = p
                if c not in self.cells:
                    self.cells[c] = CellWithParticles(self, c, key)
                self.cells[c] + key
                key += 1
            else:
                not_found.append(p)
        return not_found

    def add_particles(self, particles):
        not_found = self.__add_particles_local(particles)
        count = len(not_found)
        count_global = self.comm.allgather(count)

        self.__add_particles_global(count_global, not_found)

    
    def __add_particles_global(self, count_global, not_found):
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
            not_found = self.__add_particles_local(received)
            count = len(not_found)
            count_global = self.comm.allgather(count)
            # info('%d %s' % (loop, count_global))
        # self.comm.barrier()

        return not_found


    def step(self, u, dt):
        'Move particles by forward Euler x += u*dt'
        for cwp in self.cells.itervalues():
            vertex_coordinates, orientation = cwp.get_vertex_coordinates(), cwp.orientation()
            # Restrict once per cell
            u.restrict(self.coefficients, self.element, cwp, vertex_coordinates, cwp)
            for particle in cwp.particles:
                x = self.get_x(particle) 
                # Compute velocity at position x
                self.element.evaluate_basis_all(self.basis_matrix, x, vertex_coordinates, orientation)
                x[:] = x[:] + dt*np.dot(self.coefficients, self.basis_matrix)[:]
        self.relocate()


    def relocate(self):
        cell_map = defaultdict(list)
        empty_cells = []
        for cwp in self.cells.itervalues():

            left = []
            for particle in cwp.particles:
                x = self.get_x(particle)

                point = df.Point(*x)
                # Search only if particle moved outside original cell
                if not cwp.contains(point):
                    left.append(particle)
                    # FIXME
                    new_cell = -1
                    # Check neighbor cells
                    for neighbor in cwp.neighbors:
                        if neighbor.contains(point):
                            new_cell = neighbor.index()
                            break
                    # Do a completely new search if not found by now
                    if new_cell == -1: new_cell = self.locate(x)
                        # print '<<', new_cell
                        # if new_cell > -2:
                        #    cell_map[new_cell].append(particle)
            cwp.particles.difference_update(set(left))
            if not cwp.particles: empty_cells.append(cwp.index())
        for cell in empty_cells: self.cells.pop(cell)

        # Add locally found particles
        local_cells = cell_map.keys()
        if -1 in local_cells:
            local_cells.remove(-1)
            non_local = cell_map[-1]
        else:
            non_local = []

        for c in local_cells:
            print 'C'
            if c not in self.cells:
                self.cells[c] = CellWithParticles(self, c)
            for p in cell_map[c]: self.cells[c] + p

        particles = [self.particles[i] for i in non_local]
        count_global = self.comm.allgather(len(particles))
        not_found = self.__add_particles_global(count_global, particles)
        for i in non_local: self.particles.pop(i)

        lost = self.comm.allreduce(len(not_found))
        # info('%d' % lost)

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh, VectorFunctionSpace, info, Timer, interpolate
    from dolfin import Expression
    import sys

    mesh = UnitSquareMesh(20, 20)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    v = interpolate(Expression(('-(x[1]-0.5)', 'x[0]-0.5'), degree=1), V)

    property_layout = []#[('q', 1)]
    lpc = LPCollection(V, property_layout)
    size = lpc.comm.size

    nparticles = int(sys.argv[1])
    
    for i in range(1):
        particles_x = 0.8*np.random.rand(nparticles/size, 2)
        #particles_q = np.random.rand(nparticles/size)
        #particles = np.c_[particles_x, particles_q]
        lpc.add_particles(particles_x)
    # info('%g' % dt)

    #print lpc.particles
    t = Timer('foo')
    for i in range(1):
        lpc.step(v, 0.01)
        print len(lpc.cells)
    dt = t.stop()
    info('Moved %s particles in %g' % (lpc.counts(), dt))

    # print lpc.particles
    # lpc.step(v, 0.01)

    # info('%s' % lpc.particles.keys())

    #print lpc.get_property(0, 'q')
    #lpc.get_property(0, 'q')[:] += 1
    #print lpc.get_property(range(1, 5), 'q')
