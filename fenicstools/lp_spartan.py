from collections import defaultdict
from itertools import count
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
    def __init__(self, V, property_layout):
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
        assert all(v > 0 for v in property_layout.values())
        offsets = [0, self.mesh.geometry().dim()] + [property_layout[k] for k in property_layout]
        self.psize = sum(offsets)
        self.offsets = np.cumsum(offsets)
        self.keys = ['x'] + property_layout.keys()

        self.particles = {}
        self.cells = {}
        self.ticket = count()

        comm = mesh.mpi_comm().tompi4py()
        assert comm.size == 1 or comm.size % 2 == 0
        self.next_rank = (comm.rank + 1) % comm.size
        self.prev_rank = (comm.rank + comm.size - 1) % comm.size
        self.comm = comm

        element = V.dolfin_element()
        
        num_tensor_entries = 1
        for i in range(element.value_rank()): num_tensor_entries *= element.value_dimension(i)

        self.coefficients = np.zeros(element.space_dimension())
        self.basis_matrix = np.zeros((element.space_dimension(), num_tensor_entries))

        self.element = element
        self.num_tensor_entries = num_tensor_entries


    def locate(self, x):
        point = df.Point(*x)
        c = self.tree.compute_first_entity_collision(point)
        return c if c < self.lim else -1

    def __add_particles_local(self, particles):
        not_found = []
        for p in particles:
            x = p[:self.dim]

            c = self.locate(x)
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
        self.comm.barrier()

        return not_found


    def step(self, u, dt):
        'Move particles by forward Euler x += u*dt'
        for cwp in self.cells.itervalues():
            vertex_coordinates, orientation = cwp.get_vertex_coordinates(), cwp.orientation()
            # Restrict once per cell
            u.restrict(self.coefficients,
                       self.element,
                       cwp,
                       vertex_coordinates,
                       cwp)
            for particle in cwp.particles:
                x = self.particles[particle][:self.dim]
                # Compute velocity at position x
                self.element.evaluate_basis_all(self.basis_matrix,
                                                x,
                                                vertex_coordinates, 
                                                orientation)
                x[:] = x[:] + dt*np.dot(self.coefficients, self.basis_matrix)[:]
        self.relocate()


    def relocate(self):
        cell_map = defaultdict(list)
        empty_cells = []
        for cwp in self.cells.itervalues():

            left = []
            for particle in cwp.particles:
                x = self.particles[particle][:self.dim]

                point = df.Point(*x)
                # Search only if particle moved outside original cell
                if not cwp.contains(point):
                    left.append(particle)
                    # FIXME
                    found = False
                    # Check neighbor cells
                    for neighbor in cwp.neighbors:
                        if neighbor.contains(point):
                            new_cell = neighbor.index()
                            found = True
                            break
                    # Do a completely new search if not found by now
                    if not found: new_cell = self.locate(x)
                    # Record to map
                    cell_map[new_cell].append(particle)
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
    # info('%g' % dt)

    #print lpc.particles
    for i in range(20):
        print i
        lpc.step(v, 0.001)

    # print lpc.particles
    # lpc.step(v, 0.01)

    # info('%s' % lpc.particles.keys())
