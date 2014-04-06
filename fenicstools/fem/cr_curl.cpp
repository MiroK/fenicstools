#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/SubSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/plot/plot.h>

namespace dolfin
{
  // Compute curl of 3d vector field u.
  void cr_curl(Function& curlu, const Function& u)
  {
    std::shared_ptr<const GenericDofMap>
      CR1_dofmap = u.function_space()->dofmap(),
      DG0_dofmap = curlu.function_space()->dofmap();

    // Get u and curlu as vectors
    std::shared_ptr<const GenericVector> U = u.vector();
    std::shared_ptr<GenericVector> CURLU = curlu.vector();

    // Figure out about the local dofs of DG0 
    std::pair<std::size_t, std::size_t>
    first_last_dof = DG0_dofmap->ownership_range();
    std::size_t first_dof = first_last_dof.first;
    std::size_t last_dof = first_last_dof.second;
    std::size_t n_local_dofs = last_dof - first_dof;

    // Make room for local values of U
    std::vector<double> CURLU_values(CURLU->local_size());

    // Get topological dimension so that we know what Facet is
    const Mesh mesh = *u.function_space()->mesh();
    std::size_t tdim = mesh.topology().dim(); 
    // Get the info on length of u and gdim for the dot product
    std::size_t gdim = mesh.geometry().dim();

    // Fill the values
    for(CellIterator cell(mesh); !cell.end(); ++cell)
    {
      std::vector<dolfin::la_index>
      cell_dofs = DG0_dofmap->cell_dofs(cell->index());

      // In 3d there are 3 dofs per cell and they are all on the same process
      dolfin::la_index max_dof = 
        *std::max_element(cell_dofs.begin(), cell_dofs.end());
      if((first_dof <= max_dof) and (max_dof < last_dof))
      {
        Point cell_mp = cell->midpoint();
        double cell_volume = cell->volume();
        
        // Dofs of CR on all facets of the cell, global order
        std::vector<dolfin::la_index>
        facets_dofs = CR1_dofmap->cell_dofs(cell->index());
       
        std::vector<double> cell_integrals(3);
        std::size_t local_facet_index = 0;
        for(FacetIterator facet(*cell); !facet.end(); ++facet)
        {
          double facet_measure = 0;
          if(tdim == 2)
            facet_measure = Edge(mesh, facet->index()).length();
          else if(tdim == 3)
            facet_measure = Face(mesh, facet->index()).area();
          
          Point facet_normal = facet->normal();

          // Flip the normal if it is not outer already
          Point facet_mp = facet->midpoint();
          int sign = (facet_normal.dot(facet_mp - cell_mp) > 0) ? 1 : -1;
          facet_normal *= sign;

          // Dofs of CR on the facet, local order
          std::vector<std::size_t> facet_dofs; // u_x, u_y, u_z
          CR1_dofmap->tabulate_facet_dofs(facet_dofs, local_facet_index);

          // (n x u)_i*meas(facet)
          for(std::size_t i = 0; i < gdim; i++)
          {
            double u0 = (*U)[facets_dofs[facet_dofs[(i + 1)%gdim]]];
            double u1 = (*U)[facets_dofs[facet_dofs[(i + 2)%gdim]]];
            double n0 = facet_normal[(i + 1)%gdim];
            double n1 = facet_normal[(i + 2)%gdim];
            cell_integrals[i] += (-u0*n1 + u1*n0)*facet_measure;
          }
          local_facet_index += 1;
        }
        for(std::size_t i = 0; i < gdim; i++)
        {
          double cell_integral = cell_integrals[i]/cell_volume;
          dolfin::la_index cell_dof = cell_dofs[i];
          CURLU_values[cell_dof - first_dof] = cell_integral;
        }
      }
    }
    CURLU->set_local(CURLU_values);
    CURLU->apply("insert");
  }
} 
