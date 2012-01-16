#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include "Panzer_CellData.hpp"
#include "Panzer_IntegrationRule.hpp"
#include "Panzer_IntegrationValues.hpp"
#include "Panzer_ArrayTraits.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Panzer_BasisValues.hpp"

using Teuchos::RCP;
using Teuchos::rcp;
using panzer::IntegrationRule;
using Intrepid::FieldContainer;

namespace panzer {

  TEUCHOS_UNIT_TEST(integration_values, volume)
  {
    Teuchos::RCP<shards::CellTopology> topo = 
       Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

    const int num_cells = 20;
    const int base_cell_dimension = 2;
    const panzer::CellData cell_data(num_cells, base_cell_dimension,topo);

    const int cubature_degree = 2;    
    RCP<IntegrationRule> int_rule = 
      rcp(new IntegrationRule(cubature_degree, cell_data));
    
    panzer::IntegrationValues<double,Intrepid::FieldContainer<double> > 
      int_values;

    int_values.setupArrays(int_rule);

    const int num_vertices = int_rule->topology->getNodeCount();
    FieldContainer<double> node_coordinates(num_cells, num_vertices,
					    base_cell_dimension);



    // Set up node coordinates.  Here we assume the following
    // ordering.  This needs to be consistent with shards topology,
    // otherwise we will get negative determinates

    // 3(0,1)---2(1,1)
    //   |    0  |
    //   |       |
    // 0(0,0)---1(1,0)

    typedef panzer::ArrayTraits<double,FieldContainer<double> >::size_type size_type;
    const size_type x = 0;
    const size_type y = 1;
    for (size_type cell = 0; cell < node_coordinates.dimension(0); ++cell) {
      node_coordinates(cell,0,x) = 0.0;
      node_coordinates(cell,0,y) = 0.0;
      node_coordinates(cell,1,x) = 1.0;
      node_coordinates(cell,1,y) = 0.0;
      node_coordinates(cell,2,x) = 1.0;
      node_coordinates(cell,2,y) = 1.0;
      node_coordinates(cell,3,x) = 0.0;
      node_coordinates(cell,3,y) = 1.0;
    }

    int_values.evaluateValues(node_coordinates);
    
    const std::string basis_type = "Q2";
  
    RCP<panzer::BasisIRLayout> basis = rcp(new panzer::BasisIRLayout(basis_type, *int_rule));

    panzer::BasisValues<double,Intrepid::FieldContainer<double> > basis_values;

    basis_values.setupArrays(basis);
    
    basis_values.evaluateValues(int_values.cub_points,
				int_values.jac_inv,
				int_values.weighted_measure,
				node_coordinates);

    TEST_FLOATING_EQUALITY(basis_values.basis_coordinates(0,0,x),
			   0.0, 1.0e-8);
    TEST_FLOATING_EQUALITY(basis_values.basis_coordinates(0,0,y),
			   0.0, 1.0e-8);

    TEST_FLOATING_EQUALITY(basis_values.basis_coordinates(0,1,x),
			   1.0, 1.0e-8);
    TEST_FLOATING_EQUALITY(basis_values.basis_coordinates(0,1,y),
			   0.0, 1.0e-8);

    TEST_FLOATING_EQUALITY(basis_values.basis_coordinates(0,2,x),
			   1.0, 1.0e-8);
    TEST_FLOATING_EQUALITY(basis_values.basis_coordinates(0,2,y),
			   1.0, 1.0e-8);

    TEST_FLOATING_EQUALITY(basis_values.basis_coordinates(0,3,x),
			   0.0, 1.0e-8);
    TEST_FLOATING_EQUALITY(basis_values.basis_coordinates(0,3,y),
			   1.0, 1.0e-8);

  }
  TEUCHOS_UNIT_TEST(basis_values, grad_quad)
  {
    Teuchos::RCP<shards::CellTopology> topo = 
       Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

    const int num_cells = 4;
    const int base_cell_dimension = 2;
    const panzer::CellData cell_data(num_cells, base_cell_dimension,topo);

    const int cubature_degree = 4;    
    RCP<IntegrationRule> int_rule = 
      rcp(new IntegrationRule(cubature_degree, cell_data));
    const int num_qp = int_rule->num_points;
    
    panzer::IntegrationValues<double,Intrepid::FieldContainer<double> > 
      int_values;

    int_values.setupArrays(int_rule);

    const int num_vertices = int_rule->topology->getNodeCount();
    FieldContainer<double> node_coordinates(num_cells, num_vertices,
					    base_cell_dimension);



    // Set up node coordinates.  Here we assume the following
    // ordering.  This needs to be consistent with shards topology,
    // otherwise we will get negative determinates

    // 3(0,1)---2(1,1)
    //   |    0  |
    //   |       |
    // 0(0,0)---1(1,0)

    typedef panzer::ArrayTraits<double,FieldContainer<double> >::size_type size_type;
    const size_type x = 0;
    const size_type y = 1;
    for (size_type cell = 0; cell < node_coordinates.dimension(0); ++cell) {
      int xleft = cell % 2;
      int yleft = int(cell/2);

      node_coordinates(cell,0,x) = xleft*0.5;
      node_coordinates(cell,0,y) = yleft*0.5;

      node_coordinates(cell,1,x) = (xleft+1)*0.5;
      node_coordinates(cell,1,y) = yleft*0.5; 

      node_coordinates(cell,2,x) = (xleft+1)*0.5;
      node_coordinates(cell,2,y) = (yleft+1)*0.5;

      node_coordinates(cell,3,x) = xleft*0.5;
      node_coordinates(cell,3,y) = (yleft+1)*0.5;

      out << "Cell " << cell << " = ";
      for(int i=0;i<4;i++)
         out << "(" << node_coordinates(cell,i,x) << ", "
                    << node_coordinates(cell,i,y) << ") ";
      out << std::endl;
    }

    int_values.evaluateValues(node_coordinates);
    
    const std::string basis_type = "Q1";
  
    RCP<panzer::BasisIRLayout> basis = rcp(new panzer::BasisIRLayout(basis_type, *int_rule));

    panzer::BasisValues<double,Intrepid::FieldContainer<double> > basis_values;

    basis_values.setupArrays(basis);
    
    basis_values.evaluateValues(int_values.cub_points,
				int_values.jac_inv,
				int_values.weighted_measure,
				node_coordinates);

    double relCellVol = 0.25*0.25; // this is the relative (to the reference cell) volume
    for(int i=0;i<num_qp;i++) {
       double x = int_values.cub_points(i,0);
       double y = int_values.cub_points(i,1);
       double weight = int_values.cub_weights(i);

       // check reference values
       TEST_EQUALITY(basis_values.basis_ref(0,i),0.25*(x-1.0)*(y-1.0));
       TEST_EQUALITY(basis_values.grad_basis_ref(0,i,0),0.25*(y-1.0));
       TEST_EQUALITY(basis_values.grad_basis_ref(0,i,1),0.25*(x-1.0));

       // check basis values
       for(int cell=0;cell<num_cells;cell++) {

          TEST_EQUALITY(int_values.jac_det(cell,i),relCellVol);

          // check out basis on transformed elemented
          TEST_EQUALITY(basis_values.basis_ref(0,i),basis_values.basis(cell,0,i));
          TEST_EQUALITY(basis_values.weighted_basis(cell,0,i),relCellVol*weight*basis_values.basis(cell,0,i));

          TEST_EQUALITY(basis_values.grad_basis(cell,0,i,0),4.0*basis_values.grad_basis_ref(0,i,0));
          TEST_EQUALITY(basis_values.grad_basis(cell,0,i,1),4.0*basis_values.grad_basis_ref(0,i,1));

          TEST_EQUALITY(basis_values.weighted_grad_basis(cell,0,i,0),relCellVol*weight*basis_values.grad_basis(cell,0,i,0));
          TEST_EQUALITY(basis_values.weighted_grad_basis(cell,0,i,1),relCellVol*weight*basis_values.grad_basis(cell,0,i,1));
       }
    }
  }

}
