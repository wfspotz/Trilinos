/*====================================================================
 * ------------------------
 * | CVS File Information |
 * ------------------------
 *
 * $RCSfile$
 *
 * $Author$
 *
 * $Date$
 *
 * $Revision$
 *
 * $Name$
 *====================================================================*/
#ifndef lint
static char *cvs_output = "$Id$";
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "dr_const.h"
#include "dr_input_const.h"
#include "dr_util_const.h"
#include "dr_par_util_const.h"
#include "dr_err_const.h"
#include "dr_output_const.h"

void print_distributed_mesh(
     int Proc, 
     int Num_Proc, 
     PROB_INFO_PTR prob,
     ELEM_INFO *elements)
{
int i, j, k;
int elem;


/*
 * Print the distributed mesh description for each processor.  This routine
 * is useful for debugging the input meshes (Nemesis or Chaco).  It is 
 * serial, so it should not be used for production runs.
 */

  print_sync_start(Proc, 1);

  printf ("############## Mesh information for processor %d ##############\n",
          Proc);
  printf ("Number Dimensions:     %d\n", Mesh.num_dims);
  printf ("Number Nodes:          %d\n", Mesh.num_nodes);
  printf ("Number Elements:       %d\n", Mesh.num_elems);
  printf ("Number Element Blocks: %d\n", Mesh.num_el_blks);
  printf ("Number Node Sets:      %d\n", Mesh.num_node_sets);
  printf ("Number Side Sets:      %d\n", Mesh.num_side_sets);

  for (i = 0; i < Mesh.num_el_blks; i++) {
    printf("\nElement block #%d\n", (i+1));
    printf("\tID:                 %d\n", Mesh.eb_ids[i]);
    printf("\tElement Type:       %s\n", Mesh.eb_names[i]);
    printf("\tElement Count:      %d\n", Mesh.eb_cnts[i]);
    printf("\tNodes Per Element:  %d\n", Mesh.eb_nnodes[i]);
    printf("\tAttrib Per Element: %d\n", Mesh.eb_nattrs[i]);
  }

  if (prob->read_coord) {
    printf("\nElement connect table, weights and coordinates:\n");
    for (i = 0; i < Mesh.num_elems; i++) {
      printf("%d:\n", elements[i].globalID);
      for (j = 0; j < Mesh.eb_nnodes[elements[i].elem_blk]; j++) {
        printf("\t%d (%f)|", elements[i].connect[j], elements[i].cpu_wgt);
        for (k = 0; k < Mesh.num_dims; k++) {
          printf(" %f", elements[i].coord[j][k]);
        }
        printf("\n");
      }
    }
  }

  if (prob->gen_graph) {
    /* now print the adjacencies */
    printf("\nElement adjacencies:\n");
    printf("elem\tnadj\tadj,proc\n");
    for (i = 0; i < Mesh.num_elems; i++) {
      printf("%d\t", elements[i].globalID);
      printf("%d\t", elements[i].nadj);
      for (j = 0; j < elements[i].nadj; j++) {
        if (elements[i].adj_proc[j] == Proc)
          elem = elements[elements[i].adj[j]].globalID;
        else
          elem = elements[i].adj[j];
        printf("%d,%d ", elem, elements[i].adj_proc[j]);
      }
      printf("\n");
    }
  }

  print_sync_end(Proc, Num_Proc, 1);
}


/*--------------------------------------------------------------------------*/
/* Purpose: Output the new element assignments.                             */
/*--------------------------------------------------------------------------*/
/* Author(s):  Matthew M. St.John (9226)                                    */
/*--------------------------------------------------------------------------*/
int output_results(int Proc,
                   int Num_Proc,
                   PARIO_INFO_PTR pio_info,
                   ELEM_INFO elements[])
/*
 * For the first swipe at this, don't try to create a new
 * exodus/nemesis file or anything. Just get the global ids,
 * sort them, and print them to a new ascii file.
 */
{
  /* Local declarations. */
  char   par_out_fname[FILENAME_MAX+1], ctemp[FILENAME_MAX+1];

  int   *global_ids;
  int    i, j;

  FILE  *fp;
/***************************** BEGIN EXECUTION ******************************/

  global_ids = (int *) malloc(Mesh.num_elems * sizeof(int));
  if (!global_ids) {
    Gen_Error(0, "fatal: insufficient memory");
    return 0;
  }

  for (i = j = 0; i < Mesh.elem_array_len; i++) {
    if (elements[i].globalID >= 0) {
      global_ids[j] = elements[i].globalID;
      j++;
    }
  }

  sort_int(Mesh.num_elems, global_ids);

  /* generate the parallel filename for this processor */
  strcpy(ctemp, pio_info->pexo_fname);
  strcat(ctemp, ".out");
  gen_par_filename(ctemp, par_out_fname, pio_info, Proc, Num_Proc);

  fp = fopen(par_out_fname, "w");

  fprintf(fp, "Global element ids assigned to processor %d\n", Proc);
  for (i = 0; i < Mesh.num_elems; i++)
    fprintf(fp, "%d\n", global_ids[i]);

  fclose(fp);

  return 1;
}
