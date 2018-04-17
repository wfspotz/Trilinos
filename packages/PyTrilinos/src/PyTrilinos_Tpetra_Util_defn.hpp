// @HEADER
// ***********************************************************************
//
//          PyTrilinos: Python Interfaces to Trilinos Packages
//                 Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia
// Corporation, the U.S. Government retains certain rights in this
// software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact William F. Spotz (wfspotz@sandia.gov)
//
// ***********************************************************************
// @HEADER

#ifndef PYTRILINOS_TPETRA_UTIL_DEFN_HPP
#define PYTRILINOS_TPETRA_UTIL_DEFN_HPP

#include "PyTrilinos_Tpetra_Util_decl.hpp"
//#include "swigpyrun.h"

////////////////////////////////////////////////////////////////////////

namespace PyTrilinos
{

////////////////////////////////////////////////////////////////////////

template< class Scalar >
PyObject *
convertToDistArray(const Tpetra::MultiVector< Scalar,
                                              PYTRILINOS_LOCAL_ORD,
                                              PYTRILINOS_GLOBAL_ORD > & tmv)
{
  // Initialization
  PyObject   * dap       = NULL;
  PyObject   * dim_data  = NULL;
  PyObject   * dim_dict  = NULL;
  PyObject   * dist_type = NULL;
  PyObject   * start     = NULL;
  PyObject   * stop      = NULL;
  PyObject   * indices   = NULL;
  PyObject   * buffer    = NULL;
  Py_ssize_t   ndim      = 1;
  npy_intp     dims[3];
  Teuchos::ArrayRCP< const Scalar > data;

  // Get the underlying Tpetra::Map< PYTRILINOS_LOCAL_ORD, PYTRILINOS_GLOBAL_ORD >
  Teuchos::RCP< const Tpetra::Map< PYTRILINOS_LOCAL_ORD,
                                   PYTRILINOS_GLOBAL_ORD > > tm = tmv.getMap();

  // Allocate the DistArray Protocol object and set the version key
  // value
  dap = PyDict_New();
  if (!dap) goto fail;
  if (PyDict_SetItemString(dap,
                           "__version__",
                           PyString_FromString("0.9.0")) == -1) goto fail;

  // Get the Dimension Data and the number of dimensions.  If the
  // underlying Tpetra::BlockMap has variable element sizes, an error
  // will be detected here.
  dim_data = convertToDimData(tm, tmv.getNumVectors());
  if (!dim_data) goto fail;
  ndim = PyTuple_Size(dim_data);

  // Assign the Dimension Data key value.
  if (PyDict_SetItemString(dap,
                           "dim_data",
                           dim_data) == -1) goto fail;

  // Extract the buffer dimensions from the Dimension Data, construct
  // the buffer and assign the buffer key value
  for (Py_ssize_t i = 0; i < ndim; ++i)
  {
    dim_dict = PyTuple_GetItem(dim_data, i);
    if (!dim_dict) goto fail;
    dist_type = PyDict_GetItemString(dim_dict, "dist_type");
    if (!dist_type) goto fail;
    if (strcmp(convertPyStringToChar(dist_type), "b") == 0)
    {
      start = PyDict_GetItemString(dim_dict, "start");
      if (!start) goto fail;
      stop = PyDict_GetItemString(dim_dict, "stop");
      if (!stop) goto fail;
      dims[i] = PyInt_AsLong(stop) - PyInt_AsLong(start);
      if (PyErr_Occurred()) goto fail;
    }
    else if (strcmp(convertPyStringToChar(dist_type), "u") == 0)
    {
      indices = PyDict_GetItemString(dim_dict, "indices");
      if (!indices) goto fail;
      dims[i] = PyArray_DIM((PyArrayObject*)indices,0);
      if (PyErr_Occurred()) goto fail;
    }
    else
    {
      PyErr_Format(PyExc_ValueError,
                   "Unsupported distribution type '%s'",
                   convertPyStringToChar(dist_type));
      goto fail;
    }
  }
  data = tmv.getData(0);
  buffer = PyArray_SimpleNewFromData(ndim,
                                     dims,
                                     NumPy_TypeCode< Scalar >(),
                                     (void*)data.getRawPtr());
  if (!buffer) goto fail;
  if (PyDict_SetItemString(dap,
                           "buffer",
                           buffer) == -1) goto fail;

  // Return the DistArray Protocol object
  return dap;

  // Error handling
  fail:
  Py_XDECREF(dap);
  Py_XDECREF(dim_data);
  Py_XDECREF(dim_dict);
  Py_XDECREF(buffer);
  return NULL;
}

////////////////////////////////////////////////////////////////////////

template< class Scalar >
Teuchos::RCP< Tpetra::MultiVector< Scalar,
                                   PYTRILINOS_LOCAL_ORD,
                                   PYTRILINOS_GLOBAL_ORD,
                                   DefaultNodeType > > *
convertPythonToTpetraMultiVector(PyObject * pyobj,
                                 int * newmem)
{
  // SWIG initialization
  static swig_type_info * swig_TMV_ptr =
    SWIG_TypeQuery("Teuchos::RCP< Tpetra::MultiVector< Scalar,PYTRILINOS_LOCAL_ORD,PYTRILINOS_GLOBAL_ORD,DefaultNodeType > >*");
  static swig_type_info * swig_DMDV_ptr =
    SWIG_TypeQuery("Teuchos::RCP< Domi::MDVector< Scalar,Domi::DefaultNode::DefaultNodeType > >*");
  //
  // Get the default communicator
  const Teuchos::RCP< const Teuchos::Comm<int> > comm =
    Teuchos::DefaultComm<int>::getComm();
  //
  // Result objects
  void *argp = 0;
  PyObject * distarray = 0;
  Teuchos::RCP< Tpetra::MultiVector< Scalar,
                                     PYTRILINOS_LOCAL_ORD,
                                     PYTRILINOS_GLOBAL_ORD,
                                     DefaultNodeType > > smartresult;
  Teuchos::RCP< Tpetra::MultiVector< Scalar,
                                     PYTRILINOS_LOCAL_ORD,
                                     PYTRILINOS_GLOBAL_ORD,
                                     DefaultNodeType > > * result;
#ifdef HAVE_DOMI
  Teuchos::RCP< Domi::MDVector< Scalar > > dmdv_rcp;
#endif
  *newmem = 0;
  //
  // Check if the Python object is a wrapped Tpetra::MultiVector
  int res = SWIG_ConvertPtrAndOwn(pyobj, &argp, swig_TMV_ptr, 0, newmem);
  if (SWIG_IsOK(res))
  {
    result =
      reinterpret_cast< Teuchos::RCP< Tpetra::MultiVector< Scalar,
                                                           PYTRILINOS_LOCAL_ORD,
                                                           PYTRILINOS_GLOBAL_ORD,
                                                           DefaultNodeType > > * >(argp);
    return result;
  }

#ifdef HAVE_DOMI
  //
  // Check if the Python object is a wrapped Domi::MDVector< Scalar >
  *newmem = 0;
  res = SWIG_ConvertPtrAndOwn(pyobj, &argp, swig_DMDV_ptr, 0, newmem);
  if (SWIG_IsOK(res))
  {
    dmdv_rcp =
      *reinterpret_cast< Teuchos::RCP< Domi::MDVector< Scalar > > * >(argp);
    try
    {
      smartresult =
        dmdv_rcp->template getTpetraMultiVectorView< PYTRILINOS_LOCAL_ORD,
                                                     PYTRILINOS_GLOBAL_ORD >();
      *newmem = *newmem | SWIG_CAST_NEW_MEMORY;
    }
    catch (Domi::TypeError & e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }
    catch (Domi::MDMapNoncontiguousError & e)
    {
      PyErr_SetString(PyExc_ValueError, e.what());
      return NULL;
    }
    catch (Domi::MapOrdinalError & e)
    {
      PyErr_SetString(PyExc_IndexError, e.what());
      return NULL;
    }
    result = new Teuchos::RCP< Tpetra::MultiVector< Scalar,
                                                    PYTRILINOS_LOCAL_ORD,
                                                    PYTRILINOS_GLOBAL_ORD,
                                                    DefaultNodeType > >(smartresult);
    return result;
  }
  //
  // Check if the Python object supports the DistArray Protocol
  if (PyObject_HasAttrString(pyobj, "__distarray__"))
  {
    try
    {
      if (!(distarray = PyObject_CallMethod(pyobj, (char*) "__distarray__", (char*) "")))
        return NULL;
      DistArrayProtocol dap(distarray);
      dmdv_rcp = convertToMDVector< Scalar >(comm, dap);
      Py_DECREF(distarray);
    }
    catch (PythonException & e)
    {
      e.restore();
      return NULL;
    }
    try
    {
      smartresult =
        dmdv_rcp->template getTpetraMultiVectorView< PYTRILINOS_LOCAL_ORD,
                                                     PYTRILINOS_GLOBAL_ORD>();
      *newmem = SWIG_CAST_NEW_MEMORY;
    }
    catch (Domi::TypeError & e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }
    catch (Domi::MDMapNoncontiguousError & e)
    {
      PyErr_SetString(PyExc_ValueError, e.what());
      return NULL;
    }
    catch (Domi::MapOrdinalError & e)
    {
      PyErr_SetString(PyExc_IndexError, e.what());
      return NULL;
    }
    result =
      new Teuchos::RCP< Tpetra::MultiVector< Scalar,
                                             PYTRILINOS_LOCAL_ORD,
                                             PYTRILINOS_GLOBAL_ORD,
                                             DefaultNodeType > >(smartresult);
    return result;
  }
#endif

  //
  // Check if the environment is serial, and if so, check if the
  // Python object is a NumPy array
  if (comm->getSize() == 1)
  {
    if (PyArray_Check(pyobj))
    {
      PyArrayObject * array =
        (PyArrayObject*) PyArray_ContiguousFromObject(pyobj, NumPy_TypeCode< Scalar >(), 0, 0);
      if (!array) return NULL;
      size_t numVec, vecLen;
      int ndim = PyArray_NDIM(array);
      if (ndim == 1)
      {
        numVec = 1;
        vecLen = PyArray_DIM(array, 0);
      }
      else
      {
        numVec = PyArray_DIM(array, 0);
        vecLen = 1;
        for (int i=1; i < ndim; ++i) vecLen *= PyArray_DIM(array, i);
      }
      Scalar * data = (Scalar*) PyArray_DATA(array);
      Teuchos::ArrayView< Scalar > arrayView(data, vecLen*numVec);
      Teuchos::RCP< const Tpetra::Map< PYTRILINOS_LOCAL_ORD,
                                       PYTRILINOS_GLOBAL_ORD,DefaultNodeType > >
        map = Teuchos::rcp(new Tpetra::Map< PYTRILINOS_LOCAL_ORD,
                                            PYTRILINOS_GLOBAL_ORD,
                                            DefaultNodeType >(vecLen, 0, comm));
      smartresult =
        Teuchos::rcp(new Tpetra::MultiVector< Scalar,
                                              PYTRILINOS_LOCAL_ORD,
                                              PYTRILINOS_GLOBAL_ORD,
                                              DefaultNodeType >(map,
                                                                arrayView,
                                                                vecLen,
                                                                numVec));
      result =
        new Teuchos::RCP< Tpetra::MultiVector< Scalar,
                                               PYTRILINOS_LOCAL_ORD,
                                               PYTRILINOS_GLOBAL_ORD,
                                               DefaultNodeType > >(smartresult);
      return result;
    }
  }
  //
  // If we get to this point, then none of our known converters will
  // work, so it is time to set a Python error
  PyErr_Format(PyExc_TypeError, "Could not convert argument of type '%s'\n"
               "to a Tpetra::MultiVector",
               convertPyStringToChar(PyObject_Str(PyObject_Type(pyobj))));
  return NULL;
}

////////////////////////////////////////////////////////////////////////

template< class Scalar >
Teuchos::RCP< Tpetra::Vector< Scalar,
                              PYTRILINOS_LOCAL_ORD,
                              PYTRILINOS_GLOBAL_ORD,
                              DefaultNodeType > > *
convertPythonToTpetraVector(PyObject * pyobj,
                            int * newmem)
{
  // SWIG initialization
  static swig_type_info * swig_TV_ptr =
    SWIG_TypeQuery("Teuchos::RCP< Tpetra::Vector< Scalar,PYTRILINOS_LOCAL_ORD,PYTRILINOS_GLOBAL_ORD,DefaultNodeType > >*");
  static swig_type_info * swig_DMDV_ptr =
    SWIG_TypeQuery("Teuchos::RCP< Domi::MDVector< Scalar,Domi::DefaultNode::DefaultNodeType > >*");
  //
  // Get the default communicator
  const Teuchos::RCP< const Teuchos::Comm<int> > comm =
    Teuchos::DefaultComm<int>::getComm();
  //
  // Result objects
  void *argp = 0;
  PyObject * distarray = 0;
  Teuchos::RCP< Tpetra::Vector< Scalar,
                                PYTRILINOS_LOCAL_ORD,
                                PYTRILINOS_GLOBAL_ORD,
                                DefaultNodeType > > smartresult;
  Teuchos::RCP< Tpetra::Vector< Scalar,
                                PYTRILINOS_LOCAL_ORD,
                                PYTRILINOS_GLOBAL_ORD,
                                DefaultNodeType > > * result;
#ifdef HAVE_DOMI
  Teuchos::RCP< Domi::MDVector< Scalar > > dmdv_rcp;
#endif
  *newmem = 0;
  //
  // Check if the Python object is a wrapped Tpetra::Vector
  int res = SWIG_ConvertPtrAndOwn(pyobj, &argp, swig_TV_ptr, 0, newmem);
  if (SWIG_IsOK(res))
  {
    result =
      reinterpret_cast< Teuchos::RCP< Tpetra::Vector< Scalar,
                                                      PYTRILINOS_LOCAL_ORD,
                                                      PYTRILINOS_GLOBAL_ORD,
                                                      DefaultNodeType > > * >(argp);
    return result;
  }

#ifdef HAVE_DOMI
  //
  // Check if the Python object is a wrapped Domi::MDVector< Scalar >
  *newmem = 0;
  res = SWIG_ConvertPtrAndOwn(pyobj, &argp, swig_DMDV_ptr, 0, newmem);
  if (SWIG_IsOK(res))
  {
    dmdv_rcp =
      *reinterpret_cast< Teuchos::RCP< Domi::MDVector< Scalar > > * >(argp);
    try
    {
      smartresult =
        dmdv_rcp->template getTpetraVectorView< PYTRILINOS_LOCAL_ORD,
                                                PYTRILINOS_GLOBAL_ORD >();
      *newmem = *newmem | SWIG_CAST_NEW_MEMORY;
    }
    catch (Domi::TypeError & e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }
    catch (Domi::MDMapNoncontiguousError & e)
    {
      PyErr_SetString(PyExc_ValueError, e.what());
      return NULL;
    }
    catch (Domi::MapOrdinalError & e)
    {
      PyErr_SetString(PyExc_IndexError, e.what());
      return NULL;
    }
    result = new Teuchos::RCP< Tpetra::Vector< Scalar,
                                               PYTRILINOS_LOCAL_ORD,
                                               PYTRILINOS_GLOBAL_ORD,
                                               DefaultNodeType > >(smartresult);
    return result;
  }
  //
  // Check if the Python object supports the DistArray Protocol
  if (PyObject_HasAttrString(pyobj, "__distarray__"))
  {
    try
    {
      if (!(distarray = PyObject_CallMethod(pyobj, (char*) "__distarray__", (char*) "")))
        return NULL;
      DistArrayProtocol dap(distarray);
      dmdv_rcp = convertToMDVector< Scalar >(comm, dap);
      Py_DECREF(distarray);
    }
    catch (PythonException & e)
    {
      e.restore();
      return NULL;
    }
    try
    {
      smartresult =
        dmdv_rcp->template getTpetraVectorView< PYTRILINOS_LOCAL_ORD,
                                                PYTRILINOS_GLOBAL_ORD >();
      *newmem = SWIG_CAST_NEW_MEMORY;
    }
    catch (Domi::TypeError & e)
    {
      PyErr_SetString(PyExc_TypeError, e.what());
      return NULL;
    }
    catch (Domi::MDMapNoncontiguousError & e)
    {
      PyErr_SetString(PyExc_ValueError, e.what());
      return NULL;
    }
    catch (Domi::MapOrdinalError & e)
    {
      PyErr_SetString(PyExc_IndexError, e.what());
      return NULL;
    }
    result = new Teuchos::RCP< Tpetra::Vector< Scalar,
                                               PYTRILINOS_LOCAL_ORD,
                                               PYTRILINOS_GLOBAL_ORD,
                                               DefaultNodeType > >(smartresult);
    return result;
  }
#endif

  //
  // Check if the environment is serial, and if so, check if the
  // Python object is a NumPy array
  if (comm->getSize() == 1)
  {
    if (PyArray_Check(pyobj))
    {
      PyArrayObject * array =
        (PyArrayObject*) PyArray_ContiguousFromObject(pyobj, NumPy_TypeCode< Scalar >(), 0, 0);
      if (!array) return NULL;
      int ndim = PyArray_NDIM(array);
      size_t vecLen = 1;
      for (int i=1; i < ndim; ++i) vecLen *= PyArray_DIM(array, i);
      Scalar * data = (Scalar*) PyArray_DATA(array);
      Teuchos::ArrayView< const Scalar > arrayView(data, vecLen);
      Teuchos::RCP< const Tpetra::Map< PYTRILINOS_LOCAL_ORD,
                                       PYTRILINOS_GLOBAL_ORD,
                                       DefaultNodeType > > map =
        Teuchos::rcp(new Tpetra::Map< PYTRILINOS_LOCAL_ORD,
                                      PYTRILINOS_GLOBAL_ORD,
                                      DefaultNodeType >(vecLen, 0, comm));
      smartresult =
        Teuchos::rcp(new Tpetra::Vector< Scalar,
                                         PYTRILINOS_LOCAL_ORD,
                                         PYTRILINOS_GLOBAL_ORD,
                                         DefaultNodeType >(map, arrayView));
      result =
        new Teuchos::RCP< Tpetra::Vector< Scalar,
                                          PYTRILINOS_LOCAL_ORD,
                                          PYTRILINOS_GLOBAL_ORD,
                                          DefaultNodeType > >(smartresult);
      return result;
    }
  }
  //
  // If we get to this point, then none of our known converters will
  // work, so it is time to set a Python error
  PyErr_Format(PyExc_TypeError, "Could not convert argument of type '%s'\n"
               "to a Tpetra::Vector",
               convertPyStringToChar(PyObject_Str(PyObject_Type(pyobj))));
  return NULL;
}

////////////////////////////////////////////////////////////////////////

}

#endif
