/* -*- mode: C -*-  */
/* vim:set ts=2 sw=2 sts=2 et: */
/*
   IGraph library.
   Copyright (C) 2006-2023  Tamas Nepusz <ntamas@gmail.com>

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc.,  51 Franklin Street, Fifth Floor, Boston, MA
   02110-1301 USA

*/

#ifndef Py_IGRAPHMODULE_H
#define Py_IGRAPHMODULE_H

#ifdef __cplusplus
extern "C" {
#endif

/* C API functions */
#define PyIGraph_FromCGraph_NUM 0
#define PyIGraph_FromCGraph_RETURN PyObject*
#define PyIGraph_FromCGraph_PROTO (igraph_t *graph)

#define PyIGraph_ToCGraph_NUM 1
#define PyIGraph_ToCGraph_RETURN igraph_t*
#define PyIGraph_ToCGraph_PROTO (PyObject *graph)

/* Total number of C API pointers */
#define PyIGraph_API_pointers 2

#ifdef IGRAPH_MODULE
  /* This section is used when compiling igraphmodule.c */
  static PyIGraph_FromCGraph_RETURN PyIGraph_FromCGraph PyIGraph_FromCGraph_PROTO;
  static PyIGraph_ToCGraph_RETURN PyIGraph_ToCGraph PyIGraph_ToCGraph_PROTO;
#else
  /* This section is used in modules that use igraph's API */
  static void** PyIGraph_API;
# define PyIGraph_FromCGraph \
         (*(PyIGraph_FromCGraph_RETURN (*)PyIGraph_FromCGraph_PROTO) \
		 PyIGraph_API[PyIGraph_FromCGraph_NUM])
# define PyIGraph_ToCGraph \
         (*(PyIGraph_ToCGraph_RETURN (*)PyIGraph_ToCGraph_PROTO) \
		 PyIGraph_API[PyIGraph_ToCGraph_NUM])

  /* Return -1 and set exception on error, 0 on success */
  static int import_igraph(void) {
    PyIGraph_API = (void **)PyCapsule_Import("igraph._igraph._C_API", 0);
    return (PyIGraph_API != NULL) ? 0 : -1;
  }

#endif

#ifdef __cplusplus
}
#endif

#endif   /* !defined(Py_IGRAPHMODULE_H) */
