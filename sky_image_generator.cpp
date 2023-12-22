// build with: c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(cd /pybind11 && python3 -m pybind11 --includes) -I/opt/conda/include/opencv4/ -I/opt/conda/lib/python3.8/site-packages/numpy/core/include/ -L/opt/conda/lib/ sky_image_generator.cpp -o sky_image_generator$(python3-config --extension-suffix) -lopencv_core -lopencv_imgcodecs

#include <pybind11/pybind11.h>

#include "sky_image_generator.h"

#include <numpy/ndarrayobject.h>

// from https://github.com/Algomorph/pyboostcvconverter/blob/a63a147d1c1ed46b4c6d549604b21c230ffdeda8/src/pyboost_cv4_converter.cpp
#ifndef CV_MAX_DIM
		const int CV_MAX_DIM = 32;
#endif

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static PyObject* opencv_error = 0;

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

//===================   THREADING     ==============================================================
class PyAllowThreads {
public:
	PyAllowThreads() :
			_state(PyEval_SaveThread()) {
	}
	~PyAllowThreads() {
		PyEval_RestoreThread(_state);
	}
private:
	PyThreadState* _state;
};

class PyEnsureGIL {
public:
	PyEnsureGIL() :
			_state(PyGILState_Ensure()) {
	}
	~PyEnsureGIL() {
		PyGILState_Release(_state);
	}
private:
	PyGILState_STATE _state;
};

enum {
	ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2
};

class NumpyAllocator: public cv::MatAllocator {
public:
	NumpyAllocator() {
		stdAllocator = cv::Mat::getStdAllocator();
	}
	~NumpyAllocator() {
	}

	cv::UMatData* allocate(PyObject* o, int dims, const int* sizes, int type,
			size_t* step) const {
		cv::UMatData* u = new cv::UMatData(this);
		u->data = u->origdata = (uchar*) PyArray_DATA((PyArrayObject*) o);
		npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
		for (int i = 0; i < dims - 1; i++)
			step[i] = (size_t) _strides[i];
		step[dims - 1] = CV_ELEM_SIZE(type);
		u->size = sizes[0] * step[0];
		u->userdata = o;
		return u;
	}

	cv::UMatData* allocate(int dims0, const int* sizes, int type, void* data,
			size_t* step, cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const {
		if (data != 0) {
			CV_Error(cv::Error::StsAssert, "The data should normally be NULL!");
			// probably this is safe to do in such extreme case
			return stdAllocator->allocate(dims0, sizes, type, data, step, flags,
					usageFlags);
		}
		PyEnsureGIL gil;
		int depth = CV_MAT_DEPTH(type);
		int cn = CV_MAT_CN(type);
		const int f = (int) (sizeof(size_t) / 8);
		int typenum =
				depth == CV_8U ? NPY_UBYTE :
				depth == CV_8S ? NPY_BYTE :
				depth == CV_16U ? NPY_USHORT :
				depth == CV_16S ? NPY_SHORT :
				depth == CV_32S ? NPY_INT :
				depth == CV_32F ? NPY_FLOAT :
				depth == CV_64F ?
									NPY_DOUBLE :
									f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
		int i, dims = dims0;
		cv::AutoBuffer<npy_intp> _sizes(dims + 1);
		for (i = 0; i < dims; i++)
			_sizes[i] = sizes[i];
		if (cn > 1)
			_sizes[dims++] = cn;
		//printf("ff %d %ld %ld %ld\n", dims, _sizes[1], _sizes[2], _sizes[3]); fflush(stdout);
		PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
		if (!o)
			CV_Error_(cv::Error::StsError,
					("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
		return allocate(o, dims0, sizes, type, step);
	}

	bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags,
			cv::UMatUsageFlags usageFlags) const {
		return stdAllocator->allocate(u, accessFlags, usageFlags);
	}

	void deallocate(cv::UMatData* u) const {
		if (u) {
			PyEnsureGIL gil;
			PyObject* o = (PyObject*) u->userdata;
			Py_XDECREF(o);
			delete u;
		}
	}

	const MatAllocator* stdAllocator;
};

//===================   ALLOCATOR INITIALIZTION   ==================================================
NumpyAllocator g_numpyAllocator;

//===================   STANDALONE CONVERTER FUNCTIONS     =========================================


PyObject* fromMatToNDArray(const cv::Mat& m) {
	if (!m.data)
		Py_RETURN_NONE;
	cv::Mat temp, *p = (cv::Mat*) &m;
	if (!p->u || p->allocator != &g_numpyAllocator) {
		temp.allocator = &g_numpyAllocator;
		ERRWRAP2(m.copyTo(temp));
		p = &temp;
	}
	PyObject* o = (PyObject*) p->u->userdata;
	Py_INCREF(o);

	return o;
}
// end from



pybind11::object generate_image(unsigned resolution, double sun_elevation, double sun_azimuth, double visibility, double albedo) {
	PyObject *o = fromMatToNDArray(make_image(resolution, sun_elevation, sun_azimuth, visibility, albedo));
	return pybind11::reinterpret_steal<pybind11::object>(o);
}

void* init_numpy() {
    import_array();
    return NULL;
}

PYBIND11_MODULE(sky_image_generator, m) {
    init_numpy();

    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("generate_image", &generate_image, "Generate an image using the clear-sky model");
}
