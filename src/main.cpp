#include <pybind11/pybind11.h>
#include <cpu.h>
#include <gpu.h>
#include <net.h>
#include <option.h>
#include <blob.h>
#include <paramdict.h>

#include "pybind11_datareader.h"
#include "pybind11_allocator.h"
#include "pybind11_modelbin.h"
using namespace ncnn;

namespace py = pybind11;

PYBIND11_MODULE(pyncnn, m)
{
    py::class_<Allocator, PyAllocator<>>(m, "Allocator");
    py::class_<PoolAllocator, Allocator, PyAllocatorOther<PoolAllocator>>(m, "PoolAllocator")
        .def(py::init<>())
        .def("set_size_compare_ratio", &PoolAllocator::set_size_compare_ratio)
        .def("clear", &PoolAllocator::clear)
        .def("fastMalloc", &PoolAllocator::fastMalloc)
        .def("fastFree", &PoolAllocator::fastFree);
    py::class_<UnlockedPoolAllocator, Allocator, PyAllocatorOther<UnlockedPoolAllocator>>(m, "UnlockedPoolAllocator")
        .def(py::init<>())
        .def("set_size_compare_ratio", &UnlockedPoolAllocator::set_size_compare_ratio)
        .def("clear", &UnlockedPoolAllocator::clear)
        .def("fastMalloc", &UnlockedPoolAllocator::fastMalloc)
        .def("fastFree", &UnlockedPoolAllocator::fastFree);

    py::class_<DataReader, PyDataReader<>>(m, "DataReader")
        .def(py::init<>())
        .def("scan", &DataReader::scan)
        .def("read", &DataReader::read);
    py::class_<DataReaderFromEmpty, DataReader, PyDataReaderOther<DataReaderFromEmpty>>(m, "DataReaderFromEmpty")
        .def(py::init<>())
        .def("scan", &DataReaderFromEmpty::scan)
        .def("read", &DataReaderFromEmpty::read);

    py::class_<Blob>(m, "Blob")
        .def(py::init<>())
#if NCNN_STRING
        .def_readwrite("name", &Blob::name)
#endif // NCNN_STRING
        .def_readwrite("producer", &Blob::producer)
        .def_readwrite("consumers", &Blob::consumers);

    py::class_<ModelBin, PyModelBin<>>(m, "ModelBin");
    py::class_<ModelBinFromDataReader, ModelBin, PyModelBinOther<ModelBinFromDataReader>>(m, "ModelBinFromDataReader")
        .def(py::init<const DataReader&>())
        .def("load", &ModelBinFromDataReader::load);
    py::class_<ModelBinFromMatArray, ModelBin, PyModelBinOther<ModelBinFromMatArray>>(m, "ModelBinFromMatArray")
        .def(py::init<const Mat*>())
        .def("load", &ModelBinFromMatArray::load);

    py::class_<ParamDict>(m, "ParamDict")
        .def(py::init<>())
        .def("get", ( int( ParamDict::* )( int, int ) const )&ParamDict::get)
        .def("get", ( float( ParamDict::* )( int, float ) const )&ParamDict::get)
        .def("get", ( Mat(ParamDict::*)( int, const Mat& ) const )&ParamDict::get)
        .def("get", ( void( ParamDict::* )( int, int ) )&ParamDict::set)
        .def("get", ( void( ParamDict::* )( int, float ) )&ParamDict::set)
        .def("get", ( void( ParamDict::* )( int, const Mat& ) )&ParamDict::set);

    py::class_<Option>(m, "Option")
        .def(py::init<>())
        .def_readwrite("lightmode", &Option::lightmode)
        .def_readwrite("num_threads", &Option::num_threads)
        .def_readwrite("blob_allocator", &Option::blob_allocator)
        .def_readwrite("workspace_allocator", &Option::workspace_allocator)
#if NCNN_VULKAN
        .def_readwrite("blob_vkallocator", &Option::blob_vkallocator)
        .def_readwrite("workspace_vkallocator", &Option::workspace_vkallocator)
        .def_readwrite("staging_vkallocator", &Option::staging_vkallocator)
#endif // NCNN_VULKAN
        .def_readwrite("use_winograd_convolution", &Option::use_winograd_convolution)
        .def_readwrite("use_sgemm_convolution", &Option::use_sgemm_convolution)
        .def_readwrite("use_int8_inference", &Option::use_int8_inference)
        .def_readwrite("use_vulkan_compute", &Option::use_vulkan_compute)
        .def_readwrite("use_fp16_packed", &Option::use_fp16_packed)
        .def_readwrite("use_fp16_storage", &Option::use_fp16_storage)
        .def_readwrite("use_fp16_arithmetic", &Option::use_fp16_arithmetic)
        .def_readwrite("use_int8_storage", &Option::use_int8_storage)
        .def_readwrite("use_int8_arithmetic", &Option::use_int8_arithmetic)
        .def_readwrite("use_packing_layout", &Option::use_packing_layout);

    py::class_<Mat>(m, "Mat", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<int, size_t, Allocator*>(),
            py::arg("w") = 1,
            py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def(py::init<int, int, size_t, Allocator*>(),
            py::arg("w") = 1, py::arg("h") = 1,
            py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, size_t, Allocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
            py::arg("elemsize") = 4, py::arg("allocator") = nullptr)

        .def(py::init<int, size_t, int, Allocator*>(),
            py::arg("w") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, int, size_t, int, Allocator*>(),
            py::arg("w") = 1, py::arg("h") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, size_t, int, Allocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)

        .def(py::init<const Mat&>())

        .def(py::init<int, void*, size_t, Allocator*>(),
            py::arg("w") = 1, py::arg("data") = nullptr,
            py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def(py::init<int, int, void*, size_t, Allocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("data") = nullptr,
            py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, void*, size_t, Allocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("data") = nullptr,
            py::arg("elemsize") = 4, py::arg("allocator") = nullptr)

        .def(py::init<int, void*, size_t, int, Allocator*>(),
            py::arg("w") = 1, py::arg("data") = nullptr,
            py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, int, void*, size_t, int, Allocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("data") = nullptr,
            py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init<int, int, int, void*, size_t, int, Allocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("data") = nullptr,
            py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def(py::init([](py::buffer const b) {
            py::buffer_info info = b.request();
            if (info.ndim > 3)
                throw std::runtime_error("Incompatible buffer dims");

            printf("numpy dtype = %s\n", info.format.c_str());
            size_t elemsize = 4u;
            if (info.format == py::format_descriptor<float>::format() ||
                info.format == py::format_descriptor<int>::format()) {
                elemsize = 4u;
            } else if (info.format == "e") {
                elemsize = 2u;
            } else if (info.format == py::format_descriptor<int8_t>::format() ||
                info.format == py::format_descriptor<uint8_t>::format()) {
                elemsize = 1u;
            }

            Mat* v = nullptr;
            if (info.ndim == 1) {
                v = new Mat(info.shape[1], info.shape[0], info.ptr, elemsize);
            } else if (info.ndim == 2) {
                v = new Mat(info.shape[1], info.shape[0], info.ptr, elemsize);
            } else if (info.ndim == 3) {
                v = new Mat(info.shape[2], info.shape[1], info.shape[0], info.ptr, elemsize);
            }
            return v;
        }))
        .def_buffer([](Mat &m) -> py::buffer_info {
            std::string format;
            if (m.elemsize == 4) {  //float or int32, so what?
                format = py::format_descriptor<float>::format();
            } if (m.elemsize == 2) {  
                format = "e";
            } if (m.elemsize == 1) {  //int8 or uint8, so what?
                format = py::format_descriptor<int8_t>::format();
            }
            std::vector<ssize_t> shape;
            std::vector<ssize_t> strides;
            //todo strides not correct
            if (m.dims == 1) {
                shape.push_back(m.w);
                strides.push_back(m.cstep);
            } else if (m.dims == 2) {
                shape.push_back(m.h);
                shape.push_back(m.w);
                strides.push_back(m.cstep);
                strides.push_back(m.elemsize);
            } else if (m.dims == 3) {
                shape.push_back(m.c);
                shape.push_back(m.h);
                shape.push_back(m.w);
                strides.push_back(m.cstep);
                strides.push_back(m.elemsize);
            }
            return py::buffer_info(
                m.data,                             /* Pointer to buffer */
                m.elemsize,                      /* Size of one scalar */
                format,                             /* Python struct-style format descriptor */
                m.dims,                            /* Number of dimensions */
                shape,                              /* Buffer dimensions */
                strides                             /* Strides (in bytes) for each index */
            );
        })
        //todo assign
        //.def(py::self=py::self)
        .def("fill", ( void( Mat::* )( int ) )( &Mat::fill ))
        .def("fill", ( void( Mat::* )( float ) )( &Mat::fill ))
        .def("clone", ( Mat(Mat::*)( Allocator* ) )&Mat::clone, py::arg("allocator") = nullptr)
        .def("reshape", ( Mat(Mat::*)( int, Allocator* ) const )&Mat::reshape,
            py::arg("w") = 1, py::arg("allocator") = nullptr)
        .def("reshape", ( Mat(Mat::*)( int, int, Allocator* ) const )&Mat::reshape,
            py::arg("w") = 1, py::arg("h") = 1, py::arg("allocator") = nullptr)
        .def("reshape", ( Mat(Mat::*)( int, int, int, Allocator* ) const )&Mat::reshape,
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("allocator") = nullptr)
        .def("create", ( void( Mat::* )( int, size_t, Allocator* ) )&Mat::create,
            py::arg("w") = 1,
            py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def("create", ( void( Mat::* )( int, int, size_t, Allocator* ) )&Mat::create,
            py::arg("w") = 1, py::arg("h") = 1,
            py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def("create", ( void( Mat::* )( int, int, int, size_t, Allocator* ) )&Mat::create,
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
            py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def("create", ( void( Mat::* )( int, size_t, int, Allocator* ) )&Mat::create,
            py::arg("w") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def("create", ( void( Mat::* )( int, int, size_t, int, Allocator* ) )&Mat::create,
            py::arg("w") = 1, py::arg("h") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def("create", ( void( Mat::* )( int, int, int, size_t, int, Allocator* ) )&Mat::create,
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def("create_like", ( void( Mat::* )( const Mat&, Allocator* ) )&Mat::create_like,
            py::arg("m") = nullptr, py::arg("allocator") = nullptr)
#if NCNN_VULKAN
        .def("create_like", (void(Mat::*)(const VkMat&, Allocator*))&Mat::create_like,
        	py::arg("m") = nullptr, py::arg("allocator") = nullptr)
#endif // NCNN_VULKAN
        .def("addref", &Mat::addref)
        .def("release", &Mat::release)
        .def("empty", &Mat::empty)
        .def("total", &Mat::total)
        .def("channel", ( Mat(Mat::*)( int ) )&Mat::channel)
        .def("channel", ( const Mat(Mat::*)( int ) const )&Mat::channel)
        .def("row", ( float*( Mat::* )( int ) )&Mat::row)
        .def("row", ( const float*( Mat::* )( int ) const )&Mat::row)
        .def("channel_range", ( Mat(Mat::*)( int, int ) )&Mat::channel_range)
        .def("channel_range", ( const Mat(Mat::*)( int, int ) const )&Mat::channel_range)
        .def("row_range", ( Mat(Mat::*)( int, int ) )&Mat::row_range)
        .def("row_range", ( const Mat(Mat::*)( int, int ) const )&Mat::row_range)
        .def("range", ( Mat(Mat::*)( int, int ) )&Mat::range)
        .def("range", ( const Mat(Mat::*)( int, int ) const )&Mat::range)
        //todo __getitem__ in python crashed
        //.def("__getitem__", [](const Mat& m, size_t i) { return m[i]; })
        //todo convenient construct from pixel data
        //.def("from_pixels", (Mat(Mat::*)(const unsigned char*, int, int, int, Allocator*))&Mat::from_pixels)
        .def("to_pixels", ( void( Mat::* )( unsigned char*, int ) const )&Mat::to_pixels)
        .def("to_pixels", ( void( Mat::* )( unsigned char*, int, int ) const )&Mat::to_pixels)
        .def("to_pixels_resize", ( void( Mat::* )( unsigned char*, int, int, int ) const )&Mat::to_pixels_resize)
        .def("to_pixels_resize", ( void( Mat::* )( unsigned char*, int, int, int, int ) const )&Mat::to_pixels_resize)
        .def("substract_mean_normalize", &Mat::substract_mean_normalize)
        .def("from_float16", &Mat::from_float16)
        .def_readwrite("data", &Mat::data)
        .def_readwrite("refcount", &Mat::refcount)
        .def_readwrite("elemsize", &Mat::elemsize)
        .def_readwrite("elempack", &Mat::elempack)
        .def_readwrite("allocator", &Mat::allocator)
        .def_readwrite("dims", &Mat::dims)
        .def_readwrite("w", &Mat::w)
        .def_readwrite("h", &Mat::h)
        .def_readwrite("c", &Mat::c)
        .def_readwrite("cstep", &Mat::cstep);

#if NCNN_VULKAN
    py::class_<VkMat>(m, "VkMat")
        .def(py::init<>())
        .def(py::init<int, size_t, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1,
            py::arg("elemsize") = 4,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def(py::init<int, int, size_t, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("h") = 1,
            py::arg("elemsize") = 4,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def(py::init<int, int, int, size_t, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
            py::arg("elemsize") = 4,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)

        .def(py::init<int, size_t, int, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def(py::init<int, int, size_t, int, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("h") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def(py::init<int, int, int, size_t, int, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)

        .def(py::init<const VkMat&>())

        .def(py::init<int, VkBufferMemory*, size_t, size_t, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("data") = nullptr,
            py::arg("offset") = 0, py::arg("elemsize") = 4,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def(py::init<int, int, VkBufferMemory*, size_t, size_t, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("data") = nullptr,
            py::arg("offset") = 0, py::arg("elemsize") = 4,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def(py::init<int, int, int, VkBufferMemory*, size_t, size_t, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("data") = nullptr,
            py::arg("offset") = 0, py::arg("elemsize") = 4,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)

        .def(py::init<int, VkBufferMemory*, size_t, size_t, int, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("data") = nullptr,
            py::arg("offset") = 0, py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def(py::init<int, int, VkBufferMemory*, size_t, size_t, int, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("data") = nullptr,
            py::arg("offset") = 0, py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def(py::init<int, int, int, VkBufferMemory*, size_t, size_t, int, VkAllocator*, VkAllocator*>(),
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("data") = nullptr,
            py::arg("offset") = 0, py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)

        .def("create", ( void( VkMat::* )( int, size_t, VkAllocator*, VkAllocator* ) )&VkMat::create,
            py::arg("w") = 1,
            py::arg("elemsize") = 4,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def("create", ( void( VkMat::* )( int, int, size_t, VkAllocator*, VkAllocator* ) )&VkMat::create,
            py::arg("w") = 1, py::arg("h") = 1,
            py::arg("elemsize") = 4,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def("create", ( void( VkMat::* )( int, int, int, size_t, VkAllocator*, VkAllocator* ) )&VkMat::create,
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
            py::arg("elemsize") = 4,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)

        .def("create", ( void( VkMat::* )( int, size_t, int, VkAllocator*, VkAllocator* ) )&VkMat::create,
            py::arg("w") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def("create", ( void( VkMat::* )( int, int, size_t, int, VkAllocator*, VkAllocator* ) )&VkMat::create,
            py::arg("w") = 1, py::arg("h") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def("create", ( void( VkMat::* )( int, int, int, size_t, int, VkAllocator*, VkAllocator* ) )&VkMat::create,
            py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
            py::arg("elemsize") = 4, py::arg("elempack") = 1,
            py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)

        .def("create_like", ( void( VkMat::* )( const Mat&, VkAllocator*, VkAllocator* ) )&VkMat::create_like)
        .def("create_like", ( void( VkMat::* )( const VkMat&, VkAllocator*, VkAllocator* ) )&VkMat::create_like)
        .def("prepare_staging_buffer", &VkMat::prepare_staging_buffer)
        .def("discard_staging_buffer", &VkMat::discard_staging_buffer)
        .def("upload", &VkMat::upload)
        .def("download", &VkMat::download)
        .def("mapped", &VkMat::mapped)
        .def("mapped_ptr", &VkMat::mapped_ptr)
        .def("addref", &VkMat::addref)
        .def("release", &VkMat::release)
        .def("empty", &VkMat::empty)
        .def("total", &VkMat::total)
        .def("channel", ( VkMat(VkMat::*)( int ) )&VkMat::channel)
        .def("channel", ( const VkMat(VkMat::*)( int ) const )&VkMat::channel)
        .def("channel_range", ( VkMat(VkMat::*)( int, int ) )&VkMat::channel_range)
        .def("channel_range", ( const VkMat(VkMat::*)( int, int ) const )&VkMat::channel_range)
        .def("row_range", ( VkMat(VkMat::*)( int, int ) )&VkMat::row_range)
        .def("row_range", ( const VkMat(VkMat::*)( int, int ) const )&VkMat::row_range)
        .def("range", ( VkMat(VkMat::*)( int, int ) )&VkMat::range)
        .def("range", ( const VkMat(VkMat::*)( int, int ) const )&VkMat::range)
        //.def("buffer", &VkMat::buffer)
        .def("buffer_offset", &VkMat::buffer_offset)
        //.def("staging_buffer", &VkMat::staging_buffer)
        .def("staging_buffer_offset", &VkMat::staging_buffer_offset)
        .def_readwrite("data", &VkMat::data)
        .def_readwrite("offset", &VkMat::offset)
        .def_readwrite("staging_data", &VkMat::staging_data)
        .def_readwrite("refcount", &VkMat::refcount)
        .def_readwrite("staging_refcount", &VkMat::staging_refcount)
        .def_readwrite("elemsize", &VkMat::elemsize)
        .def_readwrite("elempack", &VkMat::elempack)
        .def_readwrite("allocator", &VkMat::allocator)
        .def_readwrite("staging_allocator", &VkMat::staging_allocator)
        .def_readwrite("dims", &VkMat::dims)
        .def_readwrite("w", &VkMat::w)
        .def_readwrite("h", &VkMat::h)
        .def_readwrite("c", &VkMat::c)
        .def_readwrite("cstep", &VkMat::cstep);


    py::class_<VkImageMat>(m, "VkImageMat")
        .def(py::init<>())
        .def(py::init<int, int, VkFormat, VkImageAllocator*>())
        .def(py::init<const VkImageMat&>())
        .def(py::init<int, int, VkImageMemory*, VkFormat, VkImageAllocator*>())
        .def("create", &VkImageMat::create)
        .def("addref", &VkImageMat::addref)
        .def("release", &VkImageMat::release)
        .def("empty", &VkImageMat::empty)
        .def("total", &VkImageMat::total)
        //.def("image", &VkImageMat::image)
        //.def("imageview", &VkImageMat::imageview)
        .def_readwrite("data", &VkImageMat::data)
        .def_readwrite("refcount", &VkImageMat::refcount)
        .def_readwrite("allocator", &VkImageMat::allocator)
        .def_readwrite("width", &VkImageMat::width)
        .def_readwrite("height", &VkImageMat::height)
        .def_readwrite("format", &VkImageMat::format);
#endif //NCNN_VULKAN

    py::class_<Extractor>(m, "Extractor")
        .def("set_light_mode", &Extractor::set_light_mode)
        .def("set_num_threads", &Extractor::set_num_threads)
        .def("set_blob_allocator", &Extractor::set_blob_allocator)
        .def("set_blob_allocator", &Extractor::set_workspace_allocator)
#if NCNN_VULKAN
        .def("set_vulkan_compute", &Extractor::set_vulkan_compute)
        .def("set_blob_vkallocator", &Extractor::set_blob_vkallocator)
        .def("set_workspace_vkallocator", &Extractor::set_workspace_vkallocator)
        .def("set_staging_vkallocator", &Extractor::set_staging_vkallocator)
#endif // NCNN_VULKAN
#if NCNN_STRING
        .def("input", ( int( Extractor::* )( const char*, const Mat& ) )&Extractor::input)
        .def("extract", ( int( Extractor::* )( const char*, Mat& ) )&Extractor::extract)
#endif
        .def("input", ( int( Extractor::* )( int, const Mat& ) )&Extractor::input)
        .def("extract", ( int( Extractor::* )( int, Mat& ) )&Extractor::extract)
#if NCNN_VULKAN
        .def("input", ( int( Extractor::* )( const char*, const VkMat& ) )&Extractor::input)
        .def("extract", ( int( Extractor::* )( const char*, VkMat&, VkCompute& ) )&Extractor::extract)
#if NCNN_STRING
#endif // NCNN_STRING
        .def("input", ( int( Extractor::* )( int, const VkMat& ) )&Extractor::input)
        .def("extract", ( int( Extractor::* )( int, VkMat&, VkCompute& ) )&Extractor::extract)
#endif // NCNN_VULKAN
        ;

    py::class_<Net>(m, "Net")
        .def(py::init<>())
        .def_readwrite("opt", &Net::opt)
#if NCNN_VULKAN
        .def("set_vulkan_device", ( void( Net::* )( int ) )&Net::set_vulkan_device)
        .def("set_vulkan_device", ( void( Net::* )( const VulkanDevice* ) )&Net::set_vulkan_device)
        .def("vulkan_device", &Net::vulkan_device)
#endif // NCNN_VULKAN
        //tode register_custom_layer
#if NCNN_STRING
        .def("load_param", ( int( Net::* )( const DataReader& ) )&Net::load_param)
#endif // NCNN_STRING
        .def("load_param_bin", ( int( Net::* )( const DataReader& ) )&Net::load_param_bin)
        .def("load_model", ( int( Net::* )( const DataReader& ) )&Net::load_model)

#if NCNN_STDIO
#if NCNN_STRING
        .def("load_param", ( int( Net::* )( const char* ) )&Net::load_param)
        .def("load_param_mem", ( int( Net::* )( const char* ) )&Net::load_param_mem)
#endif // NCNN_STRING
        .def("load_param_bin", ( int( Net::* )( const char* ) )&Net::load_param_bin)
        .def("load_model", ( int( Net::* )( const char* ) )&Net::load_model)
#endif // NCNN_STDIO

        //todo load from memory
        //.def("load_param", (int (Net::*)(const unsigned char*))(&Net::load_param))
        //.def("load_model", (int (Net::*)(const unsigned char*))(&Net::load_model))

        .def("clear", &Net::clear)
        .def("create_extractor", &Net::create_extractor);

    m.def("cpu_support_arm_neon", &cpu_support_arm_neon);
    m.def("cpu_support_arm_vfpv4", &cpu_support_arm_vfpv4);
    m.def("cpu_support_arm_asimdhp", &cpu_support_arm_asimdhp);
    m.def("get_cpu_count", &get_cpu_count);
    m.def("get_cpu_powersave", &get_cpu_powersave);
    m.def("set_cpu_powersave", &set_cpu_powersave);
    m.def("get_omp_num_threads", &get_omp_num_threads);
    m.def("set_omp_num_threads", &set_omp_num_threads);
    m.def("get_omp_dynamic", &get_omp_dynamic);
    m.def("set_omp_dynamic", &set_omp_dynamic);

#if NCNN_VULKAN
    m.def("create_gpu_instance", &create_gpu_instance);
    m.def("destroy_gpu_instance", &destroy_gpu_instance);
    m.def("get_gpu_count", &get_gpu_count);
    m.def("get_default_gpu_index", &get_default_gpu_index);
    m.def("get_gpu_info", &get_gpu_info, py::arg("device_index") = get_default_gpu_index());
    m.def("get_gpu_device", &get_gpu_device, py::arg("device_index") = get_default_gpu_index());

    py::class_<VkBufferMemory>(m, "VkBufferMemory")
        .def(py::init<>())
        //.def_readwrite("buffer", &VkBufferMemory::buffer)
        .def_readwrite("offset", &VkBufferMemory::offset)
        .def_readwrite("capacity", &VkBufferMemory::capacity)
        //.def_readwrite("memory", &VkBufferMemory::memory)
        .def_readwrite("mapped_ptr", &VkBufferMemory::mapped_ptr)
        .def_readwrite("state", &VkBufferMemory::state)
        .def_readwrite("refcount", &VkBufferMemory::refcount);

    py::class_<VkAllocator, PyVkAllocator<>>(m, "VkAllocator")
        .def_readwrite("vkdev", &VkAllocator::vkdev)
        .def_readwrite("memory_type_index", &VkAllocator::memory_type_index)
        .def_readwrite("mappable", &VkAllocator::mappable)
        .def_readwrite("coherent", &VkAllocator::coherent);
    py::class_<VkBlobBufferAllocator, VkAllocator, PyVkAllocatorOther<VkBlobBufferAllocator>>(m, "VkBlobBufferAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("clear", &VkBlobBufferAllocator::clear)
        .def("fastMalloc", &VkBlobBufferAllocator::fastMalloc)
        .def("fastFree", &VkBlobBufferAllocator::fastFree);
    py::class_<VkWeightBufferAllocator, VkAllocator, PyVkAllocatorOther<VkWeightBufferAllocator>>(m, "VkWeightBufferAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("clear", &VkWeightBufferAllocator::clear)
        .def("fastMalloc", &VkWeightBufferAllocator::fastMalloc)
        .def("fastFree", &VkWeightBufferAllocator::fastFree);
    py::class_<VkStagingBufferAllocator, VkAllocator, PyVkAllocatorOther<VkStagingBufferAllocator>>(m, "VkStagingBufferAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("set_size_compare_ratio", &VkStagingBufferAllocator::set_size_compare_ratio)
        .def("clear", &VkStagingBufferAllocator::clear)
        .def("fastMalloc", &VkStagingBufferAllocator::fastMalloc)
        .def("fastFree", &VkStagingBufferAllocator::fastFree);
    py::class_<VkWeightStagingBufferAllocator, VkAllocator, PyVkAllocatorOther<VkWeightStagingBufferAllocator>>(m, "VkWeightStagingBufferAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("fastMalloc", &VkWeightStagingBufferAllocator::fastMalloc)
        .def("fastFree", &VkWeightStagingBufferAllocator::fastFree);

    py::class_<VkImageMemory>(m, "VkImageMemory")
        .def(py::init<>())
        //.def_readwrite("image", &VkImageMemory::image)
        //.def_readwrite("imageview", &VkImageMemory::imageview)
        //.def_readwrite("memory", &VkImageMemory::memory)
        .def_readwrite("state", &VkImageMemory::state)
        .def_readwrite("refcount", &VkImageMemory::refcount);

    py::class_<VkImageAllocator, VkAllocator, PyVkImageAllocator<>>(m, "VkImageAllocator")
        .def("clear", &VkImageAllocator::clear);
    py::class_<VkSimpleImageAllocator, VkImageAllocator, PyVkImageAllocatorOther<VkSimpleImageAllocator>>(m, "VkSimpleImageAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("fastMalloc", &VkSimpleImageAllocator::fastMalloc)
        .def("fastFree", &VkSimpleImageAllocator::fastFree);

    py::class_<GpuInfo>(m, "GpuInfo")
        .def(py::init<>())
        //.def_readwrite("physical_device", &GpuInfo::physical_device)
        //.def_readwrite("physicalDeviceMemoryProperties", &GpuInfo::physicalDeviceMemoryProperties)
        .def_readwrite("api_version", &GpuInfo::api_version)
        .def_readwrite("driver_version", &GpuInfo::driver_version)
        .def_readwrite("vendor_id", &GpuInfo::vendor_id)
        .def_readwrite("device_id", &GpuInfo::device_id)
        //.def_readwrite("pipeline_cache_uuid", &GpuInfo::pipeline_cache_uuid)
        .def_readwrite("type", &GpuInfo::type)
        .def_readwrite("max_shared_memory_size", &GpuInfo::max_shared_memory_size)
        //.def_readwrite("max_workgroup_count", &GpuInfo::max_workgroup_count)
        .def_readwrite("max_workgroup_invocations", &GpuInfo::max_workgroup_invocations)
        //.def_readwrite("max_workgroup_size", &GpuInfo::max_workgroup_size)
        .def_readwrite("memory_map_alignment", &GpuInfo::memory_map_alignment)
        .def_readwrite("buffer_offset_alignment", &GpuInfo::buffer_offset_alignment)
        .def_readwrite("non_coherent_atom_size", &GpuInfo::non_coherent_atom_size)
        .def_readwrite("timestamp_period", &GpuInfo::timestamp_period)
        .def_readwrite("compute_queue_family_index", &GpuInfo::compute_queue_family_index)
        .def_readwrite("graphics_queue_family_index", &GpuInfo::graphics_queue_family_index)
        .def_readwrite("transfer_queue_family_index", &GpuInfo::transfer_queue_family_index)
        .def_readwrite("compute_queue_count", &GpuInfo::compute_queue_count)
        .def_readwrite("graphics_queue_count", &GpuInfo::graphics_queue_count)
        .def_readwrite("transfer_queue_count", &GpuInfo::transfer_queue_count)
        .def_readwrite("bug_local_size_spec_const", &GpuInfo::bug_local_size_spec_const)
        .def_readwrite("support_fp16_packed", &GpuInfo::support_fp16_packed)
        .def_readwrite("support_fp16_storage", &GpuInfo::support_fp16_storage)
        .def_readwrite("support_fp16_arithmetic", &GpuInfo::support_fp16_arithmetic)
        .def_readwrite("support_int8_storage", &GpuInfo::support_int8_storage)
        .def_readwrite("support_int8_arithmetic", &GpuInfo::support_int8_arithmetic)
        .def_readwrite("support_ycbcr_conversion", &GpuInfo::support_ycbcr_conversion)
        .def_readwrite("support_VK_KHR_8bit_storage", &GpuInfo::support_VK_KHR_8bit_storage)
        .def_readwrite("support_VK_KHR_16bit_storage", &GpuInfo::support_VK_KHR_16bit_storage)
        .def_readwrite("support_VK_KHR_bind_memory2", &GpuInfo::support_VK_KHR_bind_memory2)
        .def_readwrite("support_VK_KHR_dedicated_allocation", &GpuInfo::support_VK_KHR_dedicated_allocation)
        .def_readwrite("support_VK_KHR_descriptor_update_template", &GpuInfo::support_VK_KHR_descriptor_update_template)
        .def_readwrite("support_VK_KHR_external_memory", &GpuInfo::support_VK_KHR_external_memory)
        .def_readwrite("support_VK_KHR_get_memory_requirements2", &GpuInfo::support_VK_KHR_get_memory_requirements2)
        .def_readwrite("support_VK_KHR_maintenance1", &GpuInfo::support_VK_KHR_maintenance1)
        .def_readwrite("support_VK_KHR_push_descriptor", &GpuInfo::support_VK_KHR_push_descriptor)
        .def_readwrite("support_VK_KHR_sampler_ycbcr_conversion", &GpuInfo::support_VK_KHR_sampler_ycbcr_conversion)
        .def_readwrite("support_VK_KHR_shader_float16_int8", &GpuInfo::support_VK_KHR_shader_float16_int8)
        .def_readwrite("support_VK_KHR_shader_float_controls", &GpuInfo::support_VK_KHR_shader_float_controls)
        .def_readwrite("support_VK_KHR_storage_buffer_storage_class", &GpuInfo::support_VK_KHR_storage_buffer_storage_class)
        .def_readwrite("support_VK_KHR_swapchain", &GpuInfo::support_VK_KHR_swapchain)
        .def_readwrite("support_VK_EXT_queue_family_foreign", &GpuInfo::support_VK_EXT_queue_family_foreign);

    py::class_<VulkanDevice>(m, "VulkanDevice")
        .def(py::init<int>(), py::arg("device_index") = get_default_gpu_index())
        //.def_readonly("info", &VulkanDevice::info)
        //.def("get_shader_module", &VulkanDevice::get_shader_module)
        //.def("create_shader_module", &VulkanDevice::create_shader_module)
        //.def("compile_shader_module", &VulkanDevice::compile_shader_module)
        //.def("compile_shader_module", &VulkanDevice::compile_shader_module)
        .def("find_memory_index", &VulkanDevice::find_memory_index)
        .def("is_mappable", &VulkanDevice::is_mappable)
        .def("is_coherent", &VulkanDevice::is_coherent)
        //.def("acquire_queue", &VulkanDevice::acquire_queue)
        //.def("reclaim_queue", &VulkanDevice::reclaim_queue)
        .def("acquire_blob_allocator", &VulkanDevice::acquire_blob_allocator)
        .def("reclaim_blob_allocator", &VulkanDevice::reclaim_blob_allocator)
        .def("acquire_staging_allocator", &VulkanDevice::acquire_staging_allocator)
        .def("reclaim_staging_allocator", &VulkanDevice::reclaim_staging_allocator)
        //tode not compelete
        ;

    py::class_<Command>(m, "Command")
        .def(py::init<const VulkanDevice*, uint32_t>());
    py::class_<VkCompute, Command>(m, "VkCompute")
        .def(py::init<const VulkanDevice*>())
        .def("record_upload", &VkCompute::record_upload)
        .def("record_download", &VkCompute::record_download)
        .def("record_clone", &VkCompute::record_clone)
        .def("record_copy_region", &VkCompute::record_copy_region)
        .def("record_copy_regions", &VkCompute::record_copy_regions)
        .def("record_pipeline", &VkCompute::record_pipeline)
        .def("record_download", &VkCompute::record_download)

#if NCNN_BENCHMARK
        .def("record_write_timestamp", &VkCompute::record_write_timestamp)
#endif // NCNN_BENCHMARK

        .def("record_queue_transfer_acquire", &VkCompute::record_queue_transfer_acquire)
        .def("submit_and_wait", &VkCompute::submit_and_wait)
        .def("reset", &VkCompute::reset)

#if NCNN_BENCHMARK
        .def("create_query_pool", &VkCompute::create_query_pool)
        .def("get_query_pool_results", &VkCompute::get_query_pool_results)
#endif // NCNN_BENCHMARK
        ;

    py::class_<VkTransfer, Command>(m, "VkTransfer")
        .def(py::init<const VulkanDevice*>())
        .def("record_upload", &VkTransfer::record_upload)
        .def("submit_and_wait", &VkTransfer::submit_and_wait);

#endif // NCNN_VULKAN

    m.doc() = R"pbdoc(
        ncnn python wrapper
        -----------------------
        .. currentmodule:: pyncnn
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}