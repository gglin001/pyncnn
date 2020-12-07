#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <cpu.h>
#include <gpu.h>
#include <net.h>
#include <option.h>
#include <blob.h>
#include <paramdict.h>
#include <layer_declaration_install.h>

#include "pybind11_mat.h"
#include "pybind11_datareader.h"
#include "pybind11_allocator.h"
#include "pybind11_modelbin.h"
#include "pybind11_layer.h"

using namespace ncnn;

namespace py = pybind11;

// // todo multi custom layer need multi g_layer_creator, but now only one???
// static std::function<Layer*()> g_layer_creator = nullptr;
// ncnn::Layer* LayerCreator()
// {
//     return new ::Layer(g_layer_creator);
// }

PYBIND11_MODULE(ncnn, m)
{
    py::class_<Allocator, PyAllocator<> >(m, "Allocator");
    py::class_<PoolAllocator, Allocator, PyAllocatorOther<PoolAllocator> >(m, "PoolAllocator")
        .def(py::init<>())
        .def("set_size_compare_ratio", &PoolAllocator::set_size_compare_ratio)
        .def("clear", &PoolAllocator::clear)
        .def("fastMalloc", &PoolAllocator::fastMalloc)
        .def("fastFree", &PoolAllocator::fastFree);
    py::class_<UnlockedPoolAllocator, Allocator, PyAllocatorOther<UnlockedPoolAllocator> >(m, "UnlockedPoolAllocator")
        .def(py::init<>())
        .def("set_size_compare_ratio", &UnlockedPoolAllocator::set_size_compare_ratio)
        .def("clear", &UnlockedPoolAllocator::clear)
        .def("fastMalloc", &UnlockedPoolAllocator::fastMalloc)
        .def("fastFree", &UnlockedPoolAllocator::fastFree);

    py::class_<DataReader, PyDataReader<> >(m, "DataReader")
        .def(py::init<>())
        .def("scan", &DataReader::scan)
        .def("read", &DataReader::read);
    py::class_<DataReaderFromEmpty, DataReader, PyDataReaderOther<DataReaderFromEmpty> >(m, "DataReaderFromEmpty")
        .def(py::init<>())
        .def("scan", &DataReaderFromEmpty::scan)
        .def("read", &DataReaderFromEmpty::read);

    py::class_<Blob>(m, "Blob")
        .def(py::init<>())
#if NCNN_STRING
        .def_readwrite("name", &Blob::name)
#endif // NCNN_STRING
        .def_readwrite("producer", &Blob::producer)
        .def_readwrite("consumers", &Blob::consumers)
        .def_readwrite("shape", &Blob::shape);

    py::class_<ModelBin, PyModelBin<> >(m, "ModelBin");
    py::class_<ModelBinFromDataReader, ModelBin, PyModelBinOther<ModelBinFromDataReader> >(m, "ModelBinFromDataReader")
        .def(py::init<const DataReader&>())
        .def("load", &ModelBinFromDataReader::load);
    py::class_<ModelBinFromMatArray, ModelBin, PyModelBinOther<ModelBinFromMatArray> >(m, "ModelBinFromMatArray")
        .def(py::init<const Mat*>())
        .def("load", &ModelBinFromMatArray::load);

    py::class_<ParamDict>(m, "ParamDict")
        .def(py::init<>())
        .def("get", (int (ParamDict::*)(int, int) const) & ParamDict::get)
        .def("get", (float (ParamDict::*)(int, float) const) & ParamDict::get)
        .def("get", (Mat(ParamDict::*)(int, const Mat&) const) & ParamDict::get)
        .def("set", (void (ParamDict::*)(int, int)) & ParamDict::set)
        .def("set", (void (ParamDict::*)(int, float)) & ParamDict::set)
        .def("set", (void (ParamDict::*)(int, const Mat&)) & ParamDict::set);

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
        .def_readwrite("use_winograd_convolution", &Option::openmp_blocktime)
        .def_readwrite("use_winograd_convolution", &Option::use_winograd_convolution)
        .def_readwrite("use_sgemm_convolution", &Option::use_sgemm_convolution)
        .def_readwrite("use_int8_inference", &Option::use_int8_inference)
        .def_readwrite("use_vulkan_compute", &Option::use_vulkan_compute)
        .def_readwrite("use_fp16_packed", &Option::use_fp16_packed)
        .def_readwrite("use_fp16_storage", &Option::use_fp16_storage)
        .def_readwrite("use_fp16_arithmetic", &Option::use_fp16_arithmetic)
        .def_readwrite("use_int8_storage", &Option::use_int8_storage)
        .def_readwrite("use_int8_arithmetic", &Option::use_int8_arithmetic)
        .def_readwrite("use_packing_layout", &Option::use_packing_layout)
        .def_readwrite("use_shader_pack8", &Option::use_shader_pack8)
        .def_readwrite("use_shader_pack8", &Option::use_image_storage)
        .def_readwrite("use_bf16_storage", &Option::use_bf16_storage)
        .def_readwrite("use_bf16_storage", &Option::use_weight_fp16_storage);

    py::class_<Mat> mat(m, "Mat", py::buffer_protocol());
    mat.def(py::init<>())
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
            {
                throw std::runtime_error("Incompatible buffer dims");
            }

            //printf("numpy dtype = %s\n", info.format.c_str());
            size_t elemsize = 4u;
            if (info.format == py::format_descriptor<double>::format())
            {
                elemsize = 8u;
            }
            if (info.format == py::format_descriptor<float>::format() || info.format == py::format_descriptor<int>::format())
            {
                elemsize = 4u;
            }
            else if (info.format == "e")
            {
                elemsize = 2u;
            }
            else if (info.format == py::format_descriptor<int8_t>::format() || info.format == py::format_descriptor<uint8_t>::format())
            {
                elemsize = 1u;
            }

            Mat* v = nullptr;
            if (info.ndim == 1)
            {
                v = new Mat((int)info.shape[0], info.ptr, elemsize);
            }
            else if (info.ndim == 2)
            {
                v = new Mat((int)info.shape[1], (int)info.shape[0], info.ptr, elemsize);
            }
            else if (info.ndim == 3)
            {
                v = new Mat((int)info.shape[2], (int)info.shape[1], (int)info.shape[0], info.ptr, elemsize);
            }
            return v;
        }))
        .def_buffer([](Mat& m) -> py::buffer_info {
            std::string format = get_mat_format(m);
            std::vector<ssize_t> shape;
            std::vector<ssize_t> strides;
            //todo strides not correct
            if (m.dims == 1)
            {
                shape.push_back(m.w);
                strides.push_back(m.elemsize);
            }
            else if (m.dims == 2)
            {
                shape.push_back(m.h);
                shape.push_back(m.w);
                strides.push_back(m.w * m.elemsize);
                strides.push_back(m.elemsize);
            }
            else if (m.dims == 3)
            {
                shape.push_back(m.c);
                shape.push_back(m.h);
                shape.push_back(m.w);
                strides.push_back(m.cstep * m.elemsize);
                strides.push_back(m.w * m.elemsize);
                strides.push_back(m.elemsize);
            }
            return py::buffer_info(
                m.data,     /* Pointer to buffer */
                m.elemsize, /* Size of one scalar */
                format,     /* Python struct-style format descriptor */
                m.dims,     /* Number of dimensions */
                shape,      /* Buffer dimensions */
                strides     /* Strides (in bytes) for each index */
            );
        })
        //todo assign
        //.def(py::self=py::self)
        .def("fill", (void (Mat::*)(int))(&Mat::fill))
        .def("fill", (void (Mat::*)(float))(&Mat::fill))
        .def("clone", (Mat(Mat::*)(Allocator*)) & Mat::clone, py::arg("allocator") = nullptr)
        .def("reshape", (Mat(Mat::*)(int, Allocator*) const) & Mat::reshape,
             py::arg("w") = 1, py::arg("allocator") = nullptr)
        .def("reshape", (Mat(Mat::*)(int, int, Allocator*) const) & Mat::reshape,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("allocator") = nullptr)
        .def("reshape", (Mat(Mat::*)(int, int, int, Allocator*) const) & Mat::reshape,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1, py::arg("allocator") = nullptr)
        .def("create", (void (Mat::*)(int, size_t, Allocator*)) & Mat::create,
             py::arg("w") = 1,
             py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def("create", (void (Mat::*)(int, int, size_t, Allocator*)) & Mat::create,
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def("create", (void (Mat::*)(int, int, int, size_t, Allocator*)) & Mat::create,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4, py::arg("allocator") = nullptr)
        .def("create", (void (Mat::*)(int, size_t, int, Allocator*)) & Mat::create,
             py::arg("w") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def("create", (void (Mat::*)(int, int, size_t, int, Allocator*)) & Mat::create,
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def("create", (void (Mat::*)(int, int, int, size_t, int, Allocator*)) & Mat::create,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1, py::arg("allocator") = nullptr)
        .def("create_like", (void (Mat::*)(const Mat&, Allocator*)) & Mat::create_like,
             py::arg("m") = nullptr, py::arg("allocator") = nullptr)
#if NCNN_VULKAN
        .def("create_like", (void (Mat::*)(const VkMat&, Allocator*)) & Mat::create_like,
             py::arg("m") = nullptr, py::arg("allocator") = nullptr)
#endif // NCNN_VULKAN
        .def("addref", &Mat::addref)
        .def("release", &Mat::release)
        .def("empty", &Mat::empty)
        .def("total", &Mat::total)
        .def("channel", (Mat(Mat::*)(int)) & Mat::channel)
        .def("channel", (const Mat (Mat::*)(int) const) & Mat::channel)
        .def("row", [](Mat& m, int y) {
            if (m.elemsize != 4)
            {
                throw std::runtime_error("only float/int32 type mat.row support now");
            }
            return py::array_t<float>(m.w, m.row(y));
        })
        .def("channel_range", (Mat(Mat::*)(int, int)) & Mat::channel_range)
        .def("channel_range", (const Mat (Mat::*)(int, int) const) & Mat::channel_range)
        .def("row_range", (Mat(Mat::*)(int, int)) & Mat::row_range)
        .def("row_range", (const Mat (Mat::*)(int, int) const) & Mat::row_range)
        .def("range", (Mat(Mat::*)(int, int)) & Mat::range)
        .def("range", (const Mat (Mat::*)(int, int) const) & Mat::range)
        .def("__getitem__", [](const Mat& m, size_t i) {
            return m[i];
        })
        .def("__setitem__", [](Mat& m, size_t i, float v) {
            m[i] = v;
        })
        //convenient construct from pixel data
        .def_static("from_pixels", [](py::buffer const b, int type, int w, int h) {
            return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h);
        })
        .def_static("from_pixels", [](py::buffer const b, int type, int w, int h, Allocator* allocator) {
            return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h, allocator);
        })
        .def_static("from_pixels", [](py::buffer const b, int type, int w, int h, int stride) {
            return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h, stride);
        })
        .def_static("from_pixels", [](py::buffer const b, int type, int w, int h, int stride, Allocator* allocator) {
            return Mat::from_pixels((const unsigned char*)b.request().ptr, type, w, h, stride, allocator);
        })
        .def_static("from_pixels_resize", [](py::buffer const b, int type, int w, int h, int target_width, int target_height) {
            return Mat::from_pixels_resize((const unsigned char*)b.request().ptr, type, w, h, target_width, target_height);
        })
        .def_static("from_pixels_resize", [](py::buffer const b, int type, int w, int h, int target_width, int target_height, Allocator* allocator) {
            return Mat::from_pixels_resize((const unsigned char*)b.request().ptr, type, w, h, target_width, target_height, allocator);
        })
        .def_static("from_pixels_resize", [](py::buffer const b, int type, int w, int h, int stride, int target_width, int target_height) {
            return Mat::from_pixels_resize((const unsigned char*)b.request().ptr, type, w, h, stride, target_width, target_height);
        })
        .def_static("from_pixels_resize", [](py::buffer const b, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator) {
            return Mat::from_pixels_resize((const unsigned char*)b.request().ptr, type, w, h, stride, target_width, target_height, allocator);
        })
        .def("to_pixels", (void (Mat::*)(unsigned char*, int) const) & Mat::to_pixels)
        .def("to_pixels", (void (Mat::*)(unsigned char*, int, int) const) & Mat::to_pixels)
        .def("to_pixels_resize", (void (Mat::*)(unsigned char*, int, int, int) const) & Mat::to_pixels_resize)
        .def("to_pixels_resize", (void (Mat::*)(unsigned char*, int, int, int, int) const) & Mat::to_pixels_resize)
        .def("substract_mean_normalize", [](Mat& mat, std::vector<float>& mean, std::vector<float>& norm) {
            return mat.substract_mean_normalize(mean.size() > 0 ? &mean[0] : 0, norm.size() > 0 ? &norm[0] : 0);
        })
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
        .def_readwrite("cstep", &Mat::cstep)
        .def("__repr__", [](const Mat& m) {
            char buf[256] = {0};
            sprintf(buf, "<ncnn.Mat w=%d h=%d c=%d dims=%d cstep=%ld elemsize=%ld elempack=%d\n\trefcount=%d data=0x%p allocator=0x%p>",
                    m.w, m.h, m.c, m.dims, m.cstep, m.elemsize, m.elempack, m.refcount ? *m.refcount : 0, m.data, m.allocator);
            return std::string(buf);
        })
        .def("shape", &Mat::shape);

    py::enum_<ncnn::Mat::PixelType>(mat, "PixelType")
        .value("PIXEL_CONVERT_SHIFT", ncnn::Mat::PixelType::PIXEL_CONVERT_SHIFT)
        .value("PIXEL_FORMAT_MASK", ncnn::Mat::PixelType::PIXEL_FORMAT_MASK)
        .value("PIXEL_CONVERT_MASK", ncnn::Mat::PixelType::PIXEL_CONVERT_MASK)

        .value("PIXEL_RGB", ncnn::Mat::PixelType::PIXEL_RGB)
        .value("PIXEL_BGR", ncnn::Mat::PixelType::PIXEL_BGR)
        .value("PIXEL_GRAY", ncnn::Mat::PixelType::PIXEL_GRAY)
        .value("PIXEL_RGBA", ncnn::Mat::PixelType::PIXEL_RGBA)
        .value("PIXEL_BGRA", ncnn::Mat::PixelType::PIXEL_BGRA)

        .value("PIXEL_RGB2BGR", ncnn::Mat::PixelType::PIXEL_RGB2BGR)
        .value("PIXEL_RGB2GRAY", ncnn::Mat::PixelType::PIXEL_RGB2GRAY)
        .value("PIXEL_RGB2RGBA", ncnn::Mat::PixelType::PIXEL_RGB2RGBA)
        .value("PIXEL_RGB2BGRA", ncnn::Mat::PixelType::PIXEL_RGB2BGRA)

        .value("PIXEL_BGR2RGB", ncnn::Mat::PixelType::PIXEL_BGR2RGB)
        .value("PIXEL_BGR2GRAY", ncnn::Mat::PixelType::PIXEL_BGR2GRAY)
        .value("PIXEL_BGR2RGBA", ncnn::Mat::PixelType::PIXEL_BGR2RGBA)
        .value("PIXEL_BGR2BGRA", ncnn::Mat::PixelType::PIXEL_BGR2BGRA)

        .value("PIXEL_GRAY2RGB", ncnn::Mat::PixelType::PIXEL_GRAY2RGB)
        .value("PIXEL_GRAY2BGR", ncnn::Mat::PixelType::PIXEL_GRAY2BGR)
        .value("PIXEL_GRAY2RGBA", ncnn::Mat::PixelType::PIXEL_GRAY2RGBA)
        .value("PIXEL_GRAY2BGRA", ncnn::Mat::PixelType::PIXEL_GRAY2BGRA)

        .value("PIXEL_RGBA2RGB", ncnn::Mat::PixelType::PIXEL_RGBA2RGB)
        .value("PIXEL_RGBA2BGR", ncnn::Mat::PixelType::PIXEL_RGBA2BGR)
        .value("PIXEL_RGBA2GRAY", ncnn::Mat::PixelType::PIXEL_RGBA2GRAY)
        .value("PIXEL_RGBA2BGRA", ncnn::Mat::PixelType::PIXEL_RGBA2BGRA)

        .value("PIXEL_BGRA2RGB", ncnn::Mat::PixelType::PIXEL_BGRA2RGB)
        .value("PIXEL_BGRA2BGR", ncnn::Mat::PixelType::PIXEL_BGRA2BGR)
        .value("PIXEL_BGRA2GRAY", ncnn::Mat::PixelType::PIXEL_BGRA2GRAY)
        .value("PIXEL_BGRA2RGBA", ncnn::Mat::PixelType::PIXEL_BGRA2RGBA);

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

        .def("create", (void (VkMat::*)(int, size_t, VkAllocator*, VkAllocator*)) & VkMat::create,
             py::arg("w") = 1,
             py::arg("elemsize") = 4,
             py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def("create", (void (VkMat::*)(int, int, size_t, VkAllocator*, VkAllocator*)) & VkMat::create,
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4,
             py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def("create", (void (VkMat::*)(int, int, int, size_t, VkAllocator*, VkAllocator*)) & VkMat::create,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4,
             py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)

        .def("create", (void (VkMat::*)(int, size_t, int, VkAllocator*, VkAllocator*)) & VkMat::create,
             py::arg("w") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def("create", (void (VkMat::*)(int, int, size_t, int, VkAllocator*, VkAllocator*)) & VkMat::create,
             py::arg("w") = 1, py::arg("h") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)
        .def("create", (void (VkMat::*)(int, int, int, size_t, int, VkAllocator*, VkAllocator*)) & VkMat::create,
             py::arg("w") = 1, py::arg("h") = 1, py::arg("c") = 1,
             py::arg("elemsize") = 4, py::arg("elempack") = 1,
             py::arg("allocator") = nullptr, py::arg("staging_allocator") = nullptr)

        .def("create_like", (void (VkMat::*)(const Mat&, VkAllocator*, VkAllocator*)) & VkMat::create_like)
        .def("create_like", (void (VkMat::*)(const VkMat&, VkAllocator*, VkAllocator*)) & VkMat::create_like)
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
        .def("channel", (VkMat(VkMat::*)(int)) & VkMat::channel)
        .def("channel", (const VkMat (VkMat::*)(int) const) & VkMat::channel)
        .def("channel_range", (VkMat(VkMat::*)(int, int)) & VkMat::channel_range)
        .def("channel_range", (const VkMat (VkMat::*)(int, int) const) & VkMat::channel_range)
        .def("row_range", (VkMat(VkMat::*)(int, int)) & VkMat::row_range)
        .def("row_range", (const VkMat (VkMat::*)(int, int) const) & VkMat::row_range)
        .def("range", (VkMat(VkMat::*)(int, int)) & VkMat::range)
        .def("range", (const VkMat (VkMat::*)(int, int) const) & VkMat::range)
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
        .def("set_workspace_allocator", &Extractor::set_workspace_allocator)
#if NCNN_VULKAN
        .def("set_vulkan_compute", &Extractor::set_vulkan_compute)
        .def("set_blob_vkallocator", &Extractor::set_blob_vkallocator)
        .def("set_workspace_vkallocator", &Extractor::set_workspace_vkallocator)
        .def("set_staging_vkallocator", &Extractor::set_staging_vkallocator)
#endif // NCNN_VULKAN
#if NCNN_STRING
        .def("input", (int (Extractor::*)(const char*, const Mat&)) & Extractor::input)
        .def("extract", (int (Extractor::*)(const char*, Mat&, int)) & Extractor::extract,
             "get result by blob name", py::arg("blob_name"), py::arg("feat"), py::arg("type") = 0)
#endif
        .def("input", (int (Extractor::*)(int, const Mat&)) & Extractor::input)
        .def("extract", (int (Extractor::*)(int, Mat&, int)) & Extractor::extract,
             "get result by blob name", py::arg("blob_name"), py::arg("feat"), py::arg("type") = 0)
#if NCNN_VULKAN
        .def("input", (int (Extractor::*)(const char*, const VkMat&)) & Extractor::input)
        .def("extract", (int (Extractor::*)(const char*, VkMat&, VkCompute&)) & Extractor::extract)
#if NCNN_STRING
#endif // NCNN_STRING
        .def("input", (int (Extractor::*)(int, const VkMat&)) & Extractor::input)
        .def("extract", (int (Extractor::*)(int, VkMat&, VkCompute&)) & Extractor::extract)
#endif // NCNN_VULKAN
        .def_readwrite("net", &Extractor::net)
        .def_readwrite("blob_mats", &Extractor::blob_mats)
        .def_readwrite("opt", &Extractor::opt);

    py::class_<Layer, PyLayer>(m, "Layer")
        .def(py::init<>())
        .def("load_param", &Layer::load_param)
        .def("load_model", &Layer::load_model)
        .def("create_pipeline", &Layer::create_pipeline)
        .def("destroy_pipeline", &Layer::destroy_pipeline)
        .def_readwrite("one_blob_only", &Layer::one_blob_only)
        .def_readwrite("support_inplace", &Layer::support_inplace)
        .def_readwrite("support_vulkan", &Layer::support_vulkan)
        .def_readwrite("support_packing", &Layer::support_packing)
        .def_readwrite("support_bf16_storage", &Layer::support_bf16_storage)
        .def_readwrite("support_fp16_storage", &Layer::support_fp16_storage)
        .def_readwrite("support_image_storage", &Layer::support_image_storage)
        .def_readwrite("use_int8_inference", &Layer::use_int8_inference)
        .def_readwrite("support_weight_fp16_storage", &Layer::support_weight_fp16_storage)
        .def("forward", (int (Layer::*)(const std::vector<Mat>&, std::vector<Mat>&, const Option&) const) & Layer::forward)
        .def("forward", (int (Layer::*)(const Mat&, Mat&, const Option&) const) & Layer::forward)
        //.def("forward_inplace", ( int( Layer::* )( std::vector<Mat>&, const Option& ) const )&Layer::forward_inplace)
        .def("forward_inplace", (int (Layer::*)(Mat&, const Option&) const) & Layer::forward_inplace)
#if NCNN_VULKAN
        .def("upload_model", &Layer::upload_model)
        .def("forward", (int (Layer::*)(const std::vector<VkMat>&, std::vector<VkMat>&, VkCompute&, const Option&) const) & Layer::forward)
        .def("forward", (int (Layer::*)(const VkMat&, VkMat&, VkCompute& cmd, const Option&) const) & Layer::forward)
        .def("forward_inplace", (int (Layer::*)(std::vector<VkMat>&, VkCompute&, const Option&) const) & Layer::forward_inplace)
        .def("forward_inplace", (int (Layer::*)(VkMat&, VkCompute&, const Option&) const) & Layer::forward_inplace)
#endif // NCNN_VULKAN
        .def_readwrite("typeindex", &Layer::typeindex)
        .def_readwrite("type", &Layer::type)
        .def_readwrite("name", &Layer::name)
        .def_readwrite("bottoms", &Layer::bottoms)
        .def_readwrite("tops", &Layer::tops)
        .def_readwrite("bottom_shapes", &Layer::bottom_shapes)
        .def_readwrite("top_shapes", &Layer::top_shapes);

    py::class_<Net>(m, "Net")
        .def(py::init<>())
        .def_readwrite("opt", &Net::opt)
        .def_readwrite("blobs", &Net::blobs)
        .def_readwrite("layers", &Net::layers)
#if NCNN_VULKAN
        .def("set_vulkan_device", (void (Net::*)(int)) & Net::set_vulkan_device)
        .def("set_vulkan_device", (void (Net::*)(const VulkanDevice*)) & Net::set_vulkan_device)
        .def("vulkan_device", &Net::vulkan_device)
#endif // NCNN_VULKAN
#if NCNN_STRING
    // .def("register_custom_layer", [](Net& net, const char* type, const std::function< ::Layer*()>& creator) {
    //     g_layer_creator = creator;
    //     return net.register_custom_layer(type, LayerCreator);
    // })
#endif //NCNN_STRING
        .def("register_custom_layer", (int (Net::*)(int, layer_creator_func)) & Net::register_custom_layer)
#if NCNN_STRING
        .def("load_param", (int (Net::*)(const DataReader&)) & Net::load_param)
#endif // NCNN_STRING
        .def("load_param_bin", (int (Net::*)(const DataReader&)) & Net::load_param_bin)
        .def("load_model", (int (Net::*)(const DataReader&)) & Net::load_model)

#if NCNN_STDIO
#if NCNN_STRING
        .def("load_param", (int (Net::*)(const char*)) & Net::load_param)
        .def("load_param_mem", (int (Net::*)(const char*)) & Net::load_param_mem)
#endif // NCNN_STRING
        .def("load_param_bin", (int (Net::*)(const char*)) & Net::load_param_bin)
        .def("load_model", (int (Net::*)(const char*)) & Net::load_model)
#endif // NCNN_STDIO

        //todo load from memory
        //.def("load_param", (int (Net::*)(const unsigned char*))(&Net::load_param))
        //.def("load_model", (int (Net::*)(const unsigned char*))(&Net::load_model))

        .def("clear", &Net::clear)
        .def("create_extractor", &Net::create_extractor);

    py::enum_<ncnn::BorderType>(m, "BorderType")
        .value("BORDER_CONSTANT", ncnn::BorderType::BORDER_CONSTANT)
        .value("BORDER_REPLICATE", ncnn::BorderType::BORDER_REPLICATE);

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
    m.def("build_with_gpu", []() {
#if NCNN_VULKAN
        return true;
#else
            return false;
#endif
    });
#if NCNN_PIXEL
    m.def("yuv420sp2rgb", [](py::buffer const yuv420sp, int w, int h, py::buffer rgb) {
        return yuv420sp2rgb((unsigned char*)yuv420sp.request().ptr, w, h, (unsigned char*)rgb.request().ptr);
    });
    m.def("resize_bilinear_c1", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_c1((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
    m.def("resize_bilinear_c2", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_c2((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
    m.def("resize_bilinear_c3", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_c3((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
    m.def("resize_bilinear_c4", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_c4((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
    m.def("resize_bilinear_c1", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride) {
        return resize_bilinear_c1((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride);
    });
    m.def("resize_bilinear_c2", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride) {
        return resize_bilinear_c2((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride);
    });
    m.def("resize_bilinear_c3", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride) {
        return resize_bilinear_c3((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride);
    });
    m.def("resize_bilinear_c4", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride) {
        return resize_bilinear_c4((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride);
    });
    m.def("resize_bilinear_yuv420sp", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h) {
        return resize_bilinear_yuv420sp((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h);
    });
#endif // NCNN_PIXEL
#if NCNN_PIXEL_ROTATE
    m.def("kanna_rotate_c1", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_c1((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
    m.def("kanna_rotate_c2", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_c2((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
    m.def("kanna_rotate_c3", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_c3((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
    m.def("kanna_rotate_c4", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_c4((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
    m.def("kanna_rotate_c1", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride, int type) {
        return kanna_rotate_c1((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride, type);
    });
    m.def("kanna_rotate_c2", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride, int type) {
        return kanna_rotate_c2((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride, type);
    });
    m.def("kanna_rotate_c3", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride, int type) {
        return kanna_rotate_c3((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride, type);
    });
    m.def("kanna_rotate_c4", [](py::buffer const src, int srcw, int srch, int srcstride, py::buffer dst, int w, int h, int stride, int type) {
        return kanna_rotate_c4((unsigned char*)src.request().ptr, srcw, srch, srcstride, (unsigned char*)dst.request().ptr, w, h, stride, type);
    });
    m.def("kanna_rotate_yuv420sp", [](py::buffer const src, int srcw, int srch, py::buffer dst, int w, int h, int type) {
        return kanna_rotate_yuv420sp((unsigned char*)src.request().ptr, srcw, srch, (unsigned char*)dst.request().ptr, w, h, type);
    });
#endif // NCNN_PIXEL_ROTATE
    m.def("copy_make_border", [](const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v) {
        return copy_make_border(src, dst, top, bottom, left, right, type, v);
    });
    m.def("copy_make_border", [](const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt) {
        return copy_make_border(src, dst, top, bottom, left, right, type, v, opt);
    });
    m.def("copy_cut_border", [](const Mat& src, Mat& dst, int top, int bottom, int left, int right) {
        return copy_cut_border(src, dst, top, bottom, left, right);
    });
    m.def("copy_cut_border", [](const Mat& src, Mat& dst, int top, int bottom, int left, int right, const Option& opt) {
        return copy_cut_border(src, dst, top, bottom, left, right, opt);
    });
    m.def("resize_bilinear", [](const Mat& src, Mat& dst, int w, int h) {
        return resize_bilinear(src, dst, w, h);
    });
    m.def("resize_bilinear", [](const Mat& src, Mat& dst, int w, int h, const Option& opt) {
        return resize_bilinear(src, dst, w, h, opt);
    });
    m.def("resize_bicubic", [](const Mat& src, Mat& dst, int w, int h) {
        return resize_bicubic(src, dst, w, h);
    });
    m.def("resize_bicubic", [](const Mat& src, Mat& dst, int w, int h, const Option& opt) {
        return resize_bicubic(src, dst, w, h, opt);
    });
    m.def("convert_packing", [](const Mat& src, Mat& dst, int elempack) {
        return convert_packing(src, dst, elempack);
    });
    m.def("convert_packing", [](const Mat& src, Mat& dst, int elempack, const Option& opt) {
        return convert_packing(src, dst, elempack, opt);
    });
    m.def("cast_float32_to_float16", [](const Mat& src, Mat& dst) {
        return cast_float32_to_float16(src, dst);
    });
    m.def("cast_float32_to_float16", [](const Mat& src, Mat& dst, const Option& opt) {
        return cast_float32_to_float16(src, dst, opt);
    });
    m.def("cast_float16_to_float32", [](const Mat& src, Mat& dst) {
        return cast_float16_to_float32(src, dst);
    });
    m.def("cast_float16_to_float32", [](const Mat& src, Mat& dst, const Option& opt) {
        return cast_float16_to_float32(src, dst, opt);
    });
    m.def("cast_int8_to_float32", [](const Mat& src, Mat& dst) {
        return cast_int8_to_float32(src, dst);
    });
    m.def("cast_int8_to_float32", [](const Mat& src, Mat& dst, const Option& opt) {
        return cast_int8_to_float32(src, dst, opt);
    });
    m.def("quantize_float32_to_int8", [](const Mat& src, Mat& dst, float scale) {
        return quantize_float32_to_int8(src, dst, scale);
    });
    m.def("quantize_float32_to_int8", [](const Mat& src, Mat& dst, float scale, const Option& opt) {
        return quantize_float32_to_int8(src, dst, scale, opt);
    });
    m.def("dequantize_int32_to_float32", [](Mat& m, float scale, py::buffer bias, int bias_data_size) {
        return dequantize_int32_to_float32(m, scale, (float*)bias.request().ptr, bias_data_size);
    });
    m.def("dequantize_int32_to_float32", [](Mat& m, float scale, py::buffer bias, int bias_data_size, const Option& opt) {
        return dequantize_int32_to_float32(m, scale, (float*)bias.request().ptr, bias_data_size, opt);
    });
    m.def("requantize_int8_to_int8", [](const Mat& src, Mat& dst, float scale_in, float scale_out, py::buffer bias, int bias_data_size, int fusion_relu) {
        return requantize_int8_to_int8(src, dst, scale_in, scale_out, (float*)bias.request().ptr, bias_data_size, fusion_relu);
    });
    m.def("requantize_int8_to_int8", [](const Mat& src, Mat& dst, float scale_in, float scale_out, py::buffer bias, int bias_data_size, int fusion_relu, const Option& opt) {
        return requantize_int8_to_int8(src, dst, scale_in, scale_out, (float*)bias.request().ptr, bias_data_size, fusion_relu, opt);
    });
#if NCNN_STRING
    m.def("layer_to_index", &layer_to_index);
    m.def("create_layer", [](const char* type) {
        return static_cast<Layer*>(create_layer(type));
    });
    m.def("create_layer", [](int index) {
        return static_cast<Layer*>(create_layer(index));
    });
#endif //NCNN_STRING

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

    py::class_<VkAllocator, PyVkAllocator<> >(m, "VkAllocator")
        .def_readwrite("vkdev", &VkAllocator::vkdev)
        .def_readwrite("memory_type_index", &VkAllocator::memory_type_index)
        .def_readwrite("mappable", &VkAllocator::mappable)
        .def_readwrite("coherent", &VkAllocator::coherent);
    py::class_<VkBlobBufferAllocator, VkAllocator, PyVkAllocatorOther<VkBlobBufferAllocator> >(m, "VkBlobBufferAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("clear", &VkBlobBufferAllocator::clear)
        .def("fastMalloc", &VkBlobBufferAllocator::fastMalloc)
        .def("fastFree", &VkBlobBufferAllocator::fastFree);
    py::class_<VkWeightBufferAllocator, VkAllocator, PyVkAllocatorOther<VkWeightBufferAllocator> >(m, "VkWeightBufferAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("clear", &VkWeightBufferAllocator::clear)
        .def("fastMalloc", &VkWeightBufferAllocator::fastMalloc)
        .def("fastFree", &VkWeightBufferAllocator::fastFree);
    py::class_<VkStagingBufferAllocator, VkAllocator, PyVkAllocatorOther<VkStagingBufferAllocator> >(m, "VkStagingBufferAllocator")
        .def(py::init<const VulkanDevice*>())
        .def("set_size_compare_ratio", &VkStagingBufferAllocator::set_size_compare_ratio)
        .def("clear", &VkStagingBufferAllocator::clear)
        .def("fastMalloc", &VkStagingBufferAllocator::fastMalloc)
        .def("fastFree", &VkStagingBufferAllocator::fastFree);
    py::class_<VkWeightStagingBufferAllocator, VkAllocator, PyVkAllocatorOther<VkWeightStagingBufferAllocator> >(m, "VkWeightStagingBufferAllocator")
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

    py::class_<VkImageAllocator, VkAllocator, PyVkImageAllocator<> >(m, "VkImageAllocator")
        .def("clear", &VkImageAllocator::clear);
    py::class_<VkSimpleImageAllocator, VkImageAllocator, PyVkImageAllocatorOther<VkSimpleImageAllocator> >(m, "VkSimpleImageAllocator")
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

    //  ========================== Layers ======================================

#define LAYER_COMMON                                           \
    .def_readwrite("typeindex", &Layer::typeindex)             \
        .def_readwrite("type", &Layer::type)                   \
        .def_readwrite("name", &Layer::name)                   \
        .def_readwrite("bottoms", &Layer::bottoms)             \
        .def_readwrite("tops", &Layer::tops)                   \
        .def_readwrite("tops", &Layer::tops)                   \
        .def_readwrite("bottom_shapes", &Layer::bottom_shapes) \
        .def_readwrite("top_shapes", &Layer::top_shapes)

    py::class_<AbsVal_final, Layer>(m, "AbsVal")
        .def(py::init<>())

            LAYER_COMMON;

    //    layer: ArgMax todo

    py::class_<BatchNorm_final, Layer>(m, "BatchNorm")
        .def(py::init<>())
        .def_readwrite("channels", &BatchNorm::channels)
        .def_readwrite("eps", &BatchNorm::eps)
        .def_readwrite("slope_data", &BatchNorm::slope_data)
        .def_readwrite("mean_data", &BatchNorm::mean_data)
        .def_readwrite("var_data", &BatchNorm::var_data)
        .def_readwrite("bias_data", &BatchNorm::bias_data)

            LAYER_COMMON;

    py::class_<Bias_final, Layer>(m, "Bias")
        .def(py::init<>())
        .def_readwrite("bias_data_size", &Bias::bias_data_size)

            LAYER_COMMON;

    py::class_<BinaryOp_final, Layer> binaryOp(m, "BinaryOp");
    binaryOp.def(py::init<>())
        .def_readwrite("op_type", &BinaryOp::op_type)
        .def_readwrite("with_scalar", &BinaryOp::with_scalar)
        .def_readwrite("b", &BinaryOp::b)

            LAYER_COMMON;

    py::enum_<BinaryOp::OperationType>(binaryOp, "OperationType")
        .value("Operation_ADD", BinaryOp::OperationType::Operation_ADD)
        .value("Operation_SUB", BinaryOp::OperationType::Operation_SUB)
        .value("Operation_MUL", BinaryOp::OperationType::Operation_MUL)
        .value("Operation_DIV", BinaryOp::OperationType::Operation_DIV)
        .value("Operation_MAX", BinaryOp::OperationType::Operation_MAX)
        .value("Operation_MIN", BinaryOp::OperationType::Operation_MIN)
        .value("Operation_POW", BinaryOp::OperationType::Operation_POW)
        .value("Operation_RSUB", BinaryOp::OperationType::Operation_RSUB)
        .value("Operation_RDIV", BinaryOp::OperationType::Operation_RDIV);

    py::class_<BNLL_final, Layer>(m, "BNLL").def(py::init<>())

        LAYER_COMMON;

    //    Cast
    //    Clip

    py::class_<Concat_final, Layer>(m, "Concat")
        .def(py::init<>())
        .def_readwrite("axis", &Concat::axis)

            LAYER_COMMON;

    py::class_<Convolution_final, Layer>(m, "Convolution")
        .def(py::init<>())
        .def_readwrite("num_output", &Convolution::num_output)
        .def_readwrite("kernel_w", &Convolution::kernel_w)
        .def_readwrite("kernel_h", &Convolution::kernel_h)
        .def_readwrite("dilation_w", &Convolution::dilation_w)
        .def_readwrite("dilation_h", &Convolution::dilation_h)
        .def_readwrite("stride_w", &Convolution::stride_w)
        .def_readwrite("stride_h", &Convolution::stride_h)
        // -233=SAME_UPPER -234=SAME_LOWER
        .def_readwrite("pad_left", &Convolution::pad_left)
        .def_readwrite("pad_right", &Convolution::pad_right)
        .def_readwrite("pad_top", &Convolution::pad_top)
        .def_readwrite("pad_bottom", &Convolution::pad_bottom)
        .def_readwrite("pad_value", &Convolution::pad_value)
        .def_readwrite("bias_term", &Convolution::bias_term)

        .def_readwrite("weight_data_size", &Convolution::weight_data_size)
        .def_readwrite("int8_scale_term", &Convolution::int8_scale_term)
        // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
        .def_readwrite("activation_type", &Convolution::activation_type)
        .def_readwrite("activation_params", &Convolution::activation_params)

        .def_readwrite("bias_data", &Convolution::bias_data)
        .def_readwrite("weight_data", &Convolution::weight_data)

        .def_readwrite("weight_data_int8_scales", &Convolution::weight_data_int8_scales)
        .def_readwrite("bottom_blob_int8_scale", &Convolution::bottom_blob_int8_scale)
        .def_readwrite("top_blob_int8_scale", &Convolution::top_blob_int8_scale) // TODO load param
        .def_readwrite("use_int8_requantize", &Convolution::use_int8_requantize)

        // implementation type, 0 means do not use auto pack model
        .def_readwrite("impl_type", &Convolution::impl_type)

            LAYER_COMMON;

    py::class_<ConvolutionDepthWise_final, Layer>(m, "ConvolutionDepthWise")
        .def(py::init<>())
        .def_readwrite("num_output", &ConvolutionDepthWise::num_output)
        .def_readwrite("kernel_w", &ConvolutionDepthWise::kernel_w)
        .def_readwrite("kernel_h", &ConvolutionDepthWise::kernel_h)
        .def_readwrite("dilation_w", &ConvolutionDepthWise::dilation_w)
        .def_readwrite("dilation_h", &ConvolutionDepthWise::dilation_h)
        .def_readwrite("stride_w", &ConvolutionDepthWise::stride_w)
        .def_readwrite("stride_h", &ConvolutionDepthWise::stride_h)
        // -233=SAME_UPPER -234=SAME_LOWER
        .def_readwrite("pad_left", &ConvolutionDepthWise::pad_left)
        .def_readwrite("pad_right", &ConvolutionDepthWise::pad_right)
        .def_readwrite("pad_top", &ConvolutionDepthWise::pad_top)
        .def_readwrite("pad_bottom", &ConvolutionDepthWise::pad_bottom)
        .def_readwrite("pad_value", &ConvolutionDepthWise::pad_value)
        .def_readwrite("bias_term", &ConvolutionDepthWise::bias_term)
        .def_readwrite("weight_data_size", &ConvolutionDepthWise::weight_data_size)

        .def_readwrite("group", &ConvolutionDepthWise::group)

        .def_readwrite("int8_scale_term", &ConvolutionDepthWise::int8_scale_term)
        // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
        .def_readwrite("activation_type", &ConvolutionDepthWise::activation_type)
        .def_readwrite("activation_params", &ConvolutionDepthWise::activation_params)

        .def_readwrite("bias_data", &ConvolutionDepthWise::bias_data)
        .def_readwrite("weight_data", &ConvolutionDepthWise::weight_data)

        .def_readwrite("weight_data_int8_scales", &ConvolutionDepthWise::weight_data_int8_scales)
        .def_readwrite("bottom_blob_int8_scale", &ConvolutionDepthWise::bottom_blob_int8_scales)
        .def_readwrite("top_blob_int8_scale", &ConvolutionDepthWise::top_blob_int8_scale) // TODO load param
        .def_readwrite("use_int8_requantize", &ConvolutionDepthWise::use_int8_requantize)

            LAYER_COMMON;

    py::class_<Crop_final, Layer>(m, "Crop")
        .def(py::init<>())
        // -233 = dynamic offset from reference blob
        .def_readwrite("woffset", &Crop::woffset)
        .def_readwrite("hoffset", &Crop::hoffset)
        .def_readwrite("coffset", &Crop::coffset)

        // -233 = remaining
        .def_readwrite("outw", &Crop::outw)
        .def_readwrite("outh", &Crop::outh)
        .def_readwrite("outc", &Crop::outc)

        // tail offset for cropping, ignored if reference blob is provided
        // woffset is aka left, and woffset2 is aka right
        .def_readwrite("woffset2", &Crop::woffset2)
        .def_readwrite("hoffset2", &Crop::hoffset2)
        .def_readwrite("coffset2", &Crop::coffset2)

        // numpy-style slice
        // if provided, all the above attributes will be ignored
        .def_readwrite("starts", &Crop::starts)
        .def_readwrite("ends", &Crop::ends)
        .def_readwrite("axes", &Crop::axes)

            LAYER_COMMON;

    py::class_<Deconvolution_final, Layer>(m, "Deconvolution")
        .def(py::init<>())
        .def_readwrite("num_output", &Deconvolution::num_output)
        .def_readwrite("kernel_w", &Deconvolution::kernel_w)
        .def_readwrite("kernel_h", &Deconvolution::kernel_h)
        .def_readwrite("dilation_w", &Deconvolution::dilation_w)
        .def_readwrite("dilation_h", &Deconvolution::dilation_h)
        .def_readwrite("stride_w", &Deconvolution::stride_w)
        .def_readwrite("stride_h", &Deconvolution::stride_h)
        .def_readwrite("pad_left", &Deconvolution::pad_left)
        .def_readwrite("pad_right", &Deconvolution::pad_right)
        .def_readwrite("pad_top", &Deconvolution::pad_top)
        .def_readwrite("pad_bottom", &Deconvolution::pad_bottom)
        .def_readwrite("output_pad_right", &Deconvolution::output_pad_right)
        .def_readwrite("output_pad_bottom", &Deconvolution::output_pad_bottom)
        .def_readwrite("output_w", &Deconvolution::output_w)
        .def_readwrite("output_h", &Deconvolution::output_h)
        .def_readwrite("bias_term", &Deconvolution::bias_term)
        .def_readwrite("weight_data_size", &Deconvolution::weight_data_size)

        // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
        .def_readwrite("activation_type", &Deconvolution::activation_type)
        .def_readwrite("activation_params", &Deconvolution::activation_params)

        // model
        .def_readwrite("weight_data", &Deconvolution::weight_data)
        .def_readwrite("bias_data", &Deconvolution::bias_data)

            LAYER_COMMON;

    py::class_<DeconvolutionDepthWise_final, Layer>(m, "DeconvolutionDepthWise")
        .def(py::init<>())
        .def_readwrite("num_output", &DeconvolutionDepthWise::num_output)
        .def_readwrite("kernel_w", &DeconvolutionDepthWise::kernel_w)
        .def_readwrite("kernel_h", &DeconvolutionDepthWise::kernel_h)
        .def_readwrite("dilation_w", &DeconvolutionDepthWise::dilation_w)
        .def_readwrite("dilation_h", &DeconvolutionDepthWise::dilation_h)
        .def_readwrite("stride_w", &DeconvolutionDepthWise::stride_w)
        .def_readwrite("stride_h", &DeconvolutionDepthWise::stride_h)
        .def_readwrite("pad_left", &DeconvolutionDepthWise::pad_left)
        .def_readwrite("pad_right", &DeconvolutionDepthWise::pad_right)
        .def_readwrite("pad_top", &DeconvolutionDepthWise::pad_top)
        .def_readwrite("pad_bottom", &DeconvolutionDepthWise::pad_bottom)
        .def_readwrite("output_pad_right", &DeconvolutionDepthWise::output_pad_right)
        .def_readwrite("output_pad_bottom", &DeconvolutionDepthWise::output_pad_bottom)
        .def_readwrite("output_w", &DeconvolutionDepthWise::output_w)
        .def_readwrite("output_h", &DeconvolutionDepthWise::output_h)
        .def_readwrite("bias_term", &DeconvolutionDepthWise::bias_term)
        .def_readwrite("weight_data_size", &DeconvolutionDepthWise::weight_data_size)
        .def_readwrite("group", &DeconvolutionDepthWise::group)

        // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
        .def_readwrite("activation_type", &DeconvolutionDepthWise::activation_type)
        .def_readwrite("activation_params", &DeconvolutionDepthWise::activation_params)

        // model
        .def_readwrite("weight_data", &DeconvolutionDepthWise::weight_data)
        .def_readwrite("bias_data", &DeconvolutionDepthWise::bias_data)

            LAYER_COMMON;

    py::class_<Dequantize_final, Layer>(m, "Dequantize")
        .def(py::init<>())
        .def_readwrite("scale", &Dequantize::scale)
        .def_readwrite("bias_term", &Dequantize::bias_term)
        .def_readwrite("bias_data_size", &Dequantize::bias_data_size)
        .def_readwrite("bias_data", &Dequantize::bias_data)

            LAYER_COMMON;

    py::class_<DetectionOutput_final, Layer>(m, "DetectionOutput", py::buffer_protocol())
        .def(py::init<>())
        .def_readwrite("num_class", &DetectionOutput::num_class)
        .def_readwrite("nms_threshold", &DetectionOutput::nms_threshold)
        .def_readwrite("nms_top_k", &DetectionOutput::nms_top_k)
        .def_readwrite("keep_top_k", &DetectionOutput::keep_top_k)
        .def_readwrite("confidence_threshold", &DetectionOutput::confidence_threshold)

        .def_property(
            "variances",
            [](DetectionOutput_final& p) -> pybind11::array {
                auto dtype = pybind11::dtype(pybind11::format_descriptor<float>::format());
                return pybind11::array(dtype, {4}, {sizeof(float)}, p.variances, nullptr);
            },
            [](DetectionOutput_final& p) {})

        .def_readwrite("num_class", &DetectionOutput::num_class)

            LAYER_COMMON;

    py::class_<Dropout_final, Layer>(m, "Dropout")
        .def(py::init<>())
        .def_readwrite("scale", &Dropout::scale)

            LAYER_COMMON;

    py::class_<Eltwise_final, Layer> eltwise_final(m, "Eltwise");
    eltwise_final.def(py::init<>())
        .def_readwrite("op_type", &Eltwise::op_type)
        .def_readwrite("coeffs", &Eltwise::coeffs)

            LAYER_COMMON;

    py::enum_<Eltwise_final::OperationType>(eltwise_final, "OperationType")
        .value("Operation_PROD", Eltwise::OperationType::Operation_PROD)
        .value("Operation_SUM", Eltwise::OperationType::Operation_PROD)
        .value("Operation_MAX", Eltwise::OperationType::Operation_PROD);

    py::class_<ELU_final, Layer>(m, "ELU")
        .def(py::init<>())
        .def_readwrite("alpha", &ELU::alpha)

            LAYER_COMMON;

    py::class_<Embed_final, Layer>(m, "Embed")
        .def(py::init<>())
        .def_readwrite("num_output", &Embed::num_output)
        .def_readwrite("input_dim", &Embed::input_dim)
        .def_readwrite("bias_term", &Embed::bias_term)
        .def_readwrite("weight_data_size", &Embed::weight_data_size)

        .def_readwrite("weight_data", &Embed::weight_data)
        .def_readwrite("bias_data", &Embed::bias_data)

            LAYER_COMMON;

    py::class_<Exp_final, Layer>(m, "Exp")
        .def(py::init<>())
        .def_readwrite("base", &Exp::base)
        .def_readwrite("scale", &Exp::scale)
        .def_readwrite("shift", &Exp::shift)

            LAYER_COMMON;

    py::class_<ExpandDims_final, Layer>(m, "ExpandDims")
        .def(py::init<>())
        .def_readwrite("expand_w", &ExpandDims::expand_w)
        .def_readwrite("expand_h", &ExpandDims::expand_h)
        .def_readwrite("expand_c", &ExpandDims::expand_c)
        .def_readwrite("axes", &ExpandDims::axes)

            LAYER_COMMON;

    py::class_<Flatten_final, Layer>(m, "Flatten")
        .def(py::init<>())

            LAYER_COMMON;

    py::class_<HardSigmoid_final, Layer>(m, "HardSigmoid")
        .def(py::init<>())
        .def_readwrite("alpha", &HardSigmoid::alpha)
        .def_readwrite("beta", &HardSigmoid::beta)
        .def_readwrite("lower", &HardSigmoid::lower)
        .def_readwrite("upper", &HardSigmoid::upper)

            LAYER_COMMON;

    py::class_<HardSwish_final, Layer>(m, "HardSwish")
        .def(py::init<>())
        .def_readwrite("alpha", &HardSwish::alpha)
        .def_readwrite("beta", &HardSwish::beta)
        .def_readwrite("lower", &HardSwish::lower)
        .def_readwrite("upper", &HardSwish::upper)

            LAYER_COMMON;

    py::class_<InnerProduct_final, Layer>(m, "InnerProduct")
        .def(py::init<>())
        .def_readwrite("num_output", &InnerProduct::num_output)
        .def_readwrite("bias_term", &InnerProduct::bias_term)
        .def_readwrite("weight_data_size", &InnerProduct::weight_data_size)
        .def_readwrite("int8_scale_term", &InnerProduct::int8_scale_term)

        // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
        .def_readwrite("activation_type", &InnerProduct::activation_type)
        .def_readwrite("activation_params", &InnerProduct::activation_params)

        .def_readwrite("weight_data", &InnerProduct::weight_data)
        .def_readwrite("bias_data", &InnerProduct::bias_data)

        .def_readwrite("weight_data_int8_scales", &InnerProduct::weight_data_int8_scales)
        .def_readwrite("bottom_blob_int8_scale", &InnerProduct::bottom_blob_int8_scale)

            LAYER_COMMON;

    py::class_<Input_final, Layer>(m, "Input")
        .def(py::init<>())
        .def_readwrite("w", &Input::w)
        .def_readwrite("h", &Input::h)
        .def_readwrite("c", &Input::c)

            LAYER_COMMON;

    py::class_<InstanceNorm_final, Layer>(m, "InstanceNorm")
        .def(py::init<>())
        .def_readwrite("channels", &InstanceNorm::channels)
        .def_readwrite("eps", &InstanceNorm::eps)
        .def_readwrite("gamma_data", &InstanceNorm::gamma_data)
        .def_readwrite("beta_data", &InstanceNorm::beta_data)

            LAYER_COMMON;
    py::class_<Interp_final, Layer>(m, "Interp")
        .def(py::init<>())
        // 1=nearest  2=bilinear  3=bicubic
        .def_readwrite("resize_type", &Interp::resize_type)
        .def_readwrite("width_scale", &Interp::width_scale)
        .def_readwrite("height_scale", &Interp::height_scale)
        .def_readwrite("output_width", &Interp::output_width)
        .def_readwrite("output_height", &Interp::output_height)

            LAYER_COMMON;

    py::class_<Log_final, Layer>(m, "Log")
        .def(py::init<>())
        .def_readwrite("base", &Log::base)
        .def_readwrite("scale", &Log::scale)
        .def_readwrite("shift", &Log::shift)

            LAYER_COMMON;

    py::class_<LRN_final, Layer>(m, "LRN")
        .def(py::init<>())
        .def_readwrite("region_type", &LRN::region_type)
        .def_readwrite("local_size", &LRN::local_size)
        .def_readwrite("alpha", &LRN::alpha)
        .def_readwrite("beta", &LRN::beta)
        .def_readwrite("bias", &LRN::bias)

            LAYER_COMMON;

    py::class_<MemoryData_final, Layer>(m, "MemoryData")
        .def(py::init<>())
        .def_readwrite("w", &MemoryData::w)
        .def_readwrite("h", &MemoryData::h)
        .def_readwrite("c", &MemoryData::c)
        .def_readwrite("data", &MemoryData::data)

            LAYER_COMMON;

    py::class_<MVN_final, Layer>(m, "MVN")
        .def(py::init<>())
        .def_readwrite("normalize_variance", &MVN::normalize_variance)
        .def_readwrite("across_channels", &MVN::across_channels)
        .def_readwrite("eps", &MVN::eps)

            LAYER_COMMON;

    py::class_<Noop_final, Layer>(m, "Noop").def(py::init<>())

        LAYER_COMMON;

    py::class_<Normalize, Layer>(m, "Normalize")
        .def(py::init<>())
        .def_readwrite("across_spatial", &Normalize::across_spatial)
        .def_readwrite("across_channel", &Normalize::across_channel)
        .def_readwrite("channel_shared", &Normalize::channel_shared)
        .def_readwrite("eps", &Normalize::eps)
        .def_readwrite("scale_data_size", &Normalize::scale_data_size)

        // 0 = v / sqrt(v2 + eps) caffe/mxnet
        // 1 = v / max(sqrt(v2), eps) pytorch
        // 2 = v / sqrt(max(v2, eps)) tensorflow
        .def_readwrite("eps_mode", &Normalize::eps_mode)

        .def_readwrite("scale_data", &Normalize::scale_data)

            LAYER_COMMON;

    py::class_<Packing_final, Layer>(m, "Packing")
        .def(py::init<>())
        .def_readwrite("out_elempack", &Packing::out_elempack)
        .def_readwrite("use_padding", &Packing::use_padding)

            LAYER_COMMON;

    py::class_<Padding_final, Layer>(m, "Padding")
        .def(py::init<>())
        // -233 = dynamic offset from reference blob
        .def_readwrite("top", &Padding::top)
        .def_readwrite("bottom", &Padding::bottom)
        .def_readwrite("left", &Padding::left)
        .def_readwrite("right", &Padding::right)
        // 0=CONSTANT 1=REPLICATE 2=REFLECT

        .def_readwrite("type", &Padding::type)
        .def_readwrite("value", &Padding::value)

        // per channel pad value
        .def_readwrite("per_channel_pad_data_size", &Padding::per_channel_pad_data_size)
        .def_readwrite("per_channel_pad_data", &Padding::per_channel_pad_data)

            LAYER_COMMON;

    py::class_<Permute_final, Layer>(m, "Permute")
        .def(py::init<>())
        .def_readwrite("order_type", &Permute::order_type)

            LAYER_COMMON;

    py::class_<Pooling_final, Layer> pooling_final(m, "Pooling");
    pooling_final.def(py::init<>())
        .def_readwrite("pooling_type", &Pooling::pooling_type)
        .def_readwrite("kernel_w", &Pooling::kernel_w)
        .def_readwrite("kernel_h", &Pooling::kernel_h)
        .def_readwrite("stride_w", &Pooling::stride_w)
        .def_readwrite("stride_h", &Pooling::stride_h)
        .def_readwrite("pad_left", &Pooling::pad_left)
        .def_readwrite("pad_right", &Pooling::pad_right)
        .def_readwrite("pad_top", &Pooling::pad_top)
        .def_readwrite("pad_bottom", &Pooling::pad_bottom)
        .def_readwrite("global_pooling", &Pooling::global_pooling)
        // 0=full 1=valid 2=SAME_UPPER 3=SAME_LOWER
        .def_readwrite("pad_mode", &Pooling::pad_mode)
        .def_readwrite("avgpool_count_include_pad", &Pooling::avgpool_count_include_pad)

            LAYER_COMMON;

    py::enum_<Pooling_final::PoolMethod>(pooling_final, "PoolMethod")
        .value("PoolMethod_MAX", Pooling::PoolMethod::PoolMethod_MAX)
        .value("PoolMethod_AVE", Pooling::PoolMethod::PoolMethod_AVE)
        .export_values();

    py::class_<Power_final, Layer>(m, "Power")
        .def(py::init<>())
        .def_readwrite("power", &Power::power)
        .def_readwrite("scale", &Power::scale)
        .def_readwrite("shift", &Power::shift)

            LAYER_COMMON;

    py::class_<PReLU_final, Layer>(m, "PReLU")
        .def(py::init<>())
        .def_readwrite("num_slope", &PReLU::num_slope)
        .def_readwrite("slope_data", &PReLU::slope_data)

            LAYER_COMMON;

    py::class_<PriorBox_final, Layer>(m, "PriorBox")
        .def(py::init<>())
        .def_readwrite("min_sizes", &PriorBox::min_sizes)
        .def_readwrite("max_sizes", &PriorBox::max_sizes)
        .def_readwrite("aspect_ratios", &PriorBox::aspect_ratios)

        .def_property(
            "variances",
            [](PriorBox_final& p) -> pybind11::array {
                auto dtype = pybind11::dtype(pybind11::format_descriptor<float>::format());
                return pybind11::array(dtype, {4}, {sizeof(float)}, p.variances, nullptr);
            },
            [](PriorBox_final& p) {})

        .def_readwrite("flip", &PriorBox::flip)
        .def_readwrite("clip", &PriorBox::clip)
        .def_readwrite("image_width", &PriorBox::image_width)
        .def_readwrite("image_height", &PriorBox::image_height)
        .def_readwrite("step_width", &PriorBox::step_width)
        .def_readwrite("step_height", &PriorBox::step_height)
        .def_readwrite("offset", &PriorBox::offset)
        .def_readwrite("step_mmdetection", &PriorBox::step_mmdetection)
        .def_readwrite("center_mmdetection", &PriorBox::center_mmdetection)

            LAYER_COMMON;

    py::class_<Proposal_final, Layer>(m, "Proposal")
        .def(py::init<>())
        .def_readwrite("feat_stride", &Proposal::feat_stride)
        .def_readwrite("base_size", &Proposal::base_size)
        .def_readwrite("pre_nms_topN", &Proposal::pre_nms_topN)
        .def_readwrite("after_nms_topN", &Proposal::after_nms_topN)
        .def_readwrite("nms_thresh", &Proposal::nms_thresh)
        .def_readwrite("min_size", &Proposal::min_size)
        .def_readwrite("ratios", &Proposal::ratios)
        .def_readwrite("scales", &Proposal::scales)
        .def_readwrite("anchors", &Proposal::anchors)

            LAYER_COMMON;

    py::class_<PSROIPooling_final, Layer>(m, "PSROIPooling")
        .def(py::init<>())
        .def_readwrite("pooled_height", &PSROIPooling::pooled_height)
        .def_readwrite("spatial_scale", &PSROIPooling::spatial_scale)
        .def_readwrite("output_dim", &PSROIPooling::output_dim)
        .def_readwrite("pooled_width", &PSROIPooling::pooled_width)

            LAYER_COMMON;

    py::class_<Quantize_final, Layer>(m, "Quantize")
        .def(py::init<>())
        .def_readwrite("scale", &Quantize::scale)

            LAYER_COMMON;

    py::class_<Reduction_final, Layer> reduction_final(m, "Reduction");
    reduction_final.def(py::init<>())
        .def_readwrite("operation", &Reduction::operation)
        .def_readwrite("reduce_all", &Reduction::reduce_all)
        .def_readwrite("coeff", &Reduction::coeff)
        .def_readwrite("axes", &Reduction::axes)
        .def_readwrite("keepdims", &Reduction::keepdims)

            LAYER_COMMON;

    py::enum_<Reduction_final::ReductionOp>(reduction_final, "ReductionOp")
        .value("ReductionOp_SUM", Reduction::ReductionOp::ReductionOp_SUM)
        .value("ReductionOp_ASUM", Reduction::ReductionOp::ReductionOp_ASUM)
        .value("ReductionOp_SUMSQ", Reduction::ReductionOp::ReductionOp_SUMSQ)
        .value("ReductionOp_MEAN", Reduction::ReductionOp::ReductionOp_MEAN)
        .value("ReductionOp_MAX", Reduction::ReductionOp::ReductionOp_MAX)
        .value("ReductionOp_MIN", Reduction::ReductionOp::ReductionOp_MIN)
        .value("ReductionOp_PROD", Reduction::ReductionOp::ReductionOp_PROD)
        .value("ReductionOp_L1", Reduction::ReductionOp::ReductionOp_L1)
        .value("ReductionOp_L2", Reduction::ReductionOp::ReductionOp_L2)
        .value("ReductionOp_LogSum", Reduction::ReductionOp::ReductionOp_LogSum)
        .value("ReductionOp_LogSumExp", Reduction::ReductionOp::ReductionOp_LogSumExp);

    py::class_<ReLU_final, Layer>(m, "ReLU")
        .def(py::init<>())
        .def_readwrite("slope", &ReLU::slope)

            LAYER_COMMON;

    py::class_<Reorg_final, Layer>(m, "Reorg")
        .def(py::init<>())
        .def_readwrite("stride", &Reorg::stride)

            LAYER_COMMON;

    py::class_<Requantize_final, Layer>(m, "Requantize")
        .def(py::init<>())
        // bottom_blob_scale * weight_scale
        .def_readwrite("scale_in", &Requantize::scale_in)
        // top_blob_scale / (bottom_blob_scale * weight_scale)
        .def_readwrite("scale_out", &Requantize::scale_out)
        .def_readwrite("bias_term", &Requantize::bias_term)
        .def_readwrite("bias_data_size", &Requantize::bias_data_size)
        .def_readwrite("fusion_relu", &Requantize::fusion_relu)
        .def_readwrite("bias_data", &Requantize::bias_data)

            LAYER_COMMON;

    py::class_<Reshape_final, Layer>(m, "Reshape")
        .def(py::init<>())
        // reshape flag
        // 0 = copy from bottom
        // -1 = remaining
        // -233 = drop this dim (default)
        .def_readwrite("w", &Reshape::w)
        .def_readwrite("h", &Reshape::h)
        .def_readwrite("c", &Reshape::c)
        .def_readwrite("permute", &Reshape::permute)
        .def_readwrite("ndim", &Reshape::ndim)

            LAYER_COMMON;

    py::class_<ROIPooling_final, Layer>(m, "ROIPooling")
        .def(py::init<>())
        .def_readwrite("pooled_width", &ROIPooling::pooled_width)
        .def_readwrite("pooled_height", &ROIPooling::pooled_height)
        .def_readwrite("spatial_scale", &ROIPooling::spatial_scale)

            LAYER_COMMON;

    py::class_<Scale_final, Layer>(m, "Scale")
        .def(py::init<>())
        .def_readwrite("scale_data_size", &Scale::scale_data_size)
        .def_readwrite("bias_term", &Scale::bias_term)
        .def_readwrite("scale_data", &Scale::scale_data)
        .def_readwrite("bias_data", &Scale::bias_data)

            LAYER_COMMON;

    py::class_<SELU_final, Layer>(m, "SELU")
        .def(py::init<>())
        .def_readwrite("alpha", &SELU::alpha)
        .def_readwrite("lambda", &SELU::lambda)

            LAYER_COMMON;

    py::class_<ShuffleChannel_final, Layer>(m, "ShuffleChannel")
        .def(py::init<>())
        .def_readwrite("group", &ShuffleChannel::group)

            LAYER_COMMON;

    py::class_<Sigmoid_final, Layer>(m, "Sigmoid")
        .def(py::init<>())

            LAYER_COMMON;

    py::class_<Slice_final, Layer>(m, "Slice")
        .def(py::init<>())
        .def_readwrite("slices", &Slice::slices)
        .def_readwrite("axis", &Slice::axis)

            LAYER_COMMON;

    py::class_<Softmax_final, Layer>(m, "Softmax")
        .def(py::init<>())
        .def_readwrite("axis", &Softmax::axis)

            LAYER_COMMON;

    py::class_<Split_final, Layer>(m, "Split")
        .def(py::init<>())

            LAYER_COMMON;

    //    layer: SPP TODO

    py::class_<Squeeze_final, Layer>(m, "Squeeze")
        .def(py::init<>())
        .def_readwrite("squeeze_w", &Squeeze::squeeze_w)
        .def_readwrite("squeeze_h", &Squeeze::squeeze_h)
        .def_readwrite("squeeze_c", &Squeeze::squeeze_c)
        .def_readwrite("axes", &Squeeze::axes)

            LAYER_COMMON;

    py::class_<TanH_final, Layer>(m, "TanH").def(py::init<>())

        LAYER_COMMON;

    py::class_<Threshold_final, Layer>(m, "Threshold")
        .def(py::init<>())
        .def_readwrite("threshold", &Threshold::threshold)

            LAYER_COMMON;

    //    layer: Tile TODO

    py::class_<UnaryOp_final, Layer> unaryOp_final(m, "UnaryOp");
    unaryOp_final.def(py::init<>())
        .def_readwrite("op_type", &UnaryOp::op_type)

            LAYER_COMMON;

    py::enum_<UnaryOp_final::OperationType>(unaryOp_final, "OperationType")
        .value("Operation_ABS", UnaryOp::OperationType::Operation_ABS)
        .value("Operation_NEG", UnaryOp::OperationType::Operation_NEG)
        .value("Operation_FLOOR", UnaryOp::OperationType::Operation_FLOOR)
        .value("Operation_CEIL", UnaryOp::OperationType::Operation_CEIL)
        .value("Operation_SQUARE", UnaryOp::OperationType::Operation_SQUARE)
        .value("Operation_SQRT", UnaryOp::OperationType::Operation_SQRT)
        .value("Operation_RSQRT", UnaryOp::OperationType::Operation_RSQRT)
        .value("Operation_EXP", UnaryOp::OperationType::Operation_EXP)
        .value("Operation_LOG", UnaryOp::OperationType::Operation_LOG)
        .value("Operation_SIN", UnaryOp::OperationType::Operation_SIN)
        .value("Operation_COS", UnaryOp::OperationType::Operation_COS)
        .value("Operation_TAN", UnaryOp::OperationType::Operation_TAN)
        .value("Operation_ASIN", UnaryOp::OperationType::Operation_ASIN)
        .value("Operation_ACOS", UnaryOp::OperationType::Operation_ACOS)
        .value("Operation_ATAN", UnaryOp::OperationType::Operation_ATAN)
        .value("Operation_RECIPROCAL", UnaryOp::OperationType::Operation_RECIPROCAL)
        .value("Operation_TANH", UnaryOp::OperationType::Operation_TANH)
        // .export_values()
        ;

    //    layer: RNN TODO
    //    layer: LSTM TODO

    py::class_<YoloDetectionOutput_final, Layer>(m, "YoloDetectionOutput")
        .def(py::init<>())
        .def_readwrite("num_class", &YoloDetectionOutput::num_class)
        .def_readwrite("num_box", &YoloDetectionOutput::num_box)
        .def_readwrite("confidence_threshold", &YoloDetectionOutput::confidence_threshold)
        .def_readwrite("nms_threshold", &YoloDetectionOutput::nms_threshold)
        .def_readwrite("biases", &YoloDetectionOutput::biases)
        // ncnn::Layer* softmax; TODO

        LAYER_COMMON;

    py::class_<Yolov3DetectionOutput_final, Layer>(m, "Yolov3DetectionOutput")
        .def(py::init<>())
        .def_readwrite("num_class", &Yolov3DetectionOutput::num_class)
        .def_readwrite("num_box", &Yolov3DetectionOutput::num_box)
        .def_readwrite("confidence_threshold", &Yolov3DetectionOutput::confidence_threshold)
        .def_readwrite("nms_threshold", &Yolov3DetectionOutput::nms_threshold)
        .def_readwrite("biases", &Yolov3DetectionOutput::biases)
        .def_readwrite("mask", &Yolov3DetectionOutput::mask)
        .def_readwrite("anchors_scale", &Yolov3DetectionOutput::anchors_scale)
        .def_readwrite("mask_group_num", &Yolov3DetectionOutput::mask_group_num)
        // ncnn::Layer* softmax; TODO

        LAYER_COMMON;
}