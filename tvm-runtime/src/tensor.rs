use core::ffi::c_void;
use std::ptr::null_mut;

use tvm_ffi::{
    collections::tensor::DLTensorExt as _, dtype::AsDLDataType, DLDataType, DLDevice, DLDeviceType,
    NDAllocator, Tensor as TVMFFITensor,
};
use tvm_ffi_sys::DLTensor;
use tvm_runtime_sys::{TVMDeviceAPIAllocDataSpace, TVMDeviceAPIFreeDataSpace, TVMDeviceAPIGet};

fn host_of_device(device: DLDevice) -> Option<DLDeviceType> {
    match device.device_type {
        DLDeviceType::kDLCPU => None,
        DLDeviceType::kDLCUDA => Some(DLDeviceType::kDLCUDAHost),
        DLDeviceType::kDLCUDAHost => None,
        DLDeviceType::kDLOpenCL => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLVulkan => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLMetal => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLVPI => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLROCM => Some(DLDeviceType::kDLROCMHost),
        DLDeviceType::kDLROCMHost => None,
        DLDeviceType::kDLExtDev => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLCUDAManaged => Some(DLDeviceType::kDLCUDAHost),
        DLDeviceType::kDLOneAPI => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLWebGPU => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLHexagon => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLMAIA => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLTrn => Some(DLDeviceType::kDLCPU),
    }
}

fn is_host(device: DLDevice) -> bool {
    host_of_device(device).is_none()
}

#[derive(Debug, Clone)]
pub struct DeviceNDAlloc {}

unsafe impl Send for DeviceNDAlloc {}

unsafe impl Sync for DeviceNDAlloc {}

unsafe impl NDAllocator for DeviceNDAlloc {
    const MIN_ALIGN: usize = 64;

    unsafe fn alloc_data(&mut self, prototype: &DLTensor) -> *mut c_void {
        let numel = prototype.numel() as usize;
        let item_size = prototype.item_size();
        let size = numel * item_size as usize;
        let layout = std::alloc::Layout::from_size_align(size, Self::MIN_ALIGN).unwrap();
        TVMDeviceAPIAllocDataSpace(
            TVMDeviceAPIGet(prototype.device, false as i32),
            prototype.device,
            layout.size(),
            layout.align(),
            prototype.dtype,
        )
    }

    unsafe fn free_data(&mut self, tensor: &DLTensor) {
        TVMDeviceAPIFreeDataSpace(
            TVMDeviceAPIGet(tensor.device, false as i32),
            tensor.device,
            tensor.data,
        );
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct Tensor {
    inner: TVMFFITensor,
}

impl Tensor {
    pub fn empty(shape: &[i64], dtype: DLDataType, device: DLDevice) -> Self {
        Self {
            inner: TVMFFITensor::from_nd_alloc(DeviceNDAlloc {}, shape, dtype, device),
        }
    }

    pub fn empty_like(other: &Tensor, device: DLDevice) -> Self {
        Self::empty(other.shape(), other.dtype(), device)
    }

    pub fn shape(&self) -> &[i64] {
        self.inner.shape()
    }

    pub fn dtype(&self) -> DLDataType {
        self.inner.dtype()
    }

    pub fn device(&self) -> DLDevice {
        self.inner.device()
    }

    pub fn dltensor(&self) -> &DLTensor {
        self.inner.dltensor()
    }

    pub fn dltensor_mut(&mut self) -> &mut DLTensor {
        self.inner.dltensor_mut()
    }

    pub fn data_ptr(&self) -> *const core::ffi::c_void {
        self.inner.data_ptr()
    }

    pub fn data_ptr_mut(&mut self) -> *mut core::ffi::c_void {
        self.inner.data_ptr_mut()
    }

    pub fn data_as_slice<T: AsDLDataType>(&self) -> tvm_ffi::error::Result<&[T]> {
        self.inner.data_as_slice()
    }

    pub fn data_as_slice_mut<T: AsDLDataType>(&mut self) -> tvm_ffi::error::Result<&mut [T]> {
        self.inner.data_as_slice_mut()
    }

    pub fn copy_from(&mut self, src: &Self) -> tvm_ffi::error::Result<()> {
        if is_host(self.device()) && is_host(src.device()) {
            // Host to Host copy
            let numel = self.shape().iter().product::<i64>() as usize;
            let item_size = (self.dtype().bits / 8) as usize;
            let nbytes = numel * item_size;

            let src_data = src.data_as_slice::<u8>()?;
            debug_assert_eq!(
                nbytes,
                src_data.len(),
                "data length ({}) does not match tensor size ({})",
                src_data.len(),
                nbytes
            );
            let src_ptr = src_data.as_ptr() as *const u8;
            let dst_ptr = self.data_as_slice_mut()?.as_mut_ptr() as *mut u8;
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, nbytes);
            }
        } else {
            let device = if is_host(self.device()) {
                // Device to Host copy
                src.device()
            } else {
                // Host to Device copy, or
                // Device to Device copy
                self.device()
            };
            unsafe {
                let handle = TVMDeviceAPIGet(device, false as i32);
                tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
                    handle,
                    src.dltensor(),
                    self.dltensor_mut(),
                    null_mut(),
                );
                tvm_runtime_sys::TVMDeviceAPIStreamSync(handle, device, null_mut());
                tvm_runtime_sys::TVMDeviceAPIDestroy(handle);
            }
        }
        Ok(())
    }

    pub fn copy_from_slice(&mut self, src: &[u8]) -> tvm_ffi::error::Result<()> {
        if self.device().device_type == DLDeviceType::kDLCPU {
            // Host to Host copy
            let numel = self.shape().iter().product::<i64>() as usize;
            let item_size = (self.dtype().bits / 8) as usize;
            let nbytes = numel * item_size;

            debug_assert_eq!(
                nbytes,
                src.len(),
                "data length ({}) does not match tensor size ({})",
                src.len(),
                nbytes
            );
            let src_ptr = src.as_ptr() as *const u8;
            let dst_ptr = self.data_as_slice_mut()?.as_mut_ptr() as *mut u8;
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, nbytes);
            }
        } else {
            // Host to Device copy
            let host_dltensor = DLTensor {
                data: src.as_ptr() as *mut c_void,
                device: DLDevice {
                    device_type: DLDeviceType::kDLCPU,
                    device_id: 0,
                },
                ndim: self.dltensor().ndim,
                dtype: self.dltensor().dtype,
                shape: self.dltensor().shape,
                strides: self.dltensor().strides,
                byte_offset: self.dltensor().byte_offset,
            };
            unsafe {
                let handle = TVMDeviceAPIGet(self.device(), false as i32);
                tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
                    handle,
                    &host_dltensor,
                    self.dltensor_mut(),
                    null_mut(),
                );
                tvm_runtime_sys::TVMDeviceAPIStreamSync(handle, self.device(), null_mut());
                tvm_runtime_sys::TVMDeviceAPIDestroy(handle);
            }
        }

        Ok(())
    }

    pub fn reshape(&mut self, shape: &[i64]) -> tvm_ffi::error::Result<Tensor> {
        let nelem_before = self.shape().iter().product::<i64>();
        let nelem_after = shape.iter().product::<i64>();
        if nelem_before != nelem_after {
            tvm_ffi::bail!(
                tvm_ffi::VALUE_ERROR,
                "Cannot reshape from {:?} to {:?}",
                self.shape(),
                shape,
            );
        }

        let mut new_tensor = Self::empty(shape, self.dtype(), self.device());
        new_tensor.copy_from(self).unwrap();

        Ok(new_tensor)
    }
}

impl From<TVMFFITensor> for Tensor {
    fn from(inner: TVMFFITensor) -> Self {
        Self { inner }
    }
}

impl From<Tensor> for TVMFFITensor {
    fn from(value: Tensor) -> Self {
        value.inner
    }
}
