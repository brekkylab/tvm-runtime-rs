use core::ffi::c_void;
use std::ptr::null_mut;

use tvm_ffi::{
    collections::tensor::DLTensorExt as _, DLDataType, DLDevice, DLDeviceType, NDAllocator,
    Tensor as TVMFFITensor,
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
struct DeviceNDAlloc {}

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
        Self::empty(other.inner.shape(), other.inner.dtype(), device)
    }

    fn copy_from(&mut self, src: &Self) -> tvm_ffi::error::Result<()> {
        if is_host(self.inner.device()) && is_host(src.inner.device()) {
            // Host to Host copy
            let numel = self.inner.shape().iter().product::<i64>() as usize;
            let item_size = (self.inner.dtype().bits / 8) as usize;
            let nbytes = numel * item_size;

            let src_data = src.inner.data_as_slice::<u8>()?;
            debug_assert_eq!(
                nbytes,
                src_data.len(),
                "data length ({}) does not match tensor size ({})",
                src_data.len(),
                nbytes
            );
            let src_ptr = src_data.as_ptr() as *const u8;
            let dst_ptr = self.inner.data_as_slice_mut()?.as_mut_ptr() as *mut u8;
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, nbytes);
            }
        } else {
            let device = if is_host(self.inner.device()) {
                // Device to Host copy
                src.inner.device()
            } else {
                // Host to Device copy, or
                // Device to Device copy
                self.inner.device()
            };
            unsafe {
                let handle = TVMDeviceAPIGet(device, false as i32);
                tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
                    handle,
                    src.inner.dltensor(),
                    self.inner.dltensor_mut(),
                    null_mut(),
                );
                tvm_runtime_sys::TVMDeviceAPIStreamSync(handle, device, null_mut());
                tvm_runtime_sys::TVMDeviceAPIDestroy(handle);
            }
        }
        Ok(())
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
