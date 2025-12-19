use core::{ffi::c_void, ptr::null_mut};

use tvm_ffi::{
    collections::tensor::DLTensorExt, CPUNDAlloc, DLDevice, DLDeviceType, NDAllocator, Tensor,
};
use tvm_ffi_sys::DLTensor;
use tvm_runtime_sys::{TVMDeviceAPIAllocDataSpace, TVMDeviceAPIFreeDataSpace, TVMDeviceAPIGet};

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

pub trait TensorCopy {
    fn copy_from(&mut self, src: &Self) -> tvm_ffi::error::Result<()>;

    fn copy_to(&self, dst: &mut Self) -> tvm_ffi::error::Result<()>;

    fn copy_from_slice(&mut self, slice: &[u8]) -> tvm_ffi::error::Result<()>;
}

impl TensorCopy for Tensor {
    fn copy_from(&mut self, src: &Self) -> tvm_ffi::error::Result<()> {
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

    fn copy_to(&self, dst: &mut Self) -> tvm_ffi::error::Result<()> {
        if is_host(self.device()) && is_host(dst.device()) {
            // Host to Host copy
            let numel = self.shape().iter().product::<i64>() as usize;
            let item_size = (self.dtype().bits / 8) as usize;
            let nbytes = numel * item_size;

            let src_data = self.data_as_slice::<u8>()?;
            debug_assert_eq!(
                nbytes,
                src_data.len(),
                "data length ({}) does not match tensor size ({})",
                src_data.len(),
                nbytes
            );
            let src_ptr = src_data.as_ptr() as *const u8;
            let dst_ptr = dst.data_as_slice_mut()?.as_mut_ptr() as *mut u8;
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, nbytes);
            }
        } else {
            let device = if is_host(self.device()) {
                // Host to Device copy
                dst.device()
            } else {
                // Device to Host copy, or
                // Device to Device copy
                self.device()
            };
            unsafe {
                let handle = TVMDeviceAPIGet(device, false as i32);
                tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
                    handle,
                    self.dltensor(),
                    dst.dltensor_mut(),
                    null_mut(),
                );
                tvm_runtime_sys::TVMDeviceAPIStreamSync(handle, device, null_mut());
                tvm_runtime_sys::TVMDeviceAPIDestroy(handle);
            }
        }
        Ok(())
    }

    fn copy_from_slice(&mut self, src: &[u8]) -> tvm_ffi::error::Result<()> {
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
}

pub trait TensorReshape {
    fn reshape(&mut self, shape: &[i64]) -> tvm_ffi::error::Result<Tensor>;
}

impl TensorReshape for Tensor {
    fn reshape(&mut self, shape: &[i64]) -> tvm_ffi::error::Result<Tensor> {
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

        let mut new_tensor = if self.device().device_type == DLDeviceType::kDLCPU {
            Tensor::from_nd_alloc(CPUNDAlloc {}, shape, self.dtype(), self.device())
        } else {
            Tensor::from_nd_alloc(DeviceNDAlloc {}, shape, self.dtype(), self.device())
        };
        new_tensor.copy_from(self).unwrap();

        Ok(new_tensor)
    }
}

// #[derive(Debug, Clone)]
// pub struct RTensor {
//     shape: Vec<i64>,
//     dltensor: DLTensor,
//     alloc: DeviceNDAlloc,
// }

// unsafe impl Send for RTensor {}

// impl RTensor {
//     pub fn new(
//         device: DLDevice,
//         shape: impl IntoIterator<Item = impl Into<i64>>,
//         dtype: DLDataType,
//     ) -> Self {
//         let mut shape = shape.into_iter().map(|v| v.into()).collect::<Vec<_>>();
//         let ndim = shape.len() as i32;
//         let mut dltensor = DLTensor {
//             data: null_mut(),
//             device,
//             ndim,
//             dtype,
//             shape: shape.as_mut_ptr(),
//             strides: null_mut(),
//             byte_offset: 0,
//         };
//         let mut alloc = DeviceNDAlloc {};
//         dltensor.data = unsafe { alloc.alloc_data(&dltensor) };
//         Self {
//             shape,
//             dltensor,
//             alloc,
//         }
//     }

//     pub fn nbytes(&self) -> usize {
//         let dtype_bytes = (self.dltensor.dtype.bits / 8) as u32;
//         let elems = self.shape.iter().product::<i64>() as u32;
//         (dtype_bytes * elems) as usize
//     }

//     pub fn host_device_type(&self) -> Option<DLDeviceType> {
//         match self.dltensor.device.device_type {
//             DLDeviceType::kDLCPU => None,
//             DLDeviceType::kDLCUDA => Some(DLDeviceType::kDLCUDAHost),
//             DLDeviceType::kDLCUDAHost => None,
//             DLDeviceType::kDLOpenCL => Some(DLDeviceType::kDLCPU),
//             DLDeviceType::kDLVulkan => Some(DLDeviceType::kDLCPU),
//             DLDeviceType::kDLMetal => Some(DLDeviceType::kDLCPU),
//             DLDeviceType::kDLVPI => Some(DLDeviceType::kDLCPU),
//             DLDeviceType::kDLROCM => Some(DLDeviceType::kDLROCMHost),
//             DLDeviceType::kDLROCMHost => None,
//             DLDeviceType::kDLExtDev => Some(DLDeviceType::kDLCPU),
//             DLDeviceType::kDLCUDAManaged => Some(DLDeviceType::kDLCUDAHost),
//             DLDeviceType::kDLOneAPI => Some(DLDeviceType::kDLCPU),
//             DLDeviceType::kDLWebGPU => Some(DLDeviceType::kDLCPU),
//             DLDeviceType::kDLHexagon => Some(DLDeviceType::kDLCPU),
//             DLDeviceType::kDLMAIA => Some(DLDeviceType::kDLCPU),
//             DLDeviceType::kDLTrn => Some(DLDeviceType::kDLCPU),
//         }
//     }

//     pub fn is_host_tensor(&self) -> bool {
//         !self.is_device_tensor()
//     }

//     pub fn is_device_tensor(&self) -> bool {
//         self.host_device_type().is_some()
//     }

//     pub fn copy_from(&mut self, src: &mut Self) -> () {
//         if self.is_host_tensor() && src.is_host_tensor() {
//             // Host -> Host Copy
//             let numel = self.dltensor.numel() as usize;
//             let item_size = self.dltensor.item_size() as usize;
//             let expected_bytes = numel * item_size;
//             let data = src.data_as_slice();
//             debug_assert_eq!(
//                 expected_bytes,
//                 data.len(),
//                 "RTensor::copy_data: data length ({}) does not match tensor size ({})",
//                 data.len(),
//                 expected_bytes
//             );
//             unsafe {
//                 core::ptr::copy_nonoverlapping(
//                     data.as_ptr(),
//                     self.dltensor.data as *mut u8,
//                     expected_bytes,
//                 )
//             };
//         } else if self.is_host_tensor() {
//             // Device -> Host Copy
//             unsafe {
//                 tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
//                     TVMDeviceAPIGet(src.dltensor.device, false as i32),
//                     &mut src.dltensor,
//                     &mut self.dltensor,
//                     null_mut(),
//                 )
//             };
//         } else {
//             // Host -> Device Copy
//             unsafe {
//                 tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
//                     TVMDeviceAPIGet(self.dltensor.device, false as i32),
//                     &mut src.dltensor,
//                     &mut self.dltensor,
//                     null_mut(),
//                 )
//             };
//         }
//     }

//     pub fn copy_from_slice(&mut self, data: &[u8]) {
//         if self.is_host_tensor() {
//             let numel = self.dltensor.numel() as usize;
//             let item_size = self.dltensor.item_size() as usize;
//             let expected_bytes = numel * item_size;
//             debug_assert_eq!(
//                 expected_bytes,
//                 data.len(),
//                 "RTensor::copy_data: data length ({}) does not match tensor size ({})",
//                 data.len(),
//                 expected_bytes
//             );
//             unsafe {
//                 core::ptr::copy_nonoverlapping(
//                     data.as_ptr(),
//                     self.dltensor.data as *mut u8,
//                     expected_bytes,
//                 )
//             };
//         } else {
//             let host_device_type = self.host_device_type().unwrap_or(DLDeviceType::kDLCPU);
//             let mut host_dltensor = DLTensor {
//                 data: data.as_ptr() as *mut c_void,
//                 device: DLDevice {
//                     device_type: host_device_type,
//                     device_id: 0,
//                 },
//                 ndim: self.dltensor.ndim,
//                 dtype: self.dltensor.dtype,
//                 shape: self.dltensor.shape,
//                 strides: self.dltensor.strides,
//                 byte_offset: self.dltensor.byte_offset,
//             };
//             unsafe {
//                 tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
//                     TVMDeviceAPIGet(self.dltensor.device, false as i32),
//                     &mut host_dltensor,
//                     &mut self.dltensor,
//                     null_mut(),
//                 )
//             };
//         }
//     }

//     pub fn data(&self) -> *mut u8 {
//         self.dltensor.data as *mut u8
//     }

//     pub fn data_as_slice(&self) -> &[u8] {
//         unsafe { std::slice::from_raw_parts(self.data(), self.nbytes()) }
//     }

//     pub fn shape(&self) -> &Vec<i64> {
//         &self.shape
//     }
// }

// impl Drop for RTensor {
//     fn drop(&mut self) {
//         unsafe { DeviceNDAlloc {}.free_data(&self.dltensor) };
//     }
// }

// impl AsRef<DLTensor> for RTensor {
//     fn as_ref(&self) -> &DLTensor {
//         &self.dltensor
//     }
// }

// impl AsMut<DLTensor> for RTensor {
//     fn as_mut(&mut self) -> &mut DLTensor {
//         &mut self.dltensor
//     }
// }

#[cfg(test)]
mod tests {
    use tvm_ffi_sys::{DLDataType, DLDataTypeCode};

    use super::*;

    #[test]
    fn test_tensor_copy() -> () {
        let mut src = Vec::<f32>::new();
        for i in 0..9 {
            src.push(i as f32);
        }
        let src_u8: &[u8] = unsafe {
            core::slice::from_raw_parts(
                src.as_ptr() as *const u8,
                src.len() * core::mem::size_of::<f32>(),
            )
        };

        let mut tensor_metal = tvm_ffi::Tensor::from_nd_alloc(
            DeviceNDAlloc {},
            &[3, 3],
            DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 32,
                lanes: 1,
            },
            DLDevice {
                device_type: tvm_ffi::DLDeviceType::kDLMetal,
                device_id: 0,
            },
        );
        tensor_metal.copy_from_slice(src_u8).unwrap();

        let mut tensor_cpu = tvm_ffi::Tensor::from_nd_alloc(
            CPUNDAlloc {},
            &[3, 3],
            DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 32,
                lanes: 1,
            },
            DLDevice {
                device_type: tvm_ffi::DLDeviceType::kDLCPU,
                device_id: 0,
            },
        );
        tensor_cpu.copy_from(&tensor_metal).unwrap();

        assert_eq!(src.as_slice(), tensor_cpu.data_as_slice::<f32>().unwrap());
    }
}
