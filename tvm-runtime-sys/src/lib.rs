use std::os::raw::{c_char, c_int, c_void};

use tvm_ffi_sys::{DLDataType, DLDevice, DLTensor, TVMFFIStreamHandle};

#[repr(C)]
pub struct TVMDeviceAPI {
    _private: [u8; 0],
}

extern "C" {
    pub fn TVMDeviceAPIGet(dev: DLDevice, allow_missing: c_int) -> *mut TVMDeviceAPI;
    pub fn TVMDeviceAPIDestroy(handle: *mut TVMDeviceAPI);

    pub fn TVMDeviceAPISetDevice(handle: *mut TVMDeviceAPI, dev: DLDevice);

    pub fn TVMDeviceAPIGetAttr(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
        kind: c_int,
        out_any: *mut c_void, // ffi::Any*
    );

    pub fn TVMDeviceAPIGetDataSize(
        handle: *mut TVMDeviceAPI,
        arr: *const DLTensor,
        mem_scope: *const c_char, // nullable
    ) -> usize;

    pub fn TVMDeviceAPIGetTargetProperty(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
        property: *const c_char,
        out_any: *mut c_void,
    );

    pub fn TVMDeviceAPIAllocDataSpace(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
        nbytes: usize,
        alignment: usize,
        type_hint: DLDataType,
    ) -> *mut c_void;

    pub fn TVMDeviceAPIAllocDataSpaceND(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
        ndim: c_int,
        shape: *const i64,
        dtype: DLDataType,
        mem_scope: *const c_char, // nullable
    ) -> *mut c_void;

    pub fn TVMDeviceAPIFreeDataSpace(handle: *mut TVMDeviceAPI, dev: DLDevice, ptr: *mut c_void);

    pub fn TVMDeviceAPICopyDataFromTo(
        handle: *mut TVMDeviceAPI,
        from: *const DLTensor,
        to: *mut DLTensor,
        stream: TVMFFIStreamHandle, // nullable
    );

    pub fn TVMDeviceAPICreateStream(handle: *mut TVMDeviceAPI, dev: DLDevice)
        -> TVMFFIStreamHandle;

    pub fn TVMDeviceAPIFreeStream(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
        stream: TVMFFIStreamHandle,
    );

    pub fn TVMDeviceAPIStreamSync(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
        stream: TVMFFIStreamHandle,
    );

    pub fn TVMDeviceAPISetStream(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
        stream: TVMFFIStreamHandle,
    );

    pub fn TVMDeviceAPIGetCurrentStream(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
    ) -> TVMFFIStreamHandle;

    pub fn TVMDeviceAPISyncStreamFromTo(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
        event_src: TVMFFIStreamHandle,
        event_dst: TVMFFIStreamHandle,
    );

    pub fn TVMDeviceAPIAllocWorkspace(
        handle: *mut TVMDeviceAPI,
        dev: DLDevice,
        nbytes: usize,
        type_hint: DLDataType,
    ) -> *mut c_void;

    pub fn TVMDeviceAPIFreeWorkspace(handle: *mut TVMDeviceAPI, dev: DLDevice, ptr: *mut c_void);

    pub fn TVMDeviceAPISupportsDevicePointerArithmeticsOnHost(handle: *mut TVMDeviceAPI) -> c_int;
}
