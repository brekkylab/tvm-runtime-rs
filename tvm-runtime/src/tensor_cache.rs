use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};
use tvm_ffi::{Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Tensor};

use crate::TensorCopy;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TensorFormat {
    #[default]
    #[serde(rename = "f32-to-bf16")]
    F32ToBf16,
    #[serde(rename = "raw")]
    Raw,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct TensorCacheEntry {
    pub name: String,
    pub shape: Vec<u32>,
    pub dtype: String,
    pub format: TensorFormat,
    pub byte_offset: usize,
    pub nbytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ShardFormat {
    #[default]
    #[serde(rename = "raw-shard")]
    RawShard,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct TensorShardEntry {
    pub data_path: String,
    pub format: ShardFormat,
    pub nbytes: usize,
    pub records: Vec<TensorCacheEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "PascalCase")]
pub struct TensorCacheMetadata {
    pub param_size: f32,
    pub param_bytes: f32,
    pub bits_per_param: f32,
}

#[derive(Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TensorCache {
    pub metadata: TensorCacheMetadata,
    pub records: Vec<TensorShardEntry>,

    #[serde(skip)]
    pool: HashMap<String, Tensor>,
}

impl TensorCache {
    pub fn from_str(s: &str) -> serde_json::Result<Self> {
        let cache: TensorCache = serde_json::from_str(s)?;
        Ok(cache)
    }

    pub fn from(
        path: &PathBuf,
        device_type: tvm_ffi::DLDeviceType,
        device_id: i32,
    ) -> anyhow::Result<Self> {
        let device = tvm_ffi::DLDevice::new(device_type, device_id);
        let tensor_cache_json_path = std::fs::read_to_string(path.join("ndarray-cache.json"))
            .map_err(|e| anyhow::anyhow!("Failed to open tensor-cache.json: {}", e.to_string()))?;
        let mut tensor_cache = TensorCache::from_str(&tensor_cache_json_path).map_err(|e| {
            anyhow::anyhow!("Failed to deserialize tensor-cache.json: {}", e.to_string())
        })?;
        for file_record in tensor_cache.records.iter() {
            let record_bytes = std::fs::read(path.join(&file_record.data_path)).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to open the record {}: {}",
                    file_record.data_path,
                    e.to_string()
                )
            })?;
            for param_record in file_record.records.iter() {
                let dtype = DLDataType::try_from_str(&param_record.dtype).unwrap();
                let mut tensor = Tensor::from_nd_alloc(
                    crate::rtensor::DeviceNDAlloc {},
                    param_record
                        .shape
                        .iter()
                        .map(|dim| dim.clone() as i64)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    dtype,
                    device,
                );

                if dtype.code == DLDataTypeCode::kDLFloat as u8
                    && dtype.bits == 32
                    && param_record.format == TensorFormat::F32ToBf16
                {
                    // Decode bf16 to f32
                    let mut buffer: Vec<u16> = Vec::with_capacity(param_record.nbytes / 2);
                    let mut decoded: Vec<u32> = Vec::with_capacity(param_record.nbytes / 2);
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            record_bytes.as_ptr().wrapping_add(param_record.byte_offset),
                            buffer.as_mut_ptr() as *mut u8,
                            param_record.nbytes,
                        )
                    };
                    for (idx, item) in buffer.into_iter().enumerate() {
                        decoded[idx] = (item as u32) << 16;
                    }
                    tensor
                        .copy_from_slice(bytemuck::cast_slice(&decoded))
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to copy param data: {}", e.to_string())
                        })?;
                } else {
                    // Copy sliced data
                    let sliced = unsafe {
                        std::slice::from_raw_parts(
                            record_bytes.as_ptr().add(param_record.byte_offset),
                            param_record.nbytes,
                        )
                    };
                    tensor.copy_from_slice(sliced).map_err(|e| {
                        anyhow::anyhow!("Failed to copy param data: {}", e.to_string())
                    })?;
                }

                tensor_cache
                    .pool
                    .insert(param_record.name.clone(), tensor.into());
            }
        }
        Ok(tensor_cache)
    }

    pub fn to_string(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    pub fn get_params(&self, param_names: Vec<&str>) -> Array<Tensor> {
        let mut array = Array::<Tensor>::default();
        for param in param_names {
            let tensor = self.pool.get(param).unwrap();
            array.push(tensor.clone());
        }
        array
    }
}
