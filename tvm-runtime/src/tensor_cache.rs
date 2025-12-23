use std::{collections::HashMap, path::PathBuf};

use anyhow::{anyhow, bail};
use serde::{Deserialize, Serialize};
use tvm_ffi::{Array, DLDataType, DLDataTypeCode, DLDataTypeExt, DLDevice};

use crate::Tensor;

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
    pub fn from(json_path: &PathBuf, device: DLDevice) -> anyhow::Result<Self> {
        if !json_path.exists() {
            bail!("tensor cache json file does not exist");
        }

        let tensor_cache_json = std::fs::read_to_string(json_path)
            .map_err(|e| anyhow!("Failed to open tensor cache json: {}", e.to_string()))?;
        let mut tensor_cache = serde_json::from_str::<Self>(&tensor_cache_json)
            .map_err(|e| anyhow!("Failed to deserialize tensor cache json: {}", e.to_string()))?;

        let base_path = json_path.parent().ok_or(anyhow!(
            "Failed to get the parent path of tensor cache json file"
        ))?;
        for file_record in tensor_cache.records.iter() {
            let record_bytes =
                std::fs::read(base_path.join(&file_record.data_path)).map_err(|e| {
                    anyhow!(
                        "Failed to open the record {}: {}",
                        file_record.data_path,
                        e.to_string()
                    )
                })?;
            for param_record in file_record.records.iter() {
                let dtype = DLDataType::try_from_str(&param_record.dtype).unwrap();
                let mut tensor = Tensor::empty(
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

                    let decoded_ptr = unsafe {
                        std::slice::from_raw_parts(
                            decoded.as_ptr() as *const u8,
                            decoded.len() * std::mem::size_of::<u32>(),
                        )
                    };
                    tensor
                        .copy_from_slice(decoded_ptr)
                        .map_err(|e| anyhow!("Failed to copy param data: {}", e.to_string()))?;
                } else {
                    // Copy sliced data
                    let sliced = unsafe {
                        std::slice::from_raw_parts(
                            record_bytes.as_ptr().add(param_record.byte_offset),
                            param_record.nbytes,
                        )
                    };
                    tensor
                        .copy_from_slice(sliced)
                        .map_err(|e| anyhow!("Failed to copy param data: {}", e.to_string()))?;
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

    pub fn get_params(&self, param_names: Vec<&str>) -> Array<tvm_ffi::Tensor> {
        let mut params: Vec<tvm_ffi::Tensor> = vec![];
        for param in param_names {
            let tensor = self.pool.get(param).unwrap();
            params.push(tensor.clone().into());
        }
        Array::new(params)
    }
}
