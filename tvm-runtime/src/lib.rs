mod tensor;
mod tensor_cache;

pub use tensor::*;
pub use tensor_cache::*;

use std::str::FromStr;

use anyhow::Context;
use minijinja::{context, Environment};
use minijinja_contrib::{add_to_environment, pycompat::unknown_method_callback};
use serde_json::json;
use tokenizers::tokenizer::Tokenizer as HFTokenizer;

#[derive(Debug, Clone)]
pub struct ChatTemplate<'a> {
    key: String,
    env: Environment<'a>,
}

impl<'a> ChatTemplate<'a> {
    pub fn new(key: String, source: String) -> Self {
        let mut env = Environment::new();
        add_to_environment(&mut env);
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_template_owned(key.clone(), source).unwrap();

        Self { key, env }
    }

    pub fn apply(&self, message: String, add_generation_prompt: bool) -> anyhow::Result<String> {
        let ctx = context!(
            messages => json!([{"role": "system", "contents": [{"type":"text", "text": "You are an assistant."}]}, {"role": "user", "contents": message}]),
            add_generation_prompt => add_generation_prompt,
        );
        self.env
            .get_template(&self.key)
            .unwrap()
            .render(ctx)
            .context("minijinja::render failed")
    }
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: HFTokenizer,
}

impl Tokenizer {
    pub fn new(config: &str) -> Self {
        Tokenizer {
            inner: HFTokenizer::from_str(config).unwrap().into(),
        }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        let encoded = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Tokenizer::encode failed: {}", e))?;
        Ok(encoded.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Tokenizer::decode failed: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use serde_json::Value;
    use std::{io::Write, path::PathBuf};
    use tvm_ffi::{
        collections::array::ArrayObj, AnyCompatible, DLDataType, DLDataTypeCode, DLDevice,
        DLDeviceType,
    };

    use super::*;

    const PAGE_SIZE: i64 = 16;

    #[test]
    fn test_module() -> () {
        let base_path = PathBuf::from("/Users/haejoonkim/.cache/ailoy");
        let model_path = base_path.join("Qwen--Qwen3-8B");
        let runtime_path = base_path.join("Qwen--Qwen3-8B--aarch64-apple-darwin--metal");

        let exec = tvm_ffi::Module::load_from_file(runtime_path.join("rt.dylib").to_string_lossy())
            .unwrap();
        let vm: tvm_ffi::Module = exec
            .get_function("vm_load_executable")
            .unwrap()
            .call_tuple(())
            .unwrap()
            .try_into()
            .unwrap();
        vm.get_function("vm_initialization")
            .unwrap()
            .call_tuple((
                tvm_ffi::DLDeviceType::kDLMetal as i32, // device_type
                0i32,                                   // device_id
                2i32,                                   // vm_allocator_type
                tvm_ffi::DLDeviceType::kDLCPU as i32,   // host_device_type
                0i32,                                   // host_device_id
                2i32,                                   // host_vm_allocator_type
            ))
            .unwrap();
        let metadata: tvm_ffi::String = vm
            .get_function("_metadata")
            .unwrap()
            .call_tuple(())
            .unwrap()
            .try_into()
            .unwrap();
        let metadata: Value = serde_json::from_str(&metadata).unwrap();

        let start = std::time::Instant::now();
        let tensor_cache = TensorCache::from(&model_path, DLDeviceType::kDLMetal, 0).unwrap();
        let duration = start.elapsed();
        println!("Time elapsed for tensor cache load: {:?}", duration);

        let context_window_size = metadata
            .get("context_window_size")
            .unwrap()
            .as_i64()
            .unwrap();
        let prefill_chunk_size = metadata
            .get("prefill_chunk_size")
            .unwrap()
            .as_i64()
            .unwrap();
        let sliding_window_size = metadata
            .get("sliding_window_size")
            .unwrap()
            .as_i64()
            .unwrap();

        let kv_cache = vm
            .get_function("create_tir_paged_kv_cache")
            .unwrap()
            .call_tuple((
                tvm_ffi::Shape::from([1]),                   // max_batch_size
                tvm_ffi::Shape::from([context_window_size]), // max_total_seq_len
                tvm_ffi::Shape::from([prefill_chunk_size]),  // prefill_chunk_size
                tvm_ffi::Shape::from([PAGE_SIZE]),           // page_size
                tvm_ffi::Shape::from([(sliding_window_size != -1) as i64]), // support_sliding_window
            ))
            .unwrap();

        let fkv_state_clear = tvm_ffi::Function::get_global("vm.builtin.kv_state_clear").unwrap();
        let fkv_state_add_sequence =
            tvm_ffi::Function::get_global("vm.builtin.kv_state_add_sequence").unwrap();
        let fkv_state_remove_sequence =
            tvm_ffi::Function::get_global("vm.builtin.kv_state_remove_sequence").unwrap();
        let fkv_state_fork_sequence =
            tvm_ffi::Function::get_global("vm.builtin.kv_state_fork_sequence").unwrap();
        let fkv_state_begin_forward =
            tvm_ffi::Function::get_global("vm.builtin.kv_state_begin_forward").unwrap();
        let fkv_state_end_forward =
            tvm_ffi::Function::get_global("vm.builtin.kv_state_end_forward").unwrap();
        let fkv_state_popn = tvm_ffi::Function::get_global("vm.builtin.kv_state_popn").unwrap();
        let fkv_cache_get_num_available_pages =
            tvm_ffi::Function::get_global("vm.builtin.attention_kv_cache_get_num_available_pages")
                .unwrap();
        let fkv_cache_get_total_sequence_length = tvm_ffi::Function::get_global(
            "vm.builtin.attention_kv_cache_get_total_sequence_length",
        )
        .unwrap();

        fkv_state_add_sequence
            .call_packed(&[
                (&kv_cache).into(),
                tvm_ffi::AnyView::from(&tvm_ffi::Any::from(0)),
            ])
            .unwrap();

        let fembed = vm.get_function("embed").unwrap();
        let fprefill = vm.get_function("prefill").unwrap();
        let fdecode = vm.get_function("decode").unwrap();
        let fapply_bitmask_inplace = vm.get_function("apply_bitmask_inplace").unwrap();
        let fsample_top_p_from_logits =
            tvm_ffi::Function::get_global("vm.builtin.sample_top_p_from_logits").unwrap();

        let tokenizer_config_json =
            std::fs::read_to_string(model_path.join("tokenizer.json")).unwrap();
        let tokenizer = Tokenizer::new(&tokenizer_config_json);

        let device = DLDevice {
            device_type: DLDeviceType::kDLMetal,
            device_id: 0,
        };
        let param_names = metadata
            .get("params")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.get("name").unwrap().as_str().unwrap())
            .collect::<Vec<_>>();
        let params = tensor_cache.get_params(param_names);

        let prefill = |tokens: &[u32]| {
            let new_tokens: Vec<i32> = tokens.to_vec().into_iter().map(|t| t as i32).collect();
            let prefill_chunk_size = prefill_chunk_size as usize;
            for i in (0..new_tokens.len()).step_by(prefill_chunk_size) {
                let j = if i + prefill_chunk_size < new_tokens.len() {
                    i + prefill_chunk_size
                } else {
                    new_tokens.len()
                };
                let length = j - i;

                let mut input = Tensor::empty(
                    &[length as i64],
                    DLDataType {
                        code: DLDataTypeCode::kDLInt as u8,
                        bits: 32,
                        lanes: 1,
                    },
                    device,
                );
                unsafe {
                    let tokens_sliced = std::slice::from_raw_parts(
                        new_tokens.as_ptr().add(i) as *const u8,
                        length * std::mem::size_of::<i32>(),
                    );
                    input.copy_from_slice(tokens_sliced).unwrap();
                }

                let embedding: tvm_ffi::Tensor = fembed
                    .call_packed(&[
                        tvm_ffi::AnyView::from(&<tvm_ffi::Tensor as From<Tensor>>::from(input)),
                        tvm_ffi::AnyView::from(&params),
                    ])
                    .unwrap()
                    .try_into()
                    .unwrap();
                let mut embedding: Tensor = embedding.into();

                let embedding_reshaped = embedding
                    .reshape(&[1, embedding.shape()[0], embedding.shape()[1]])
                    .unwrap();

                fkv_state_begin_forward
                    .call_packed(&[
                        tvm_ffi::AnyView::from(&kv_cache),
                        tvm_ffi::AnyView::from(&tvm_ffi::Shape::from(vec![0])), // sequence id
                        tvm_ffi::AnyView::from(&tvm_ffi::Shape::from(vec![length as i64])),
                    ])
                    .unwrap();
                fprefill
                    .call_packed(&[
                        tvm_ffi::AnyView::from(&<tvm_ffi::Tensor as From<Tensor>>::from(
                            embedding_reshaped,
                        )),
                        tvm_ffi::AnyView::from(&kv_cache),
                        tvm_ffi::AnyView::from(&params),
                    ])
                    .unwrap();
                fkv_state_end_forward
                    .call_packed(&[(&kv_cache).into()])
                    .unwrap();
            }
        };

        let decode = |last_token: u32| {
            let mut input = Tensor::empty(
                &[1 as i64],
                DLDataType {
                    code: DLDataTypeCode::kDLInt as u8,
                    bits: 32,
                    lanes: 1,
                },
                device,
            );
            unsafe {
                let tokens = vec![last_token as i32];
                let tokens_sliced = std::slice::from_raw_parts(
                    tokens.as_ptr() as *const u8,
                    std::mem::size_of::<i32>(),
                );
                input.copy_from_slice(tokens_sliced).unwrap();
            }

            let embedding: tvm_ffi::Tensor = fembed
                .call_packed(&[
                    tvm_ffi::AnyView::from(&<tvm_ffi::Tensor as From<Tensor>>::from(input)),
                    tvm_ffi::AnyView::from(&params),
                ])
                .unwrap()
                .try_into()
                .unwrap();
            let mut embedding: Tensor = embedding.into();
            let embedding_reshaped = embedding.reshape(&[1, 1, embedding.shape()[1]]).unwrap();

            fkv_state_begin_forward
                .call_packed(&[
                    tvm_ffi::AnyView::from(&kv_cache),
                    tvm_ffi::AnyView::from(&tvm_ffi::Shape::from(vec![0])), // sequence id
                    tvm_ffi::AnyView::from(&tvm_ffi::Shape::from(vec![1 as i64])),
                ])
                .unwrap();
            let output = fdecode
                .call_packed(&[
                    tvm_ffi::AnyView::from(&<tvm_ffi::Tensor as From<Tensor>>::from(
                        embedding_reshaped,
                    )),
                    tvm_ffi::AnyView::from(&kv_cache),
                    tvm_ffi::AnyView::from(&params),
                ])
                .unwrap();
            fkv_state_end_forward
                .call_packed(&[(&kv_cache).into()])
                .unwrap();

            // The output of decode is an Array of 2 items: logits(Tensor) and kv cache.
            // Since it's not possible to convert Any to Array<Any>, we just try to access to the pointer of ArrayObj directly and get the first item only.
            let logits = unsafe {
                let output_raw = tvm_ffi::Any::into_raw_ffi_any(output);
                assert_eq!(
                    output_raw.type_index,
                    tvm_ffi_sys::TVMFFITypeIndex::kTVMFFIArray as i32
                );

                let array_ptr = output_raw.data_union.v_obj as *const ArrayObj;
                let array_obj = &*array_ptr;

                let array_data_base_ptr = array_obj.data as *const tvm_ffi_sys::TVMFFIAny;
                let array_first_item = &*array_data_base_ptr;
                assert_eq!(
                    array_first_item.type_index,
                    tvm_ffi_sys::TVMFFITypeIndex::kTVMFFITensor as i32
                );

                let tensor = tvm_ffi::Tensor::try_cast_from_any_view(array_first_item).unwrap();
                tensor
            };

            logits
        };

        let sample = |logits: &tvm_ffi::Tensor, temperature: f64, top_p: f64| {
            let mut rng = rand::rng();
            let uniform_dist_threshold: f64 = rng.random();

            let sampled_token: i32 = fsample_top_p_from_logits
                .call_tuple((logits, &temperature, &top_p, &uniform_dist_threshold))
                .unwrap()
                .try_into()
                .unwrap();
            sampled_token as u32
        };

        let chat_template_j2 =
            std::fs::read_to_string(model_path.join("chat_template.j2")).unwrap();
        let chat_template = ChatTemplate::new("template".into(), chat_template_j2);

        let input = chat_template
            .apply("Explain about React's useState and useEffect.".into(), true)
            .unwrap();

        let input_tokens = tokenizer.encode(&input, true).unwrap();

        prefill(input_tokens.as_slice());

        let mut last_token = *input_tokens.last().unwrap();
        let mut stdout = std::io::stdout();

        let start = std::time::Instant::now();
        let mut tokens = 0;
        loop {
            let logits = decode(last_token);
            let new_token = sample(&logits, 0.6, 0.9);
            last_token = new_token;

            let decoded = tokenizer.decode(&[new_token], false).unwrap();
            print!("{}", decoded);
            stdout.flush().unwrap();
            tokens += 1;

            if decoded == "<|im_end|>" {
                break;
            }
        }
        println!("");
        let duration = start.elapsed().as_secs();
        println!("tps: {} tok/s", tokens / duration);
    }
}
