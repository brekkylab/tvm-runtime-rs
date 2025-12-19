/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
use std::env;
use std::path::PathBuf;

fn option_value(key: &str) -> &str {
    if env::var(key).is_ok() {
        "ON"
    } else {
        "OFF"
    }
}

fn main() {
    let cmake_source_path = PathBuf::new().join("..").join("3rdparty").join("tvm");
    let mut cfg = cmake::Config::new(cmake_source_path);
    cfg.define("CMAKE_BUILD_TYPE", "Release")
        .define("INSTALL_DEV", "OFF")
        .define("BUILD_DUMMY_LIBTVM", "ON")
        .define("USE_LIBBACTRACE", "OFF")
        .define("TVM_FFI_USE_LIBBACKTRACE", "OFF")
        .define(
            "USE_METAL",
            std::env::var("USE_METAL").unwrap_or_else(|_| "ON".to_string()),
        )
        .define(
            "USE_VULKAN",
            std::env::var("USE_VULKAN").unwrap_or_else(|_| "OFF".to_string()),
        )
        .very_verbose(true);
    let lib_dir = cfg.build().join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=tvm_runtime");
}
