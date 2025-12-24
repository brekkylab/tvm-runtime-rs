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

fn main() {
    let cmake_source_path = PathBuf::new().join("..").join("3rdparty").join("tvm");
    let mut cfg = cmake::Config::new(cmake_source_path);

    // Common flags
    cfg.very_verbose(true);
    cfg.define("INSTALL_DEV", "OFF");
    cfg.define("BUILD_DUMMY_LIBTVM", "ON");

    // Profile-based flags
    let profile = env::var("PROFILE").unwrap();
    if profile == "debug" {
        cfg.define("CMAKE_BUILD_TYPE", "Debug");
        cfg.define("USE_LIBBACKTRACE", "ON");
    } else {
        cfg.define("CMAKE_BUILD_TYPE", "Release");
        cfg.define("USE_LIBBACKTRACE", "OFF");
    }

    // Feature-based flags
    if cfg!(feature = "cuda") {
        cfg.define("USE_CUDA", "ON");
    }

    if cfg!(feature = "metal") {
        cfg.define("USE_METAL", "ON");
    }

    if cfg!(feature = "vulkan") {
        cfg.define("USE_VULKAN", "ON");
    }

    let lib_dir = cfg.build().join("lib");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=tvm_runtime");

    // Set absolute path IDs for macOS and Linux to make linking easier
    #[cfg(target_os = "macos")]
    {
        let lib_path = lib_dir.join("libtvm_runtime.dylib");
        if lib_path.exists() {
            let _ = std::process::Command::new("install_name_tool")
                .arg("-id")
                .arg(&lib_path)
                .arg(&lib_path)
                .status();
        }
    }
    #[cfg(target_os = "linux")]
    {
        let lib_path = lib_dir.join("libtvm_runtime.so");
        if lib_path.exists() {
            let _ = std::process::Command::new("patchelf")
                .arg("--set-soname")
                .arg(&lib_path)
                .arg(&lib_path)
                .status();
        }
    }
}
