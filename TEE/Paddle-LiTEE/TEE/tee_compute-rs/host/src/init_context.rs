// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::invoke_ta_init_context;
use crate::pretty_env_logger;
use proto::model_config::*;
use proto::Command;
use std::env;
use std::fs::File;
use std::io::Read;
use std::process;

#[no_mangle]
//TODO: receive tee config from paddle-lite
//pub extern "C" fn init_tee_context(tee_context_buffer_ptr: *mut c_void, buffer_len: usize_t) {
pub extern "C" fn init_tee_context() -> u64 {
    match pretty_env_logger::try_init() {
        Ok(_) => (),
        Err(..) => {
            info!("CA: logger already been set");
        }
    };

    let work_path = env::var("CONFIG_DIR").expect("$CONFIG_DIR is not set");

    let mut config_file = File::open(work_path.to_owned() + "tee_config.json").unwrap();
    let mut pubkey_file = File::open(work_path.to_owned() + "tee_config.der").unwrap();
    let mut sig_file = File::open(work_path.to_owned() + "tee_config.sig").unwrap();

    let mut serialized_config: Vec<u8> = Vec::new();
    info!("CA: Read tee config from file...");
    config_file.read_to_end(&mut serialized_config);

    let mut public_key: Vec<u8> = Vec::new();
    info!("CA: Read public key from file...");
    pubkey_file.read_to_end(&mut public_key);

    let mut signature: Vec<u8> = Vec::new();
    info!("CA: Read signature from file...");
    sig_file.read_to_end(&mut signature);

    match invoke_ta_init_context(
        serialized_config.as_slice(),
        public_key.as_slice(),
        signature.as_slice(),
    ) {
        Ok(_) => {
            return 0;
        }
        Err(e) => {
            error!("invoke_ta_init_context error: {:?}", e);
            return 1;
        }
    };
}
