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

use crate::Command;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TEEModelConfig {
    pub config_id: Vec<u8>,
    pub command_list: Vec<TEECommandInfo>,
    pub encrypt_algorithm: EncAlgo,
    pub encrypt_info: Vec<EncryptInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TEECommandInfo {
    pub command: Command,
    pub input_tensor: Vec<TEEConfigTensor>,
    pub output_tensor: Vec<TEEConfigTensor>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TEEConfigTensor {
    pub name: String,
    pub action: TEEAction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TEEAction {
    Decrypt,
    ReadFromCache,
    WriteToCache,
    ReturnToCA,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EncAlgo {
    AesGcm128,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EncryptInfo {
    pub command_index: u32,
    pub tensor_name: String,
    pub iv: Vec<u8>,
    pub cmac: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct EncryptedTensor {
    pub name: String,
    pub key: Vec<u8>,
    pub iv: Vec<u8>,
    pub cmac: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TEEContext {
    pub model_config: TEEModelConfig,
    pub config_key: Vec<u8>,
    pub prev_index_in_list: Option<usize>,
}

// TEERunInfo: Run information for each command
#[derive(Serialize, Deserialize, Debug)]
pub struct TEERunInfo {
    pub command_info: TEECommandInfo,
    pub encrypt_algorithm: EncAlgo,
    pub encrypted_tensor: Vec<EncryptedTensor>,
}
