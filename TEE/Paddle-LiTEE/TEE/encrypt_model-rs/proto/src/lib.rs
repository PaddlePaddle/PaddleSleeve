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

pub mod model_config;

use serde::{Deserialize, Serialize};
pub use serde_json;

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
pub enum Command {
    InitContext,
    Softmax,
    FcRun,
    Encrypt,
    Decrypt,
    Derive,
    Unknown,
}

impl From<u32> for Command {
    #[inline]
    fn from(value: u32) -> Command {
        match value {
            0 => Command::InitContext,
            1 => Command::Softmax,
            2 => Command::FcRun,
            3 => Command::Encrypt,
            4 => Command::Decrypt,
            5 => Command::Derive,
            _ => Command::Unknown,
        }
    }
}

impl Default for Command {
    fn default() -> Self {
        Command::Unknown
    }
}

pub enum Mode {
    Decode,
    Encode,
    Unknown,
}

impl From<u32> for Mode {
    #[inline]
    fn from(value: u32) -> Mode {
        match value {
            0 => Mode::Decode,
            1 => Mode::Encode,
            _ => Mode::Unknown,
        }
    }
}

pub enum KeySize {
    Bit128 = 16,
    Bit256 = 32,
    Unknown = 0,
}

impl From<u32> for KeySize {
    #[inline]
    fn from(value: u32) -> KeySize {
        match value {
            16 => KeySize::Bit128,
            32 => KeySize::Bit256,
            _ => KeySize::Unknown,
        }
    }
}

pub const UUID: &str = &include_str!(concat!(env!("OUT_DIR"), "/uuid.txt"));
