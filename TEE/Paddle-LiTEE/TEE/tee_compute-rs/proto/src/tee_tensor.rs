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

use crate::c_bindings::*;
use serde::{Deserialize, Serialize};
use std::convert::From;
use std::ops::{Index, IndexMut};

#[derive(Serialize, Deserialize, Debug)]
pub struct TEETensor {
    pub data: Vec<u8>,
    pub data_type: PT_DataType,
    pub shape: Vec<u64>,
    pub byte_offset: size_t,
}

// PortableTensor to TEETensor
impl From<PortableTensor> for TEETensor {
    fn from(portable_tensor: PortableTensor) -> Self {
        let data_size: usize = portable_tensor.byte_size as usize;
        let mut data: Vec<u8> = unsafe {
            std::slice::from_raw_parts(portable_tensor.bytes as *mut u8, data_size).to_vec()
        };
        let shape: Vec<u64> = unsafe {
            std::slice::from_raw_parts(
                portable_tensor.dims as *mut u64,
                portable_tensor.dim_size as usize,
            )
            .to_vec()
        };

        Self {
            data: data,
            data_type: portable_tensor.data_type,
            shape: shape,
            byte_offset: portable_tensor.byte_offset,
        }
    }
}

// TEETensor to PortableTensor
impl From<&TEETensor> for PortableTensor {
    fn from(tee_tensor: &TEETensor) -> Self {
        Self {
            bytes: tee_tensor.data.as_ptr() as *mut ::std::os::raw::c_void,
            byte_size: tee_tensor.data.len() as size_t,
            dims: tee_tensor.shape.as_ptr() as *mut u64,
            dim_size: tee_tensor.shape.len() as size_t,
            byte_offset: tee_tensor.byte_offset,
            data_type: tee_tensor.data_type,
        }
    }
}

pub trait TEEParam {}
