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
use crate::tee_tensor::*;
use serde::{Deserialize, Serialize};
use std::convert::From;
use std::ops::{Index, IndexMut};

#[derive(Serialize, Deserialize, Debug)]
pub struct TEESoftmaxParam {
    pub x: TEETensor,
    pub output: TEETensor,
    pub axis: __int32_t,
    pub use_cudnn: bool,
}
// trait for all op params
impl TEEParam for &mut TEESoftmaxParam {}

impl Index<String> for TEESoftmaxParam {
    type Output = TEETensor;
    fn index(&self, s: String) -> &TEETensor {
        match s.as_str() {
            "x" => &self.x,
            "output" => &self.output,
            _ => panic!("unknown field: {}", s),
        }
    }
}
impl IndexMut<String> for TEESoftmaxParam {
    fn index_mut(&mut self, s: String) -> &mut TEETensor {
        match s.as_str() {
            "x" => &mut self.x,
            "output" => &mut self.output,
            _ => panic!("unknown field: {}", s),
        }
    }
}

// PT_SoftmaxParam to TEESoftmaxParam
impl From<PT_SoftmaxParam> for TEESoftmaxParam {
    fn from(pt_softmax_param: PT_SoftmaxParam) -> Self {
        let x: TEETensor = TEETensor::from(pt_softmax_param.x);
        let output: TEETensor = TEETensor::from(pt_softmax_param.output);

        Self {
            x: x,
            output: output,
            axis: pt_softmax_param.axis,
            use_cudnn: pt_softmax_param.use_cudnn,
        }
    }
}

// raw pointer to TEESoftmaxParam
impl From<*mut ::std::os::raw::c_void> for TEESoftmaxParam {
    fn from(param_ptr: *mut ::std::os::raw::c_void) -> Self {
        let pt_softmax_param_ptr = param_ptr as *mut PT_SoftmaxParam;
        let pt_softmax_param = unsafe { &mut *pt_softmax_param_ptr };

        let x: TEETensor = TEETensor::from(pt_softmax_param.x);
        let output: TEETensor = TEETensor::from(pt_softmax_param.output);

        Self {
            x: x,
            output: output,
            axis: pt_softmax_param.axis,
            use_cudnn: pt_softmax_param.use_cudnn,
        }
    }
}

// TEESoftmaxParam to PT_SoftmaxParam
impl From<&TEESoftmaxParam> for PT_SoftmaxParam {
    fn from(tee_param: &TEESoftmaxParam) -> Self {
        Self {
            x: PortableTensor::from(&tee_param.x),
            output: PortableTensor::from(&tee_param.output),
            axis: tee_param.axis,
            use_cudnn: tee_param.use_cudnn,
        }
    }
}
