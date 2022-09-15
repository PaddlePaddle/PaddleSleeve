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
pub struct TEEFcParam {
    pub input: TEETensor,
    pub w: TEETensor,
    pub bias: TEETensor,
    pub output: TEETensor,
    pub scale: TEETensor,
    pub input_scale: f32,
    pub output_scale: f32,
    pub flag_act: bool,
    pub flag_trans_weights: bool,
}

// trait for all op params
impl TEEParam for &mut TEEFcParam {}

impl Index<String> for TEEFcParam {
    type Output = TEETensor;
    fn index(&self, s: String) -> &TEETensor {
        match s.as_str() {
            "input" => &self.input,
            "w" => &self.w,
            "bias" => &self.bias,
            "output" => &self.output,
            _ => panic!("unknown field: {}", s),
        }
    }
}

impl IndexMut<String> for TEEFcParam {
    fn index_mut(&mut self, s: String) -> &mut TEETensor {
        match s.as_str() {
            "input" => &mut self.input,
            "w" => &mut self.w,
            "bias" => &mut self.bias,
            "output" => &mut self.output,
            _ => panic!("unknown field: {}", s),
        }
    }
}

// PT_FcParam to TEEFcParam
impl From<PT_FcParam> for TEEFcParam {
    fn from(pt_fc_param: PT_FcParam) -> Self {
        let input: TEETensor = TEETensor::from(pt_fc_param.input);
        let w: TEETensor = TEETensor::from(pt_fc_param.w);
        let bias: TEETensor = TEETensor::from(pt_fc_param.bias);
        let output: TEETensor = TEETensor::from(pt_fc_param.output);
        let scale: TEETensor = TEETensor::from(pt_fc_param.scale);

        Self {
            input: input,
            w: w,
            bias: bias,
            output: output,
            scale: scale,
            input_scale: pt_fc_param.input_scale,
            output_scale: pt_fc_param.output_scale,
            flag_act: pt_fc_param.flag_act,
            flag_trans_weights: pt_fc_param.flag_trans_weights,
        }
    }
}

// raw pointer to TEEFcParam
impl From<*mut ::std::os::raw::c_void> for TEEFcParam {
    fn from(param_ptr: *mut ::std::os::raw::c_void) -> Self {
        let pt_fc_param_ptr = param_ptr as *mut PT_FcParam;
        let pt_fc_param = unsafe { &mut *pt_fc_param_ptr };
        let input: TEETensor = TEETensor::from(pt_fc_param.input);
        let w: TEETensor = TEETensor::from(pt_fc_param.w);
        let bias: TEETensor = TEETensor::from(pt_fc_param.bias);
        let output: TEETensor = TEETensor::from(pt_fc_param.output);
        let scale: TEETensor = TEETensor::from(pt_fc_param.scale);

        Self {
            input: input,
            w: w,
            bias: bias,
            output: output,
            scale: scale,
            input_scale: pt_fc_param.input_scale,
            output_scale: pt_fc_param.output_scale,
            flag_act: pt_fc_param.flag_act,
            flag_trans_weights: pt_fc_param.flag_trans_weights,
        }
    }
}

// TEEFcParam to PT_FcParam
impl From<&TEEFcParam> for PT_FcParam {
    fn from(tee_param: &TEEFcParam) -> Self {
        Self {
            input: PortableTensor::from(&tee_param.input),
            w: PortableTensor::from(&tee_param.w),
            bias: PortableTensor::from(&tee_param.bias),
            output: PortableTensor::from(&tee_param.output),
            scale: PortableTensor::from(&tee_param.scale),
            input_scale: tee_param.input_scale,
            output_scale: tee_param.output_scale,
            flag_act: tee_param.flag_act,
            flag_trans_weights: tee_param.flag_trans_weights,
        }
    }
}
