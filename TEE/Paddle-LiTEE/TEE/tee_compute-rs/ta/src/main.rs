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

#![no_main]

use anyhow::{anyhow, Result};
use optee_utee::{
    ta_close_session, ta_create, ta_destroy, ta_invoke_command, ta_open_session, trace_println,
};
use optee_utee::{trace, Time};
use optee_utee::{Error, ErrorKind, Parameters};
use proto::c_bindings::{PT_DataType, PT_DataType_kPTFloat, PT_DataType_kPTInt8};
use proto::fc_param::*;
use proto::model_config::*;
use proto::softmax_param::*;
use proto::Command;
use ring::pbkdf2::*;
use ring::signature;
use std::ops::{Index, IndexMut};
use std::ptr;

mod secure_storage;
use secure_storage::*;
mod crypto_aead;
use crypto_aead::*;
#[macro_use]
mod macros;
use macros::*;

#[link(name = "math_backend", kind = "static")]
extern "C" {
    fn softmax_run(din: *const f32, dout: *mut f32, outer_num: i32, inner_num: i32, axis_size: i32);
    fn transpose_float(dout: *mut f32, m: i32, n: i32);
    fn transpose_int(dout: *mut i8, m: i32, n: i32);
    fn fc_run_ff(
        M: i32,
        N: i32,
        K: i32,
        A: *const f32,
        B: *const f32,
        C: *mut f32,
        bias: *const f32,
        flag_act: bool,
        flag_gemm: bool,
    );
    fn fc_run_if(
        M: i32,
        N: i32,
        K: i32,
        A: *const i8,
        B: *const i8,
        C: *mut f32,
        bias: *const f32,
        scale: *const f32,
        flag_act: bool,
        flag_gemm: bool,
    );
    fn fc_run_ii(
        M: i32,
        N: i32,
        K: i32,
        A: *const i8,
        B: *const i8,
        C: *mut i8,
        bias: *const f32,
        scale: *const f32,
        flag_act: bool,
        flag_gemm: bool,
    );
}

static ROOT_KEY: [u8; 16] = [
    0x56, 0xa9, 0xbd, 0xc3, 0xa8, 0x93, 0x78, 0x41, 0xbc, 0xa6, 0xd5, 0x91, 0x83, 0xb7, 0xe0, 0x49,
];

#[ta_create]
fn create() -> optee_utee::Result<()> {
    trace_println!("[+] TA create");
    Ok(())
}

#[ta_open_session]
fn open_session(_params: &mut Parameters) -> optee_utee::Result<()> {
    trace_println!("[+] TA open session");
    Ok(())
}

#[ta_close_session]
fn close_session() {
    trace_println!("[+] TA close session");
}

#[ta_destroy]
fn destroy() {
    trace_println!("[+] TA destroy");
}

#[ta_invoke_command]
fn invoke_command(cmd_id: u32, params: &mut Parameters) -> optee_utee::Result<()> {
    trace_println!("[+] TA invoke command");
    match Command::from(cmd_id) {
        Command::InitContext => match init_context(params) {
            Ok(_) => {
                return Ok(());
            }
            Err(e) => {
                trace_println!("[+] Error: {}", e);
                return Err(Error::new(ErrorKind::BadState));
            }
        },
        Command::Softmax => {
            let run_info: TEERunInfo = match check_tee_context(Command::Softmax) {
                Ok(info) => info,
                Err(e) => {
                    trace_println!("[+] Error: {}", e);
                    return Err(Error::new(ErrorKind::BadState));
                }
            };
            match softmax_compute(run_info, params) {
                Ok(_) => {
                    return Ok(());
                }
                Err(e) => {
                    trace_println!("[+] Error: {}", e);
                    return Err(Error::new(ErrorKind::BadState));
                }
            }
        }
        Command::FcRun => {
            let run_info: TEERunInfo = match check_tee_context(Command::FcRun) {
                Ok(info) => info,
                Err(e) => {
                    trace_println!("[+] Error: {}", e);
                    return Err(Error::new(ErrorKind::BadState));
                }
            };
            match fc_run(run_info, params) {
                Ok(_) => {
                    return Ok(());
                }
                Err(e) => {
                    trace_println!("[+] Error: {}", e);
                    return Err(Error::new(ErrorKind::BadState));
                }
            }
        }
        _ => Err(Error::new(ErrorKind::BadParameters)),
    }
}

pub fn init_context(params: &mut Parameters) -> Result<()> {
    let mut time = Time::new();
    time.system_time();
    trace_println!("[*] Benchmark: invoke init_context: {}", time);
    trace_println!("[+] TA invoke init_context");
    let mut p0 = unsafe { params.0.as_memref().unwrap() };
    let mut p1 = unsafe { params.1.as_memref().unwrap() };
    let mut p2 = unsafe { params.2.as_memref().unwrap() };

    let serialized_config = p0.buffer();
    let mut deserialized_model_config: TEEModelConfig = serde_json::from_slice(&serialized_config)?;
    //trace_println!("[+] deserialized model config: {:?}", deserialized_model_config);
    time.system_time();
    trace_println!("[*] Benchmark: after deserialized_model_config: {}", time);

    trace_println!("[+] Verifying signature...");
    let public_key_bytes = p1.buffer();
    let signature = p2.buffer();
    let public_key =
        signature::UnparsedPublicKey::new(&signature::RSA_PKCS1_2048_8192_SHA256, public_key_bytes);
    public_key
        .verify(serialized_config, &signature)
        .map_err(|e| anyhow!("verify signature failed: {:?}", e))?;
    trace_println!("[+] Signature verify ok");
    time.system_time();
    trace_println!("[*] Benchmark: after signature verification: {}", time);

    trace_println!("[+] Initialize TEEContext...");
    //derive config key
    let iter = std::num::NonZeroU32::new(100).unwrap();
    let mut config_key = [0x00u8; 16];
    derive(
        PBKDF2_HMAC_SHA256,
        iter,
        &deserialized_model_config.config_id,
        &ROOT_KEY,
        &mut config_key,
    );
    trace_println!("[+] Derive config key: {:?}", &config_key);
    time.system_time();
    trace_println!("[*] Benchmark: after derive config key: {}", time);

    //initialize context
    let context = TEEContext {
        model_config: deserialized_model_config,
        config_key: config_key.to_vec(),
        prev_index_in_list: None,
    };
    //trace_println!("[+] TEEContext: {:?}", &context);
    //save context in secure storage
    let serialized_context: Vec<u8> = bincode::serialize(&context)?;
    let mut obj_id = b"serialized_context".to_vec();
    save_in_secure_storage(&mut obj_id, &serialized_context)?;
    time.system_time();
    trace_println!("[*] Benchmark: after save_in_secure_storage: {}", time);

    trace_println!("[+] TEEContext is saved in secure storage");
    Ok(())
}

fn check_tee_context(current_command: Command) -> Result<TEERunInfo> {
    let mut time = Time::new();
    time.system_time();
    trace_println!("[*] Benchmark: invoke check_tee_context: {}", time);

    trace_println!("--------------------------------------------------------");
    trace_println!("[+] invoke check_tee_context");
    // read tee context from secure storage
    let mut obj_id = b"serialized_context".to_vec();
    let serialized_context = load_from_secure_storage(&mut obj_id)?;
    let mut context: TEEContext = bincode::deserialize(&serialized_context)?;
    //trace_println!("[+] TEEContext: {:?}", &context);
    trace_println!("[+] TEEContext is loaded");
    let mut current_command_index = 0usize;

    trace_println!("[+] Check op sequence...op: {:?}", &current_command);
    // trace_println!("[+] config : {:?}", context);

    match context.prev_index_in_list {
        None => {
            if context.model_config.command_list[0].command == current_command {
                context.prev_index_in_list = Some(0usize);
            } else {
                return Err(anyhow!("Bad op sequence"));
            }
        }
        Some(idx) => {
            if context.model_config.command_list[idx + 1].command == current_command {
                current_command_index = (idx + 1) as usize;
                context.prev_index_in_list = Some(idx + 1);
            } else {
                return Err(anyhow!("Bad op sequence"));
            }
        }
    };

    trace_println!("[+] Check passed");

    time.system_time();
    trace_println!("[*] Benchmark: after check tee context: {}", time);

    let serialized_context: Vec<u8> = bincode::serialize(&context)?;
    let mut obj_id = b"serialized_context".to_vec();
    save_in_secure_storage(&mut obj_id, &serialized_context)?;

    time.system_time();
    trace_println!("[*] Benchmark: after save_in_secure_storage: {}", time);

    let mut encrypted_tensor = Vec::new();
    let enc_info_of_this_index: Vec<EncryptInfo> = context
        .model_config
        .encrypt_info
        .into_iter()
        .filter(|v| v.command_index == (current_command_index as u32))
        .collect();

    for each_enc_info in enc_info_of_this_index {
        let iter = std::num::NonZeroU32::new(100).unwrap();
        let mut tensor_key = [0x00u8; 16];
        let mut tensor_id: String = current_command_index.to_string();
        tensor_id.push_str(&each_enc_info.tensor_name);

        derive(
            PBKDF2_HMAC_SHA256,
            iter,
            &tensor_id.as_bytes(),
            &context.config_key,
            &mut tensor_key,
        );
        trace_println!(
            "[+] Derive key for tensor: {:?}, key: {:?}",
            &each_enc_info.tensor_name,
            &tensor_key
        );

        let new_encrypted_tensor = EncryptedTensor {
            name: each_enc_info.tensor_name,
            key: tensor_key.to_vec(),
            iv: each_enc_info.iv,
            cmac: each_enc_info.cmac,
        };
        encrypted_tensor.push(new_encrypted_tensor);
    }

    time.system_time();
    trace_println!("[*] Benchmark: after derive tensor key: {}", time);

    let run_info = TEERunInfo {
        command_info: context.model_config.command_list[current_command_index].clone(),
        encrypt_algorithm: context.model_config.encrypt_algorithm,
        encrypted_tensor: encrypted_tensor,
    };
    //trace_println!("[+] Generate TEERunInfo: {:?}", &run_info);
    trace_println!("--------------------------------------------------------");

    Ok(run_info)
}

pub fn softmax_compute(run_info: TEERunInfo, params: &mut Parameters) -> Result<()> {
    let mut time = Time::new();
    time.system_time();
    trace_println!("[*] Benchmark: invoke softmax_compute: {}", time);

    trace_println!("[+] TA invoke softmax_compute");
    let mut p0 = unsafe { params.0.as_memref().unwrap() };
    let mut p1 = unsafe { params.1.as_memref().unwrap() };
    let input = p0.buffer();
    let mut softmax_param: TEESoftmaxParam = bincode::deserialize(&input)?;
    //trace_println!("TEESoftmaxParam.x.data {:?}", softmax_param.x.data);

    time.system_time();
    trace_println!("[*] Benchmark: after deserialize tee param: {}", time);

    trace_println!("----------------------------");
    handle_input_tensor!(softmax_param, run_info, time);

    time.system_time();
    trace_println!("[*] Benchmark: after handle_input_tensor: {}", time);

    trace_println!("----------------------------");
    let din = softmax_param.x.data.as_ptr() as *const f32;
    let dout = softmax_param.output.data.as_ptr() as *mut f32;

    let mut outer_num: i32 = 1;
    for each_dim in 0..softmax_param.axis as usize {
        outer_num *= softmax_param.output.shape[each_dim] as i32;
    }
    //trace_println!("[+] outer_num: {}", &outer_num);

    let mut inner_num: i32 = 1;
    for each_dim in (softmax_param.axis as usize + 1)..softmax_param.x.shape.len() as usize {
        inner_num *= softmax_param.x.shape[each_dim] as i32;
    }
    trace_println!("[+] inner_num: {}", &inner_num);

    let axis_size = softmax_param.x.shape[softmax_param.axis as usize] as i32;
    trace_println!("[+] axis:{}", &axis_size);
    unsafe {
        softmax_run(din, dout, outer_num, inner_num, axis_size);
    }

    time.system_time();
    trace_println!("[*] Benchmark: after softmax_inner1_large_axis: {}", time);

    trace_println!("----------------------------");
    handle_output_tensor!(softmax_param, run_info);

    time.system_time();
    trace_println!("[*] Benchmark: after handle_output_tensor: {}", time);

    let serialized_output: Vec<u8> = bincode::serialize(&softmax_param)?;
    //trace_println!("[+] serialized_output_len: {:?}", serialized_output.len());
    p1.buffer().clone_from_slice(&serialized_output);
    trace_println!("[+] write to CA output buffer finished");

    time.system_time();
    trace_println!("[*] Benchmark: after return: {}", time);

    Ok(())
}

pub fn fc_run(run_info: TEERunInfo, params: &mut Parameters) -> Result<()> {
    let mut time = Time::new();
    time.system_time();
    trace_println!("[*] Benchmark: invoke fc_run: {}", time);

    trace_println!("[+] TA invoke fc_run");
    let mut p0 = unsafe { params.0.as_memref().unwrap() };
    let mut p1 = unsafe { params.1.as_memref().unwrap() };

    let input = p0.buffer();
    time.system_time();
    trace_println!("[*] Benchmark: after get input buffer: {}", time);
    trace_println!("[*] input:  {}", input.len());

    let mut fc_param: TEEFcParam = bincode::deserialize(&input)?;
    time.system_time();
    trace_println!("[*] Benchmark: after deserialize tee param: {}", time);

    // handle input tensor according to tee config
    handle_input_tensor!(fc_param, run_info, time);
    trace_println!("----------------------------");

    time.system_time();
    trace_println!("[*] Benchmark: after handle_input_tensor: {}", time);

    // transpose
    let M = fc_param.input.shape[0] as i32;
    let N = fc_param.w.shape[1] as i32;
    let K = fc_param.input.shape[1] as i32;

    trace_println!("[+] fc compute");
    trace_println!("[+] M:{}, N:{}, K:{}", M, N, K);

    let input_type = fc_param.input.data_type;
    let output_type = fc_param.output.data_type;

    if fc_param.flag_trans_weights {
        trace_println!("[+] transpose");
        match (input_type) {
            PT_DataType_kPTFloat => unsafe {
                transpose_float(
                    fc_param.w.data.as_ptr() as *mut f32,
                    fc_param.w.shape[0] as i32,
                    fc_param.w.shape[1] as i32,
                );
            },
            PT_DataType_kPTInt8 => unsafe {
                transpose_int(
                    fc_param.w.data.as_ptr() as *mut i8,
                    fc_param.w.shape[0] as i32,
                    fc_param.w.shape[1] as i32,
                );
            },
            _ => {
                return Err(anyhow!("Unknow fc compute type"));
            }
        }
    }

    // trace_println!("{:?}", fc_param);
    trace_println!("[+] M:{}, N:{}, K:{}", M, N, K);

    match (input_type, output_type) {
        (PT_DataType_kPTFloat, PT_DataType_kPTFloat) => {
            let flag_gemm: bool = M > 1;
            trace_println!("[+] fc compute<float, float> invoked");
            unsafe {
                fc_run_ff(
                    M,
                    N,
                    K,
                    fc_param.input.data.as_ptr() as *const f32,
                    fc_param.w.data.as_ptr() as *const f32,
                    fc_param.output.data.as_ptr() as *mut f32,
                    if fc_param.bias.data.is_empty() {
                        std::ptr::null::<f32>() as *const f32
                    } else {
                        fc_param.bias.data.as_ptr() as *const f32
                    },
                    fc_param.flag_act,
                    flag_gemm,
                );
            }
        }
        (PT_DataType_kPTInt8, PT_DataType_kPTFloat) => {
            let flag_gemm: bool = M > 1 && fc_param.scale.data.len() == 1;
            trace_println!("[+] fc compute<int8, float> invoked");
            trace_println!("[+] trans weight scale");
            let extend_size = if flag_gemm { M as usize } else { N as usize };
            let in_scale_len = fc_param.scale.data.len();
            let in_scale = unsafe {
                std::slice::from_raw_parts(
                    fc_param.scale.data.as_ptr() as *mut f32,
                    in_scale_len / 4,
                )
                .to_vec()
            };
            let mut scale: Vec<f32> = Vec::new();
            for i in 0..extend_size {
                if flag_gemm {
                    scale.push(in_scale[0] * fc_param.input_scale);
                } else {
                    scale.push(in_scale[i] * fc_param.input_scale);
                }
            }
            trace_println!("[+] fc compute<int8, float> invoked");
            unsafe {
                fc_run_if(
                    M,
                    N,
                    K,
                    fc_param.input.data.as_ptr() as *const i8,
                    fc_param.w.data.as_ptr() as *const i8,
                    fc_param.output.data.as_ptr() as *mut f32,
                    if fc_param.bias.data.is_empty() {
                        std::ptr::null::<f32>() as *const f32
                    } else {
                        fc_param.bias.data.as_ptr() as *const f32
                    },
                    scale.as_ptr() as *const f32,
                    fc_param.flag_act,
                    flag_gemm,
                );
            }
        }
        (PT_DataType_kPTInt8, PT_DataType_kPTInt8) => {
            let flag_gemm: bool =
                M > 1 && fc_param.scale.data.len() == 1 && fc_param.bias.data.is_empty();
            let flag_trans_bias: bool = !fc_param.bias.data.is_empty();
            trace_println!("[+] fc compute<int8, int8> invoked");
            trace_println!("[+] trans weight scale");
            let extend_size = if flag_gemm { M as usize } else { N as usize };
            let in_scale_len = fc_param.scale.data.len();
            let in_scale = unsafe {
                std::slice::from_raw_parts(
                    fc_param.scale.data.as_ptr() as *mut f32,
                    in_scale_len / 4,
                )
                .to_vec()
            };
            let mut scale: Vec<f32> = Vec::new();
            for i in 0..extend_size {
                if flag_gemm {
                    scale.push(in_scale[0] * fc_param.input_scale / fc_param.output_scale);
                } else {
                    scale.push(in_scale[i] * fc_param.input_scale / fc_param.output_scale);
                }
            }
            let mut bias: Vec<f32> = Vec::new();
            let bias_len = fc_param.bias.data.len();
            if (flag_trans_bias) {
                trace_println!("[+] trans bias, bias len:{}", bias_len);

                let in_bias = unsafe {
                    std::slice::from_raw_parts(
                        fc_param.bias.data.as_ptr() as *mut f32,
                        bias_len / 4,
                    )
                    .to_vec()
                };
                for i in 0..in_bias.len() {
                    bias.push(in_bias[i] / fc_param.output_scale);
                }
            }
            trace_println!("[+] fc compute<int8, int8> invoked");
            unsafe {
                fc_run_ii(
                    M,
                    N,
                    K,
                    fc_param.input.data.as_ptr() as *const i8,
                    fc_param.w.data.as_ptr() as *const i8,
                    fc_param.output.data.as_ptr() as *mut i8,
                    if fc_param.bias.data.is_empty() {
                        std::ptr::null::<f32>() as *const f32
                    } else {
                        bias.as_ptr() as *const f32
                    },
                    scale.as_ptr() as *const f32,
                    fc_param.flag_act,
                    flag_gemm,
                );
            }
        }
        _ => {
            return Err(anyhow!("Unknow fc compute type"));
        }
    }

    time.system_time();
    trace_println!("[*] Benchmark: after fc compute: {}", time);

    trace_println!("----------------------------");
    handle_output_tensor!(fc_param, run_info);

    time.system_time();
    trace_println!("[*] Benchmark: after handle_output_tensor: {}", time);

    let serialized_output: Vec<u8> = bincode::serialize(&fc_param)?;
    //trace_println!("[+] serialized_output_len: {:?}", serialized_output.len());
    p1.buffer().clone_from_slice(&serialized_output);
    trace_println!("[+] write to CA output buffer finished");

    time.system_time();
    trace_println!("[*] Benchmark: after return: {}", time);

    Ok(())
}

// TA configurations
const TA_FLAGS: u32 = 0;
const TA_DATA_SIZE: u32 = 10 * 1024 * 1024;
const TA_STACK_SIZE: u32 = 10 * 1024;
const TA_VERSION: &[u8] = b"0.1\0";
const TA_DESCRIPTION: &[u8] = b"This is a softmax compute.\0";
const EXT_PROP_VALUE_1: &[u8] = b"Softmax TA\0";
const EXT_PROP_VALUE_2: u32 = 0x0010;
const TRACE_LEVEL: i32 = 4;
const TRACE_EXT_PREFIX: &[u8] = b"TA\0";
const TA_FRAMEWORK_STACK_SIZE: u32 = 2048;

include!(concat!(env!("OUT_DIR"), "/user_ta_header.rs"));
