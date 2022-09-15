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
#[macro_use]
extern crate log;
extern crate pretty_env_logger;
#[macro_use]
extern crate lazy_static;
extern crate libc;
use std::sync::Mutex;

use std::fs::File;
use std::io::prelude::*;
use std::process;

use libc::c_char;
use rand::RngCore;

use optee_teec::{Context, Operation, ParamNone, ParamTmpRef, Session, Uuid};

use proto;
use proto::model_config::*;
use proto::{Command, UUID};

lazy_static! {
    static ref TEE_CONFIG: Mutex<TEEModelConfig> = Mutex::new(TEEModelConfig::default());
}

fn cipher_buffer(
    session: &mut Session,
    cmd: Command,
    in_out: &mut [u8],
    iv: &[u8],
    cmac: &mut [u8],
) -> optee_teec::Result<()> {
    let p0 = ParamTmpRef::new_output(in_out);
    let p1 = ParamTmpRef::new_input(iv);
    let p2 = ParamTmpRef::new_output(cmac);
    let mut operation = Operation::new(0, p0, p1, p2, ParamNone);
    session.invoke_command(cmd as u32, &mut operation)?;

    Ok(())
}

pub fn convert_name<'a>(op_name: &'a str, tensor_name: &'a str) -> &'a str {
    match op_name {
        "softmax" => {
            if tensor_name.contains("input") {
                return "x";
            }
            if tensor_name.contains("output") {
                return "output";
            } else {
                return "";
            }
        }
        "fc" => {
            if tensor_name.contains("input") {
                return "input";
            }
            if tensor_name.contains("weights") {
                return "w";
            }
            if tensor_name.contains("offset") {
                return "bias";
            }
            if tensor_name.contains("output") {
                return "output";
            } else {
                return "";
            }
        }
        _ => "",
    }
}

#[no_mangle]
pub extern "C" fn generate_signed_config(json_str: *const c_char, len: i32) {
    pretty_env_logger::init();
    // generate tee config and sign
    let js_bytes = unsafe { std::slice::from_raw_parts(json_str, len as usize) };
    let user_config: UserConfig = proto::serde_json::from_slice(js_bytes).unwrap();

    let mut tee_config = TEE_CONFIG.lock().unwrap();
    let mut config_id = [0x00u8; 8];
    rand::thread_rng().fill_bytes(&mut config_id);
    tee_config.config_id = config_id.to_vec();

    for i in 0..user_config.op_list.len() {
        let mut pre_cont: bool = false; // pre continuous
        let mut next_cont: bool = false; // next continuous
        if i != 0 {
            // non-first op
            if user_config.op_list[i].index - 1 == user_config.op_list[i - 1].index {
                // op index-1 = pre op index
                pre_cont = true;
            }
        }
        if i != user_config.op_list.len() - 1 {
            // non-last op
            if user_config.op_list[i].index + 1 == user_config.op_list[i + 1].index {
                // op index+1 = next op index
                next_cont = true;
            }
        }

        let mut tee_cmd_info = TEECommandInfo::default();
        tee_cmd_info.command = get_command(&user_config.op_list[i].name);
        get_input_tensor(
            &mut tee_cmd_info.input_tensor,
            &user_config.op_list[i].name,
            &user_config.op_list[i].protected_param,
            pre_cont,
        );
        get_output_tensor(
            &mut tee_cmd_info.output_tensor,
            &user_config.op_list[i].name,
            next_cont,
        );
        tee_config.command_list.push(tee_cmd_info);

        if !user_config.op_list[i].protected_param.is_empty() {
            get_iv(
                &mut tee_config.encrypt_info,
                i as u32,
                &user_config.op_list[i].name,
                &user_config.op_list[i].protected_param,
            );
        }
    }
    tee_config.encrypt_algorithm = EncAlgo::AesGcm128;

    //let tee_config_bytes = proto::serde_json::to_vec::<TEEModelConfig>(&tee_config).unwrap();
    info!("{:?}", tee_config);
}

pub fn get_command(name: &str) -> Command {
    match name {
        "softmax" => Command::Softmax,
        "fc" => Command::FcRun,
        _ => Command::Unknown,
    }
}

pub fn get_input_tensor(
    input_tensor: &mut Vec<TEEConfigTensor>,
    name: &str,
    protected_param: &Vec<String>,
    pre_cont: bool,
) {
    if pre_cont {
        let iname = convert_name(name, "input");
        let input = TEEConfigTensor {
            name: String::from(iname),
            action: TEEAction::ReadFromCache,
        };
        input_tensor.push(input);
    }

    for pp in protected_param {
        let iname = convert_name(name, pp);
        let pp_tensor = TEEConfigTensor {
            name: String::from(iname),
            action: TEEAction::Decrypt,
        };
        input_tensor.push(pp_tensor);
    }
}

pub fn get_output_tensor(output_tensor: &mut Vec<TEEConfigTensor>, name: &str, next_cont: bool) {
    let oname = convert_name(name, "output");
    let mut output = TEEConfigTensor {
        name: String::from(oname),
        action: TEEAction::ReturnToCA,
    };

    if next_cont {
        output.action = TEEAction::WriteToCache;
    }
    output_tensor.push(output);
}

pub fn get_iv(
    encrypt_info: &mut Vec<EncryptInfo>,
    cmd_idx: u32,
    name: &str,
    protected_param: &Vec<String>,
) {
    for pp in protected_param {
        let mut rand_iv = [0x00u8; 12];
        rand::thread_rng().fill_bytes(&mut rand_iv);
        let tname = convert_name(name, pp);
        let ei = EncryptInfo {
            command_index: cmd_idx,
            tensor_name: String::from(tname),
            iv: rand_iv.to_vec(),
            cmac: vec![0u8; 16],
        };
        encrypt_info.push(ei);
    }
}

// derive_key inovked by encrypt_param
pub fn derive_key(
    session: &mut Session,
    config_id: &[u8],
    tensor_id: &[u8],
) -> optee_teec::Result<()> {
    let p0 = ParamTmpRef::new_input(config_id);
    let p1 = ParamTmpRef::new_input(tensor_id);
    let mut operation = Operation::new(0, p0, p1, ParamNone, ParamNone);

    session.invoke_command(Command::Derive as u32, &mut operation)?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn encrypt_tensor(
    plaintext: *const c_char,
    ciphertext: *mut c_char,
    text_len: i32,
    cmd_idx: u32,
    op_name: *const c_char,
    on_len: i32,
    protected_tensor: *const c_char,
    pt_len: i32,
) {
    let mut ctx = Context::new().unwrap();
    let uuid = Uuid::parse_str(UUID).unwrap();
    let mut session = ctx.open_session(uuid).unwrap();

    let on_bytes = unsafe { std::slice::from_raw_parts(op_name, on_len as usize) };
    let pt_bytes = unsafe { std::slice::from_raw_parts(protected_tensor, pt_len as usize) };
    let on_str = std::str::from_utf8(on_bytes).unwrap();
    let pt_str = std::str::from_utf8(pt_bytes).unwrap();
    let tname = convert_name(on_str, pt_str); // "w"

    let mut tee_config = TEE_CONFIG.lock().unwrap();

    let mut tensor_id: String = cmd_idx.to_string();
    tensor_id.push_str(&tname); // "0w"
    info!("CA: derive config key and tensor key in TA");
    derive_key(&mut session, &tee_config.config_id, tensor_id.as_bytes()).unwrap();

    let pt_bytes = unsafe { std::slice::from_raw_parts(plaintext, text_len as usize) };
    let ct_bytes = unsafe { std::slice::from_raw_parts_mut(ciphertext, (text_len) as usize) };
    ct_bytes.clone_from_slice(pt_bytes);

    let mut encrypt_info: Vec<&mut EncryptInfo> = tee_config
        .encrypt_info
        .iter_mut()
        .filter(|v| v.command_index == cmd_idx && v.tensor_name == tname.to_string())
        .collect();

    let iv = encrypt_info[0].iv.clone();
    let mut cmac = &mut encrypt_info[0].cmac;
    info!("CA: encrypt {}'s tensor {} in TA", on_str, pt_str);
    cipher_buffer(&mut session, Command::Encrypt, ct_bytes, &iv, &mut cmac).unwrap();
}

#[no_mangle]
pub extern "C" fn write_tee_config() {
    let tee_config = TEE_CONFIG.lock().unwrap();
    info!("{:?}", tee_config);
    let tee_config_bytes = proto::serde_json::to_vec::<TEEModelConfig>(&tee_config).unwrap();
    let mut file = File::create("tee_config.json").expect("create failed");
    file.write(&tee_config_bytes).expect("write failed");
    info!("CA: Write to tee_config.json success!");
    let _sign = process::Command::new("openssl")
        .args(&[
            "dgst",
            "-sha256",
            "-sign",
            "tee_config.key",
            "-out",
            "tee_config.sig",
            "tee_config.json",
        ])
        .output()
        .expect("failed to sign file");
    info!("CA: Sign file success!");
}
