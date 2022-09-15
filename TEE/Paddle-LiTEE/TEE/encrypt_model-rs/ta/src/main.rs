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

use optee_utee::TransientObject;
use optee_utee::AE;
use optee_utee::{
    ta_close_session, ta_create, ta_destroy, ta_invoke_command, ta_open_session, trace_println,
};
use optee_utee::{Error, ErrorKind, Parameters, Result};
use proto::Command;
use ring::aead;
use ring::pbkdf2::*;
use std::io::Write;

pub struct AesCipher {
    pub key_size: usize,
    pub ae: AE,
    pub key_object: TransientObject,
    pub root_key: [u8; 16],
    pub config_key: [u8; 16],
    pub tensor_key: [u8; 16],
}

impl Default for AesCipher {
    fn default() -> Self {
        Self {
            key_size: 0,
            ae: AE::null(),
            key_object: TransientObject::null_object(),
            root_key: [
                0x56, 0xa9, 0xbd, 0xc3, 0xa8, 0x93, 0x78, 0x41, 0xbc, 0xa6, 0xd5, 0x91, 0x83, 0xb7,
                0xe0, 0x49,
            ],
            config_key: [0x00u8; 16],
            tensor_key: [0x00u8; 16],
        }
    }
}

#[ta_create]
fn create() -> Result<()> {
    trace_println!("[+] TA create");
    Ok(())
}

#[ta_open_session]
fn open_session(_params: &mut Parameters, _sess_ctx: &mut AesCipher) -> Result<()> {
    trace_println!("[+] TA open session");
    Ok(())
}

#[ta_close_session]
fn close_session(_sess_ctx: &mut AesCipher) {
    trace_println!("[+] TA close session");
}

#[ta_destroy]
fn destroy() {
    trace_println!("[+] TA destory");
}

#[ta_invoke_command]
fn invoke_command(sess_ctx: &mut AesCipher, cmd_id: u32, params: &mut Parameters) -> Result<()> {
    trace_println!("[+] TA invoke command");
    match Command::from(cmd_id) {
        Command::Encrypt => {
            return encrypt_buffer(sess_ctx, params);
        }
        Command::Decrypt => {
            return decrypt_buffer(sess_ctx, params);
        }
        Command::Derive => {
            return derive_key(sess_ctx, params);
        }
        _ => {
            return Err(Error::new(ErrorKind::BadParameters));
        }
    }
}

pub fn encrypt_buffer(aes: &mut AesCipher, params: &mut Parameters) -> Result<()> {
    let mut param0 = unsafe { params.0.as_memref().unwrap() };
    let mut in_out = param0.buffer().to_vec();
    let in_out_len = in_out.len();
    let mut param1 = unsafe { params.1.as_memref().unwrap() };
    let iv = param1.buffer();
    let mut param2 = unsafe { params.2.as_memref().unwrap() };

    trace_println!("encrypt tensor...");
    let key = aead::UnboundKey::new(&aead::AES_128_GCM, &aes.tensor_key).unwrap();
    let nonce = aead::Nonce::try_assume_unique_for_key(iv).unwrap();
    let aad = aead::Aad::from([0u8; 8]);

    let enc_key = aead::LessSafeKey::new(key);

    enc_key
        .seal_in_place_append_tag(nonce, aad, &mut in_out)
        .unwrap();
    trace_println!("encrypt finished");
    param0.buffer().write(&in_out[..in_out_len]).unwrap();
    param2.buffer().write(&in_out[in_out_len..]).unwrap();
    Ok(())
}

pub fn decrypt_buffer(aes: &mut AesCipher, params: &mut Parameters) -> Result<()> {
    let mut param0 = unsafe { params.0.as_memref().unwrap() };
    let mut in_out = param0.buffer().to_vec();
    let mut param1 = unsafe { params.1.as_memref().unwrap() };
    let iv = param1.buffer();

    let key = aead::UnboundKey::new(&aead::AES_128_GCM, &aes.tensor_key).unwrap();
    let nonce = aead::Nonce::try_assume_unique_for_key(&iv).unwrap();
    let aad = aead::Aad::from([0u8; 8]);

    let dec_key = aead::LessSafeKey::new(key);
    dec_key.open_in_place(nonce, aad, &mut in_out).unwrap();
    param0.buffer().write(&in_out).unwrap();
    Ok(())
}

pub fn derive_key(aes: &mut AesCipher, params: &mut Parameters) -> Result<()> {
    let mut p0 = unsafe { params.0.as_memref().unwrap() };
    let mut p1 = unsafe { params.1.as_memref().unwrap() };
    let config_id = p0.buffer();
    let tensor_id = p1.buffer();
    trace_println!("[+] config_id = {:?}", config_id);
    trace_println!("[+] tensor_id = {:?}", tensor_id);

    let iter = std::num::NonZeroU32::new(100).unwrap();
    derive(
        PBKDF2_HMAC_SHA256,
        iter,
        config_id,
        &aes.root_key,
        &mut aes.config_key,
    );
    derive(
        PBKDF2_HMAC_SHA256,
        iter,
        tensor_id,
        &aes.config_key,
        &mut aes.tensor_key,
    );

    trace_println!("[+] aes.config_key = {:?}", aes.config_key);
    trace_println!("[+] aes.tensor_key = {:?}", aes.tensor_key);
    Ok(())
}

const TA_FLAGS: u32 = 0;
const TA_DATA_SIZE: u32 = 16 * 1024 * 1024;
const TA_STACK_SIZE: u32 = 1024 * 1024;
const TA_VERSION: &[u8] = b"0.1\0";
const TA_DESCRIPTION: &[u8] = b"This is an AES GCM\0";
const EXT_PROP_VALUE_1: &[u8] = b"AES TA\0";
const EXT_PROP_VALUE_2: u32 = 0x0010;
const TRACE_LEVEL: i32 = 4;
const TRACE_EXT_PREFIX: &[u8] = b"TA\0";
const TA_FRAMEWORK_STACK_SIZE: u32 = 2048;

include!(concat!(env!("OUT_DIR"), "/user_ta_header.rs"));
