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

use anyhow::Result;
use optee_utee::trace_println;
use ring::aead;
use serde::{Deserialize, Serialize};

// AES_GCM 128
// encrypting/decrypting data-key
const AES_GCM_128_KEY_LENGTH: usize = 16;
const AES_GCM_128_IV_LENGTH: usize = 12;
const CMAC_LENGTH: usize = 16;
const FILE_CHUNK_SIZE: usize = 1024 * 1024;
type CMac = [u8; CMAC_LENGTH];

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AesGcm128Key {
    pub key: [u8; AES_GCM_128_KEY_LENGTH],
    pub iv: [u8; AES_GCM_128_IV_LENGTH],
}

impl AesGcm128Key {
    pub const SCHEMA: &'static str = "aes-gcm-128";

    pub fn new(in_key: &[u8], in_iv: &[u8]) -> Result<Self> {
        let mut key = [0u8; AES_GCM_128_KEY_LENGTH];
        let mut iv = [0u8; AES_GCM_128_IV_LENGTH];
        key.copy_from_slice(in_key);
        iv.copy_from_slice(in_iv);

        Ok(AesGcm128Key { key, iv })
    }

    pub fn decrypt(&self, in_out: &mut Vec<u8>) -> Result<CMac> {
        let plaintext_len = aead_decrypt(&aead::AES_128_GCM, in_out, &self.key, &self.iv)?.len();
        let mut cmac: CMac = [0u8; CMAC_LENGTH];
        cmac.copy_from_slice(&in_out[plaintext_len..]);
        in_out.truncate(plaintext_len);
        Ok(cmac)
    }

    pub fn encrypt(&self, in_out: &mut Vec<u8>) -> Result<CMac> {
        aead_encrypt(&aead::AES_128_GCM, in_out, &self.key, &self.iv)?;
        let mut cmac: CMac = [0u8; CMAC_LENGTH];
        let n = in_out.len();
        let cybertext_len = n - CMAC_LENGTH;
        cmac.copy_from_slice(&in_out[cybertext_len..]);
        Ok(cmac)
    }
}

pub fn aead_encrypt(
    alg: &'static aead::Algorithm,
    in_out: &mut Vec<u8>,
    key: &[u8],
    iv: &[u8],
) -> Result<()> {
    let key = aead::UnboundKey::new(alg, key)?;
    let nonce = aead::Nonce::try_assume_unique_for_key(iv)?;
    let aad = aead::Aad::from([0u8; 8]);

    let enc_key = aead::LessSafeKey::new(key);
    enc_key.seal_in_place_append_tag(nonce, aad, in_out)?;
    Ok(())
}

pub fn aead_decrypt<'a>(
    alg: &'static aead::Algorithm,
    in_out: &'a mut [u8],
    key: &[u8],
    iv: &[u8],
) -> Result<&'a mut [u8]> {
    let key = aead::UnboundKey::new(alg, key)?;
    let nonce = aead::Nonce::try_assume_unique_for_key(iv)?;
    let aad = aead::Aad::from([0u8; 8]);

    let dec_key = aead::LessSafeKey::new(key);
    let slice = dec_key.open_in_place(nonce, aad, in_out)?;
    Ok(slice)
}
