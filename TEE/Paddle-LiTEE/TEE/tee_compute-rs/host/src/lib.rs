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

extern crate pretty_env_logger;
#[macro_use]
mod macros;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;

mod ffi;
mod init_context;

use anyhow::Result;
use optee_teec::{Context, Operation, ParamType, Uuid};
use optee_teec::{Error, ErrorKind};
use optee_teec::{ParamNone, ParamTmpRef, ParamValue};
use proto::fc_param::*;
use proto::softmax_param::*;
use proto::tee_tensor::TEEParam;
use proto::{Command, UUID};
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub fn invoke_ta_compute<T: TEEParam + Serialize>(
    tee_param: &mut T,
    command: Command,
) -> Result<Vec<u8>> {
    info!("CA: prepare to invoke ta, command: {:?}", &command);
    let mut ctx = Context::new()?;
    let uuid = Uuid::parse_str(UUID)?;
    let mut session = ctx.open_session(uuid)?;

    let start_time = Instant::now();
    let serialized_input: Vec<u8> = bincode::serialize(&tee_param)?;
    trace!(
        "CA: after serialize input: {}ms",
        start_time.elapsed().as_millis()
    );
    debug!("CA: input.len: {:?}", &serialized_input.len());
    let mut serialized_output = vec![0u8; serialized_input.len()];

    let p0 = ParamTmpRef::new_input(serialized_input.as_slice());
    let p1 = ParamTmpRef::new_output(serialized_output.as_mut_slice());

    let mut operation = Operation::new(0, p0, p1, ParamNone, ParamNone);
    session.invoke_command(command as u32, &mut operation)?;
    info!("CA: invoke ta finished");
    Ok(serialized_output)
}

// load tee_config.signed, verify the signature, init TEEContext
pub fn invoke_ta_init_context(
    config_buffer: &[u8],
    public_key: &[u8],
    signature: &[u8],
) -> Result<()> {
    // send tee_config to ta
    let mut ctx = Context::new()?;
    let uuid = Uuid::parse_str(UUID)?;
    let mut session = ctx.open_session(uuid)?;

    let p0 = ParamTmpRef::new_input(config_buffer);
    let p1 = ParamTmpRef::new_input(public_key);
    let p2 = ParamTmpRef::new_input(signature);
    let mut operation = Operation::new(0, p0, p1, p2, ParamNone);
    session.invoke_command(Command::InitContext as u32, &mut operation)?;
    info!("CA: invoke ta finished");

    Ok(())
}
