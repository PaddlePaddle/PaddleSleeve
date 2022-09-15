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

use crate::invoke_ta_compute;
use crate::pretty_env_logger;
#[macro_use]
use crate::macros::*;

use handy::{Handle, HandleMap};
use proto::c_bindings::*;
use proto::fc_param::*;
use proto::softmax_param::*;
use proto::tee_tensor::*;
use proto::Command;
use std::sync::Mutex;
use std::time::Instant;

lazy_static! {
    static ref FC_ITEMS: Mutex<HandleMap<TEEFcParam>> = Mutex::new(HandleMap::new());
    static ref SOFTMAX_ITEMS: Mutex<HandleMap<TEESoftmaxParam>> = Mutex::new(HandleMap::new());
}

#[no_mangle]
pub extern "C" fn create_tee_param(
    op: SupportedOp,
    param_ptr: *mut ::std::os::raw::c_void,
) -> handle_t {
    let start_time = Instant::now();

    info!("CA: create_tee_param() invoked by Paddle-LiTEE kernel");
    info!("CA: op: {:?}", &Command::from(op as u32));
    match op {
        SupportedOp_FcRun => {
            debug!("CA: invoke create_param! for fc");
            create_param!(FC_ITEMS, TEEFcParam, param_ptr, start_time);
        }
        SupportedOp_Softmax => {
            debug!("CA: invoke create_param! for softmax");
            create_param!(SOFTMAX_ITEMS, TEESoftmaxParam, param_ptr, start_time);
        }
        _ => {
            error!("CA: unsupported op");
        }
    };
    trace!(
        "CA: after create_tee_param: {}ms",
        start_time.elapsed().as_millis()
    );

    0
}

#[no_mangle]
pub extern "C" fn tee_run(op: SupportedOp, param_handle: handle_t) -> u64 {
    let start_time = Instant::now();
    info!("CA: tee_run() invoked by Paddle-LiTEE kernel");
    info!("CA: op: {:?}", &Command::from(op as u32));
    // write tee_param back to CA
    match op {
        SupportedOp_FcRun => {
            debug!("CA: invoke run_tee_command! for fc");
            let command = Command::from(SupportedOp_FcRun as u32);
            run_tee_command!(FC_ITEMS, TEEFcParam, param_handle, command, start_time);
        }
        SupportedOp_Softmax => {
            debug!("CA: invoke run_tee_command! for softmax");
            let command = Command::from(SupportedOp_Softmax as u32);
            run_tee_command!(
                SOFTMAX_ITEMS,
                TEESoftmaxParam,
                param_handle,
                command,
                start_time
            );
        }
        _ => {
            error!("CA: unsupported op");
            return 1;
        }
    };
    trace!("CA: after tee_run: {}ms", start_time.elapsed().as_millis());

    0
}

#[no_mangle]
pub extern "C" fn fetch_output_tensor(
    op: SupportedOp,
    param_handle: handle_t,
    output_tensor: PortableTensor,
) -> u64 {
    let start_time = Instant::now();
    info!("CA: fetch_output_tensor() invoked by Paddle-LiTEE kernel");
    info!("CA: op: {:?}", &Command::from(op as u32));
    match op {
        SupportedOp_FcRun => {
            fetch_output!(
                FC_ITEMS,
                PT_FcParam,
                param_handle,
                output_tensor,
                start_time
            );
        }
        SupportedOp_Softmax => {
            fetch_output!(
                SOFTMAX_ITEMS,
                PT_SoftmaxParam,
                param_handle,
                output_tensor,
                start_time
            );
        }
        _ => {
            error!("CA: unsupported op");
            return 1;
        }
    };
    trace!(
        "CA: after fetch_output_tensor: {}ms",
        start_time.elapsed().as_millis()
    );

    0
}
