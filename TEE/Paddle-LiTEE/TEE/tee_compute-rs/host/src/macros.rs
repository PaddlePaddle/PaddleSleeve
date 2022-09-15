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

macro_rules! create_param {
    ($static_item:ident, $param_type:ty, $param_ptr: ident, $start_time:ident) => {
        info!("CA: create TEEParam from PTParam");
        let param: $param_type = <$param_type>::from($param_ptr);
        info!("CA: create TEEParam after from");
        let param_handle = match $static_item.lock() {
            Ok(mut p) => p.insert(param),
            Err(_) => {
                error!("CA: create TEEParam error");
                return 1;
            }
        };
        trace!(
            "CA: after create tee param: {}ms",
            $start_time.elapsed().as_millis()
        );
        info!("CA: create param handle, return to Paddle-LiTEE kernel");
        return param_handle.into_raw();
    };
}
macro_rules! run_tee_command {
    ($static_item:ident, $param_type:ty, $param_handle: ident, $command: ident, $start_time:ident) => {
        info!("CA: get handle: {}", $param_handle);
        let mut _param_h = match $static_item.lock() {
            Ok(p) => p,
            Err(e) => {
                error!("CA: handlemap lock error: {}", e);
                return 1;
            }
        };
        let mut param = match _param_h.get_mut(Handle::from_raw($param_handle)) {
            Some(p) => p,
            None => {
                error!("CA: failed in fetching tee param");
                return 1;
            }
        };
        trace!(
            "CA: after get tee param: {}ms",
            $start_time.elapsed().as_millis()
        );
        let serialized_output = match invoke_ta_compute(&mut param, $command) {
            Ok(output) => output,
            Err(e) => {
                error!("CA: invoke_ta_compute error: {}", e);
                return 1;
            }
        };
        trace!(
            "CA: after invoke_ta_compute: {}ms",
            $start_time.elapsed().as_millis()
        );
        let _tee_param: $param_type = match bincode::deserialize(&serialized_output) {
            Ok(output) => output,
            Err(e) => {
                error!("CA: tee param deserialize error: {}", e);
                return 1;
            }
        };
        trace!(
            "CA: after deserialize tee param: {}ms",
            $start_time.elapsed().as_millis()
        );
        param.output = _tee_param.output;
        info!("CA: TA compute finished, return to Paddle-LiTEE kernel");
    };
}
macro_rules! fetch_output {
    ($static_item:ident, $pt_param_type:ty, $param_handle:ident, $output_tensor:ident, $start_time:ident) => {
        info!("CA: get handle: {}", $param_handle);
        let mut _param_h = match $static_item.lock() {
            Ok(p) => p,
            Err(e) => {
                error!("CA: handlemap lock error: {}", e);
                return 1;
            }
        };
        let mut tee_param = match _param_h.remove(Handle::from_raw($param_handle)) {
            Some(p) => p,
            None => {
                error!("CA: failed in fetching tee param");
                return 1;
            }
        };
        trace!(
            "CA: after remove tee param: {}ms",
            $start_time.elapsed().as_millis()
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                tee_param.output.data.as_ptr() as *mut ::std::os::raw::c_void,
                $output_tensor.bytes,
                $output_tensor.byte_size as usize,
            );
        }
        trace!(
            "CA: after copy output data: {}ms",
            $start_time.elapsed().as_millis()
        );
        info!("CA: Write to output tensor finished, return to Paddle-LiTEE kernel");
    };
}
