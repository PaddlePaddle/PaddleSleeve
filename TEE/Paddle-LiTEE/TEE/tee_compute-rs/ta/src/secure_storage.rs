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

use optee_utee::trace_println;
use optee_utee::{DataFlag, ObjectStorageConstants, PersistentObject};
use optee_utee::{Error, ErrorKind, Result};

pub fn save_in_secure_storage(obj_id: &mut Vec<u8>, data: &Vec<u8>) -> Result<()> {
    let obj_data_flag = DataFlag::ACCESS_READ
        | DataFlag::ACCESS_WRITE
        | DataFlag::ACCESS_WRITE_META
        | DataFlag::OVERWRITE;

    let mut init_data: [u8; 0] = [0; 0];
    match PersistentObject::create(
        ObjectStorageConstants::Private,
        obj_id,
        obj_data_flag,
        None,
        &mut init_data,
    ) {
        Err(e) => {
            return Err(e);
        }

        Ok(mut object) => match object.write(&data) {
            Ok(()) => {
                return Ok(());
            }
            Err(e_write) => {
                object.close_and_delete()?;
                std::mem::forget(object);
                return Err(e_write);
            }
        },
    }
}

pub fn load_from_secure_storage(obj_id: &mut Vec<u8>) -> Result<Vec<u8>> {
    let mut buf = vec![0; 5000];

    match PersistentObject::open(
        ObjectStorageConstants::Private,
        obj_id,
        DataFlag::ACCESS_READ | DataFlag::SHARE_READ,
    ) {
        Err(e) => return Err(e),

        Ok(object) => {
            let obj_info = object.info()?;

            if obj_info.data_size() > buf.len() {
                return Err(Error::new(ErrorKind::ShortBuffer));
            }
            let read_bytes = object.read(&mut buf).unwrap();
            //trace_println!("[+] {:?}: read_bytes: {:?}", &obj_id, &read_bytes);

            if read_bytes != obj_info.data_size() as u32 {
                return Err(Error::new(ErrorKind::ExcessData));
            }

            unsafe {
                buf.set_len(read_bytes as usize);
            }
        }
    }

    Ok(buf)
}

pub fn delete_from_secure_storage(mut obj_id: Vec<u8>) -> Result<()> {
    match PersistentObject::open(
        ObjectStorageConstants::Private,
        &mut obj_id,
        DataFlag::ACCESS_READ | DataFlag::ACCESS_WRITE_META,
    ) {
        Err(e) => return Err(e),

        Ok(mut object) => {
            object.close_and_delete()?;
            std::mem::forget(object);
            return Ok(());
        }
    }

    Ok(())
}
