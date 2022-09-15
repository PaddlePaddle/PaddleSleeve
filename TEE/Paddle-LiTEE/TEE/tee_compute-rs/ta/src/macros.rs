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

macro_rules! handle_input_tensor {
    ($param:ident, $run_info:ident, $time:ident) => {
        for each_input_tensor in $run_info.command_info.input_tensor {
            match each_input_tensor.action {
                TEEAction::Decrypt => {
                    $time.system_time();
                    trace_println!("[*] Benchmark: invoke decrypt: {}", $time);

                    let tensor_name = each_input_tensor.name.clone();
                    let mut current_encrypted_tensor = EncryptedTensor::default();
                    //trace_println!("[+] Decrypt: param[tensor_name]: {:?}", &$param[tensor_name.clone()].data[..16]);
                    for each_enc_tensor in &$run_info.encrypted_tensor {
                        if each_enc_tensor.name == tensor_name {
                            current_encrypted_tensor = each_enc_tensor.clone();
                        }
                    }
                    trace_println!("[+] handle_input_tensor: action: Decrypt");
                    trace_println!("[+] current_encrypted_tensor {:?}", &current_encrypted_tensor);

                    $time.system_time();
                    trace_println!("[*] Benchmark: find encrypted tensor: {}", $time);

                    let mut buf = &mut $param[tensor_name.clone()].data;
                    buf.extend_from_slice(&current_encrypted_tensor.cmac);
                    trace_println!("[+] tensor.data.len: {:?}", buf.len());
                    trace_println!("[+] tensor.data: {:?}", &buf[..16]);

                    $time.system_time();
                    trace_println!("[*] Benchmark: before decrypt: {}", $time);

                    let decrypted_data = aead_decrypt(&ring::aead::AES_128_GCM,
                                                      &mut buf,
                                                      &current_encrypted_tensor.key,
                                                      &current_encrypted_tensor.iv
                                                      )?;
                    //trace_println!("[+] handle_input_tensor: Decrypted data: {:?}", &decrypted_data[..16]);
                    trace_println!("[+] handle_input_tensor: data decrypted");

                    $time.system_time();
                    trace_println!("[*] Benchmark: after decrypt: {}", $time);

                    $param[tensor_name.clone()].data = decrypted_data.to_vec();

                    $time.system_time();
                    trace_println!("[*] Benchmark: decrypt data copied: {}", $time);
                },
                TEEAction::ReadFromCache => {
                    let tensor_name = each_input_tensor.name.clone();
                    trace_println!("[+] handle_input_tensor: action: ReadFromCache, tensor name: {}", &tensor_name);
                    //TODO:let mut obj_id = ($param[tensor_name.clone()].data)[..16].to_vec();
                    let mut obj_id = b"output.data".to_vec();
                    $param[tensor_name].data = load_from_secure_storage(&mut obj_id)?;
                    trace_println!("[+] handle_input_tensor: input loaded from secure storage");
                },
                _ => {
                    return Err(anyhow!("handle_input_tensor: action not supported"));
                },
            };
        }
    }
}

macro_rules! handle_output_tensor {
    ($param:ident, $run_info:ident) => {
        for each_output_tensor in $run_info.command_info.output_tensor {
            match each_output_tensor.action {
                TEEAction::WriteToCache => {
                    let tensor_name = each_output_tensor.name.clone();
                    trace_println!(
                        "[+] handle_output_tensor: action: WriteToCache: tensor_name:{}",
                        &tensor_name
                    );
                    //TODO:let mut obj_id = ($param[tensor_name.clone()].data)[..16].to_vec();
                    let mut obj_id = b"output.data".to_vec();
                    save_in_secure_storage(&mut obj_id, &$param[tensor_name.clone()].data)?;
                    trace_println!("[+] handle_output_tensor: output saved in secure storage");

                    //TODO: return id to output buffer
                    //fc_param.output.data = vec![0; obj_id.len()];
                    //fc_param.output.data.clone_from_slice(&obj_id);
                }
                TEEAction::ReturnToCA => {
                    let tensor_name = each_output_tensor.name.clone();
                    trace_println!(
                        "[+] handle_output_tensor: action: ReturnToCA: tensor name:{}",
                        &tensor_name
                    );
                    trace_println!("[+] handle_output_tensor: output will return to CA buffer");
                }
                _ => {
                    return Err(anyhow!("handle_output_tensor: action not supported"));
                }
            };
        }
    };
}
