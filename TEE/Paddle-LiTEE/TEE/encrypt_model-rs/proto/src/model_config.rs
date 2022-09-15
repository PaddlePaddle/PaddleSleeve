use crate::Command;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct UserConfig {
    pub op_list: Vec<ParamInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct ParamInfo {
    pub index: u32,
    pub name: String,
    pub protected_param: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TEEModelConfig {
    pub config_id: Vec<u8>,
    pub command_list: Vec<TEECommandInfo>,
    pub encrypt_algorithm: EncAlgo,
    pub encrypt_info: Vec<EncryptInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TEECommandInfo {
    pub command: Command,
    pub input_tensor: Vec<TEEConfigTensor>,
    pub output_tensor: Vec<TEEConfigTensor>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TEEConfigTensor {
    pub name: String,
    pub action: TEEAction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TEEAction {
    Decrypt,
    ReadFromCache,
    WriteToCache,
    ReturnToCA,
}

impl Default for TEEAction {
    fn default() -> Self {
        TEEAction::Decrypt
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EncAlgo {
    AesGcm128,
}

impl Default for EncAlgo {
    fn default() -> Self {
        EncAlgo::AesGcm128
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct EncryptInfo {
    pub command_index: u32,
    pub tensor_name: String,
    pub iv: Vec<u8>,
    pub cmac: Vec<u8>,
}
