# Paddle-LiTEE

Paddle-LiTEE is the framework for model protection on mobile(ARM) devices, based
on [Teaclave TrustZone SDK](https://github.com/apache/incubator-teaclave-trustzone-sdk) 
and [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite). 

Paddle-LiTEE is designed to protect models based on ARM TrustZone. Protected 
parameters in models are encrypted in the pre-deployment procedure and decrypted 
inside ARM TrustZone during inference. Operations relevant to protected parameters 
are computed inside ARM TrustZone and the result is returned to Normal World.

## Table of Contents

- [Getting Started](#getting-started)
  - [Building Environment](#building-environment)
  - [Model Encryption Tool](#model-encryption-tool)
    - [Build](#build)
    - [Run](#run)
  - [Inference Framework](#inference-framework)
    - [Build](#if-build)
    - [Run](#if-run)
- [Features](#features)
- [Copyright and License](#copyright-and-license)

## Getting Started

#### Building Environment

We recommend using our docker (teaclave/teaclave-trustzone-sdk-build:0.3.0) which 
satisfies the building prerequisites of Rust TAs. 

```
$ docker pull teaclave/teaclave-trustzone-sdk-build:0.3.0
```

Besides, you should install building dependencies of Paddle-Lite after setting up 
the container:

```
$ apt-get install -y --no-install-recommends \
  g++-arm-linux-gnueabi gcc-arm-linux-gnueabi \
  g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
$ wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
  tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
  mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 && \
  ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
  ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake
```

#### Model Encryption Tool

The model encryption tool is invoked in the pre-deployment process and consisted 
of two parts: the model parser in Normal World and the encryption TA in 
TrustZone.

The model parser is based on Paddle-Lite v2.9.1. We've modified the original
model parser module of Paddle-Lite. The model parser reads user config, extracts
parameters from model, then send them to TrustZone for encryption. After
encryption, the model parser will write the cipher into model.

The encryption TA reads parameters then encrypt them with `AES_GCM_128`
algorithm and the root key is stored in TrustZone.  

##### Build<a id='met-build'></a>

###### 1. Build encrypt_model-rs TA

TAs are developed based on 
[Teaclave TrustZone SDK](https://github.com/apache/incubator-teaclave-trustzone-sdk).
So you should setup the environment first.

```
$ cd Paddle-LiTEE/TEE
$ git clone https://github.com/apache/incubator-teaclave-trustzone-sdk.git
$ (cd incubator-teaclave-trustzone-sdk && \
  ./setup.sh && \
  source environment && \
  make optee)
```

By default, the dynamic library `libteec.so` links to CA binary, that will cause
the `undefined reference` error when building model parser. To staticly link
`libteec.a` into CA, you should modify the linking method:

```
diff --git a/optee-teec/optee-teec-sys/build.rs b/optee-teec/optee-teec-sys/build.rs
index 2d8fc42..2465b63 100644
--- a/optee-teec/optee-teec-sys/build.rs
+++ b/optee-teec/optee-teec-sys/build.rs
@@ -22,5 +22,5 @@ fn main() {
     let optee_client_dir = env::var("OPTEE_CLIENT_DIR").unwrap_or("../../optee/optee_client/out".to_string());
     let search_path = Path::new(&optee_client_dir).join("libteec");
     println!("cargo:rustc-link-search={}", search_path.display());
-    println!("cargo:rustc-link-lib=dylib=teec");
+    println!("cargo:rustc-link-lib=static=teec");
 }
```

Then build encrypt_model-rs TA:

```
$ cd TEE/encrypt_model-rs
$ make
```

The UUID of `encrypt_model-rs` TA is `d9ae4934-ceaa-4862-838f-fb5a9f8b7f02`. 

After building you can find the `d9ae4934-ceaa-4862-838f-fb5a9f8b7f02.ta` in
`encrypt_model-rs/ta/target/aarch64-unknown-optee-trustzone/release/` and the
corresponding CA is
`encrypt_model-rs/host/target/aarch64-unknown-linux-gnu/release/libencrypt_model.a`.

###### 2. Build model parser library

Copy `libencrypt_model.a` CA into
`model_encryption_tool/Paddle-Lite/lite/model_parser/`. Then build model parser
library:

```
$ cd model_encryption_tool/
$ ./setup-build-env.sh && cd Paddle-Lite
$ cp ../../TEE/encrypt_model-rs/host/target/aarch64-unknown-linux-gnu/release/libencrypt_model.a lite/model_parser 
$ ./lite/tools/build_linux.sh full_publish
```

Same as Paddle-Lite, the Paddle-LiTEE model parser library is located at
`build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/lib/` and
named as `libpaddle_light_api_shared.so`.

##### Run<a id='met-run'></a>

###### 1. Setup OP-TEE environment

Follow instructions in OP-TEE documentation to set up OP-TEE QEMUv8 environment:
[OP-TEE with Rust &mdash; OP-TEE documentation
documentation](https://optee.readthedocs.io/en/latest/building/optee_with_rust.html).

By default, the rootfs images inside OP-TEE repo is a buildroot rootfs image
which may not fit the dependencies of Paddle-Lite. You can built a new image
based on Ubuntu 20.04 or use our 
[prebuilt image](https://nightlies.apache.org/teaclave/teaclave-trustzone-sdk/ubuntu-20.04-rootfs_ext4.img.tar.zst).
(arm64:arm64, root:root)

Execute this command to start QEMU:

start_qemu.sh:

```
$ qemu-system-aarch64 \
        -nographic \
        -serial tcp:localhost:54320 -serial tcp:localhost:54321 \
        -smp 4 \
        -s -S -machine virt,secure=on,mte=off,gic-version=3,virtualization=false \
        -cpu max,sve=off \
        -d unimp -semihosting-config enable=on,target=native \
        -m 3072 \
        -bios bl1.bin  \
        -kernel Image -no-acpi \
        -append 'console=ttyAMA0,38400 keep_bootcon root=/dev/vda rw ' \
        -drive if=none,file=ubuntu-20.04-rootfs_ext4.img,id=hd0,format=raw -device virtio-blk-device,drive=hd0  \
        -object rng-random,filename=/dev/urandom,id=rng0 -device virtio-rng-pci,rng=rng0,max-bytes=1024,period=1000 -fsdev local,id=fsdev0,path=/home/user/shared/,security_model=none -device virtio-9p-device,fsdev=fsdev0,mount_tag=host -netdev user,id=vmnic -device virtio-net-device,netdev=vmnic
```

###### 2. Run encrypt_model

After QEMU boots up, copy TA into `/lib/optee_armtz` then build encrypt_model:

```
# mkdir shared
# mount -t 9p -o trans=virtio host shared && cd shared
# tee-supplicant &
# cp d9ae4934-ceaa-4862-838f-fb5a9f8b7f02.ta /lib/optee_armtz/
# cd tool/ && ./build_enc.sh
```

The `model.nb` for testing is the `MobileNet v1` model downloaded from
[Paddle-Lite](https://paddlelite-demo.bj.bcebos.com/demo/image_classification/models/mobilenet_v1_for_cpu_v2_10.tar.gz).

Run encrypt_model:

```
# ./start_enc.sh
```

The outputs of this tool are `model_tee.nb` and `tee_config.json`.

###### 3. Generate key pairs and sign the configuration file

In the inference process, TEE verifies the signature of configuration file.
Generate the key pairs and sign the configuration after encrypting models:

```
# generate private key
$ openssl genrsa -out tee_config.key 2048
# generate public key
$ openssl rsa -in private.pem -pubout -out public.pem
# transform PEM format to DER format (which the ring crate supports)
$ openssl rsa -pubin -in public.pem -inform PEM -RSAPublicKey_out -outform DER -out tee_config.der
# sign with your private key
$ openssl dgst -sha256 -sign private.pem -out tee_config.sig tee_config.json
```

#### Inference Framework

A new kernel of operations named `ARM_TRUSTZONE` is added to Paddle-Lite.

In the pre-deployment procedure, the model encryption tool modifies `target` of
protected operations (ops) so that the `ARM_TRUSTZONE` kernel is invoked in
those ops during inference.

The CA `libtee_compute.a` is statically linked into the Paddle-Lite library
`libpaddle_light_api_shared.so` .`tee_compute` TA receives tensors sent from CA,
verifies the signature, and invokes `math backends` . The `math backends` do
basic math computations which are ported from Paddle-Lite's math backends.

##### Build<a id='if-build'></a>

###### 1. Build tee_compute TA

The building steps are same as `Build encrypt_model-rs TA`.

Besides, the `tee_compute` TA is located at
`tee_compute-rs/ta/target/ta/target/aarch64-unknown-optee-trustzone/release/90071bb2-3e97-4f9f-9523-27bc6cd4375a.ta`
and corresponding CA is located at
`tee_compute-rs/host/target/aarch64-unknown-linux-gnu/release/libtee_compute.a`. 

###### 2. Build Paddle-Lite dynamic library

Copy CA into Paddle-Lite kernel path:

```
cp tee_compute-rs/host/target/aarch64-unknown-linux-gnu/release/libtee_compute.a Paddle-Lite/lite/kernel/arm_trustzone/
```

Build Paddle-Lite with ARM TrustZone:

```
cd Paddle-Lite
./lite/tools/build_linux.sh --with_arm_trustzone=ON --with_extra=ON full_publish
```

The output is `libpaddle_light_api_shared.so` in 
`Paddle-Lite/build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/lib/libpaddle_light_api_shared.so`.

##### Run<a id='if-run'></a>

###### 1. Setup OP-TEE environment

See [Setup OP-TEE environment](#setup-op-tee-environment) above.

###### 2. Prepare essential files

Make sure you have these files:

- Paddle-Lite libraries and headers: `Paddle-Lite/build.lite.linux.armv8.gcc/`

- encrypted model: `model_tee.nb`

- configuration file and signature: `tee_config.json`, `tee_config.sig`

- developer's public key: `tee_config.der`

###### 3. Build image classification demo and run inference

`Paddle-LiTEE/image_classification_demo` is modified from 
[Paddle-Lite-Demo](https://github.com/PaddlePaddle/Paddle-Lite-Demo.git).

Edit `build.sh` and `run.sh`:

- Set `PADDLE_LITE_DIR` to the path of Paddle-Lite inference library.

- Set `CONFIG_DIR` to the path where `tee_config.json` and `tee_config.sig` 
and `tee_config.der` exist.

Build the demo and run:

```
$ cd Paddle-LiTEE/image_classification_demo/
$ ./build.sh
$ ./run.sh
```

The result is:

```
results: 3
Top0  tabby, tabby cat - 0.475009
Top1  Egyptian cat - 0.409485
Top2  tiger cat - 0.095745
Preprocess time: 33.598000 ms
Prediction time: 11138.098633 ms
Postprocess time: 7.247000 ms
```

Note:

To enable debug log:

```
$ export GLOG_v=5
```

## Features

#### Supported operations running in ARM TrustZone
- fc
- softmax

#### Supported platform
- ARM Linux

## Copyright and License

Paddle-LiTEE is maintained by Baidu Security Lab and available under the Apache-2.0 License.
