// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

//! Load rendering shaders.

mod preprocess;

use std::collections::HashSet;

use wgpu::Device;

use crate::engine::{BindType, Engine, Error, ImageFormat, ShaderId};

pub const PATHTAG_REDUCE_WG: u32 = 256;
pub const PATH_BBOX_WG: u32 = 256;
pub const PATH_COARSE_WG: u32 = 256;
pub const PATH_DRAWOBJ_WG: u32 = 256;
pub const CLIP_REDUCE_WG: u32 = 256;

macro_rules! shader {
    ($name:expr) => {&{
        let shader = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shader/",
            $name,
            ".wgsl"
        ));
        #[cfg(feature = "hot_reload")]
        let shader = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shader/",
            $name,
            ".wgsl"
        ))
        .unwrap_or_else(|e| {
            eprintln!(
                "Failed to read shader {name}, error falling back to version at compilation time. Error: {e:?}",
                name = $name
            );
            shader.to_string()
        });
        shader
    }};
}

pub struct Shaders {
    pub pathtag_reduce: ShaderId,
    pub pathtag_scan: ShaderId,
    pub path_coarse: ShaderId,
    pub backdrop: ShaderId,
    pub fine: ShaderId,
}

pub fn init_shaders(device: &Device, engine: &mut Engine) -> Result<Shaders, Error> {
    let imports = SHARED_SHADERS
        .iter()
        .copied()
        .collect::<std::collections::HashMap<_, _>>();
    let empty = HashSet::new();
    let pathtag_reduce = engine.add_shader(
        device,
        "pathtag_reduce",
        preprocess::preprocess(shader!("pathtag_reduce"), &empty, &imports).into(),
        &[BindType::Uniform, BindType::BufReadOnly, BindType::Buffer],
    )?;
    let pathtag_scan = engine.add_shader(
        device,
        "pathtag_scan",
        preprocess::preprocess(shader!("pathtag_scan"), &empty, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
        ],
    )?;
    let path_coarse_config = HashSet::new();
    // path_coarse_config.add("cubics_out");

    let path_coarse = engine.add_shader(
        device,
        "path_coarse",
        preprocess::preprocess(shader!("path_coarse"), &path_coarse_config, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Buffer,
            BindType::Buffer,
        ],
    )?;
    let backdrop = engine.add_shader(
        device,
        "backdrop",
        preprocess::preprocess(shader!("backdrop"), &empty, &imports).into(),
        &[BindType::Uniform, BindType::Buffer],
    )?;
    let fine = engine.add_shader(
        device,
        "fine",
        preprocess::preprocess(shader!("fine"), &empty, &imports).into(),
        &[
            BindType::Uniform,
            BindType::BufReadOnly,
            BindType::BufReadOnly,
            BindType::Image(ImageFormat::Rgba8),
        ],
    )?;
    Ok(Shaders {
        pathtag_reduce,
        pathtag_scan,
        path_coarse,
        backdrop,
        fine,
    })
}

macro_rules! shared_shader {
    ($name:expr) => {
        (
            $name,
            include_str!(concat!("../shader/shared/", $name, ".wgsl")),
        )
    };
}

const SHARED_SHADERS: &[(&str, &str)] = &[
    shared_shader!("config"),
    shared_shader!("pathtag"),
    shared_shader!("segment"),
];
