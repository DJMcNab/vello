//! Take an encoded scene and create a graph to render it

use bytemuck::{Pod, Zeroable};

use crate::{
    engine::{BufProxy, ImageFormat, ImageProxy, Recording, ResourceProxy},
    shaders::{self, Shaders},
    Scene,
};

const TAG_MONOID_SIZE: u64 = 12;
const TAG_MONOID_FULL_SIZE: u64 = 20;
const PATH_BBOX_SIZE: u64 = 24;
const CUBIC_SIZE: u64 = 48;
const DRAWMONOID_SIZE: u64 = 16;
const MAX_DRAWINFO_SIZE: u64 = 44;
const CLIP_BIC_SIZE: u64 = 8;
const CLIP_EL_SIZE: u64 = 32;
const CLIP_INP_SIZE: u64 = 8;
const CLIP_BBOX_SIZE: u64 = 16;
const PATH_SIZE: u64 = 32;
const DRAW_BBOX_SIZE: u64 = 16;
const BIN_HEADER_SIZE: u64 = 8;
const TILE_SIZE: u64 = 8;
const SEGMENT_SIZE: u64 = 24;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,
    target_width: u32,
    target_height: u32,
    n_drawobj: u32,
    n_path: u32,
    n_clip: u32,
    bin_data_start: u32,
    pathtag_base: u32,
    pathdata_base: u32,
    drawtag_base: u32,
    drawdata_base: u32,
    transform_base: u32,
    linewidth_base: u32,
}

fn size_to_words(byte_size: usize) -> u32 {
    (byte_size / std::mem::size_of::<u32>()) as u32
}

#[allow(unused)]
pub(crate) fn render(scene: &Scene, shaders: &Shaders) -> (Recording, ImageProxy) {
    let mut recording = Recording::default();
    let data = scene.data();
    let n_pathtag = data.path_tags.len();
    let pathtag_padded = align_up(n_pathtag, 4 * shaders::PATHTAG_REDUCE_WG);
    let pathtag_wgs = pathtag_padded / (4 * shaders::PATHTAG_REDUCE_WG as usize);
    let mut scene: Vec<u8> = Vec::with_capacity(pathtag_padded);
    let pathtag_base = size_to_words(scene.len());
    scene.extend(bytemuck::cast_slice(&data.path_tags));
    scene.resize(pathtag_padded, 0);
    let pathdata_base = size_to_words(scene.len());
    scene.extend(&data.path_data);

    let config = Config {
        width_in_tiles: 64,
        height_in_tiles: 64,
        target_width: 64 * 16,
        target_height: 64 * 16,
        pathtag_base,
        pathdata_base,
        ..Default::default()
    };
    let scene_buf = recording.upload("scene", scene);
    let config_buf = recording.upload_uniform("config", bytemuck::bytes_of(&config));

    let reduced_buf = BufProxy::new(pathtag_wgs as u64 * TAG_MONOID_SIZE, "reduced_buf");
    // TODO: really only need pathtag_wgs - 1
    recording.dispatch(
        shaders.pathtag_reduce,
        (pathtag_wgs as u32, 1, 1),
        [config_buf, scene_buf, reduced_buf],
    );

    let tagmonoid_buf = BufProxy::new(
        pathtag_wgs as u64 * shaders::PATHTAG_REDUCE_WG as u64 * TAG_MONOID_SIZE,
        "tagmonoid_buf",
    );
    recording.dispatch(
        shaders.pathtag_scan,
        (pathtag_wgs as u32, 1, 1),
        [config_buf, scene_buf, reduced_buf, tagmonoid_buf],
    );

    let path_coarse_wgs =
        (n_pathtag as u32 + shaders::PATH_COARSE_WG - 1) / shaders::PATH_COARSE_WG;
    // TODO: more principled size calc
    let tiles_buf = BufProxy::new(4097 * 8, "tiles_buf");
    let segments_buf = BufProxy::new(256 * 24, "segments_buf");
    recording.clear_all(tiles_buf);
    recording.dispatch(
        shaders.path_coarse,
        (path_coarse_wgs, 1, 1),
        [
            config_buf,
            scene_buf,
            tagmonoid_buf,
            tiles_buf,
            segments_buf,
        ],
    );
    recording.dispatch(
        shaders.backdrop,
        (config.height_in_tiles, 1, 1),
        [config_buf, tiles_buf],
    );
    let out_buf_size = config.width_in_tiles * config.height_in_tiles * 256;
    let out_image = ImageProxy::new(
        config.target_width,
        config.target_height,
        ImageFormat::Rgba8,
    );
    recording.dispatch(
        shaders.fine,
        (config.width_in_tiles, config.height_in_tiles, 1),
        [
            ResourceProxy::Buf(config_buf),
            ResourceProxy::Buf(tiles_buf),
            ResourceProxy::Buf(segments_buf),
            ResourceProxy::Image(out_image),
        ],
    );

    (recording, out_image)
}

pub fn align_up(len: usize, alignment: u32) -> usize {
    len + (len.wrapping_neg() & (alignment as usize - 1))
}
