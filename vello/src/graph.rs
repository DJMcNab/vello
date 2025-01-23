// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello's Render Graph.
//!
//! The core technology of Vello is a vector graphics rasteriser, which converts from a [scene description][Scene] to a rendered texture.
//! This by itself does not support many advanced visual effects, such as blurs, as they are incompatible with the parallelism it exploits.
//! These are instead built on top of this core pipeline, which schedules blurs and other visual effects,
//! alongside the core vector graphics rendering.
//!
// Is this true?: //! Most users of Vello should expect to use this more capable API.
//! If you have your own render graph or otherwise need greater control, the [rasteriser][Renderer] can be used standalone.
//!
//! ## Core Concepts
//!
//! The render graph consists of a few primary types:
//! - `Vello` is the core renderer type. Your application should generally only ever have one of these.
//! - A [`Painting`] is a persistent reference counted handle to a texture on the GPU.
//! - The `Gallery`
//!
//! ## Test
//!
//! This enables the use of image filters among other things.

#![warn(
    missing_debug_implementations,
    elided_lifetimes_in_paths,
    single_use_lifetimes,
    unnameable_types,
    unreachable_pub,
    clippy::return_self_not_must_use,
    clippy::cast_possible_truncation,
    clippy::missing_assert_message,
    clippy::shadow_unrelated,
    clippy::missing_panics_doc,
    clippy::print_stderr,
    clippy::use_self,
    clippy::match_same_arms,
    clippy::missing_errors_doc,
    clippy::todo,
    clippy::partial_pub_fields,
    reason = "Lint set, currently allowed crate-wide"
)]

mod filters;
mod runner;

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    num::Wrapping,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc, LazyLock,
    },
};

use filters::BlurPipeline;
use peniko::{kurbo::Affine, Blob, Brush, Extend, Image, ImageFormat, ImageQuality};
use wgpu::{Texture, TextureView};

use crate::{Renderer, Scene};
pub use runner::RenderDetails;

// --- MARK: Public API ---

pub struct Vello {
    cache: HashMap<PaintingId, (Arc<Texture>, TextureView, Generation)>,
    renderer: Renderer,
    blur: BlurPipeline,
}

impl Vello {
    pub fn new(device: &wgpu::Device, options: crate::RendererOptions) -> crate::Result<Self> {
        Ok(Self {
            cache: Default::default(),
            renderer: Renderer::new(device, options)?,
            blur: BlurPipeline::new(device),
        })
    }
}

impl Debug for Vello {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vello")
            .field("cache", &self.cache)
            .field("renderer", &"elided")
            .field("blur", &self.blur)
            .finish()
    }
}

/// A partial render graph.
///
/// There is expected to be one Gallery per thread.
pub struct Gallery {
    id: GalleryId,
    label: Cow<'static, str>,
    generation: Generation,
    incoming_deallocations: Receiver<PaintingId>,
    deallocator: Sender<PaintingId>,
    paintings: HashMap<PaintingId, (PaintingSource, Generation)>,
}

impl Debug for Gallery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}({})", self.id, self.label))
    }
}

/// A handle to an image managed by the renderer.
///
/// This resource is reference counted, and corresponding resources
/// are freed when rendering operations occur.
#[derive(Clone)]
pub struct Painting {
    inner: Arc<PaintingInner>,
}

impl Debug for Painting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}({})", self.inner.id, self.inner.label))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputSize {
    pub width: u32,
    pub height: u32,
}

impl Gallery {
    pub fn new(label: impl Into<Cow<'static, str>>) -> Self {
        let id = GalleryId::next();
        Self::new_inner(id, label.into())
    }
    pub fn new_anonymous(prefix: &'static str) -> Self {
        let id = GalleryId::next();
        let label = format!("{prefix}-{id:02}", id = id.0);
        Self::new_inner(id, label.into())
    }
    pub fn gc(&mut self) {
        let mut made_change = false;
        loop {
            let try_recv = self.incoming_deallocations.try_recv();
            let dealloc = match try_recv {
                Ok(dealloc) => dealloc,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    unreachable!("We store a sender alongside the receiver")
                }
            };
            self.paintings.remove(&dealloc);
            made_change = true;
        }
        if made_change {
            self.generation.nudge();
        }
    }
    fn new_inner(id: GalleryId, label: Cow<'static, str>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            id,
            label,
            generation: Generation::default(),
            paintings: HashMap::default(),
            deallocator: tx,
            incoming_deallocations: rx,
        }
    }
}

#[derive(Clone)]
/// A description of a new painting, used in [`Gallery::create_painting`].
#[derive(Debug)]
pub struct PaintingDescriptor {
    pub label: Cow<'static, str>,
    pub usages: wgpu::TextureUsages,
    /// Extend mode in the horizontal direction.
    pub x_extend: Extend,
    /// Extend mode in the vertical direction.
    pub y_extend: Extend,
    // pub mipmaps
}

impl Gallery {
    #[must_use]
    pub fn create_painting(
        &mut self,
        // Not &PaintingDescriptor because `label` might be owned
        desc: PaintingDescriptor,
    ) -> Painting {
        let PaintingDescriptor {
            label,
            usages,
            x_extend,
            y_extend,
        } = desc;
        Painting {
            inner: Arc::new(PaintingInner {
                label,
                deallocator: self.deallocator.clone(),
                id: PaintingId::next(),
                gallery_id: self.id,
                usages,
                x_extend,
                y_extend,
            }),
        }
    }

    /// The painting must have [been created for](Self::create_painting) this gallery.
    ///
    /// This restriction ensures that when the painting is dropped, its resources are properly freed.
    pub fn paint(&mut self, painting: &Painting) -> Option<Painter<'_>> {
        if painting.inner.gallery_id != self.id {
            // TODO: Return error about mismatched Gallery.
            return None;
        }
        self.generation.nudge();
        Some(Painter {
            gallery: self,
            painting: painting.inner.id,
        })
    }
}

/// Defines how a [`Painting`] will be drawn.
#[derive(Debug)]
pub struct Painter<'a> {
    gallery: &'a mut Gallery,
    painting: PaintingId,
}

impl Painter<'_> {
    pub fn as_image(self, image: Image) {
        self.insert(PaintingSource::Image(image));
    }
    // /// From must have the `COPY_SRC` usage.
    // pub fn as_subregion(self, from: Painting, x: u32, y: u32, width: u32, height: u32) {
    //     self.insert(PaintingSource::Region {
    //         painting: from,
    //         x,
    //         y,
    //         size: OutputSize { width, height },
    //     });
    // }
    // pub fn with_mipmaps(self, from: Painting) {
    //     self.insert(PaintingSource::WithMipMaps(from));
    // }
    pub fn as_scene(self, scene: Canvas, of_dimensions: OutputSize) {
        self.insert(PaintingSource::Canvas(scene, of_dimensions));
    }

    pub fn as_blur(self, from: Painting) {
        self.insert(PaintingSource::Blur(from));
    }

    fn insert(self, new_source: PaintingSource) {
        match self.gallery.paintings.entry(self.painting) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let entry = entry.get_mut();
                entry.0 = new_source;
                entry.1.nudge();
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert((new_source, Generation::default()));
            }
        };
    }
}

// --- MARK: Internal types ---

/// The shared elements of a `Painting`.
///
/// A painting's identity is its heap allocation; most of
/// the resources are owned by its [`Gallery`].
/// This only stores peripheral information.
struct PaintingInner {
    id: PaintingId,
    deallocator: Sender<PaintingId>,
    label: Cow<'static, str>,
    gallery_id: GalleryId,
    usages: wgpu::TextureUsages,
    x_extend: Extend,
    y_extend: Extend,
}

impl Drop for PaintingInner {
    fn drop(&mut self) {
        // Ignore the possibility that the corresponding gallery has already been dropped.
        let _ = self.deallocator.send(self.id);
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct PaintingId(u64);

impl Debug for PaintingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("#{}", self.0))
    }
}

impl PaintingId {
    fn next() -> Self {
        static PAINTING_IDS: AtomicU64 = AtomicU64::new(0);
        Self(PAINTING_IDS.fetch_add(1, Ordering::Relaxed))
    }
}

/// The id of a gallery.
///
/// The debug label is provided for error messaging when
/// a painting is used with the wrong gallery.
#[derive(Clone, Copy, PartialEq, Eq)]
struct GalleryId(u64);

impl Debug for GalleryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("#{}", self.0))
    }
}

impl GalleryId {
    fn next() -> Self {
        static GALLERY_IDS: AtomicU64 = AtomicU64::new(1);
        // Overflow handling: u64 incremented so can never overflow
        let id = GALLERY_IDS.fetch_add(1, Ordering::Relaxed);
        Self(id)
    }
}

#[derive(Debug)]
enum PaintingSource {
    Image(Image),
    Canvas(Canvas, OutputSize),
    Blur(Painting),
    // WithMipMaps(Painting),
    // Region {
    //     painting: Painting,
    //     x: u32,
    //     y: u32,
    //     size: OutputSize,
    // },
}

#[derive(Default, Debug, PartialEq, Eq, Clone)]
// Not copy because the identity is important; don't want to modify an accidental copy
struct Generation(Wrapping<u32>);

impl Generation {
    fn nudge(&mut self) {
        self.0 += 1;
    }
}

// --- MARK: Model ---
// A model of the rest of Vello.

/// A single Scene, potentially containing paintings.
pub struct Canvas {
    scene: Box<Scene>,
    paintings: HashMap<u64, Painting>,
}

#[derive(Debug)]
/// Created using [`Canvas::new_image`].
pub struct PaintingConfig {
    image: Image,
}

impl PaintingConfig {
    fn new(painting: &Painting, width: u16, height: u16) -> Self {
        // Create a fake Image, with an empty Blob. We can re-use the allocation between these.
        static EMPTY_ARC: LazyLock<Arc<[u8; 0]>> = LazyLock::new(|| Arc::new([]));
        let data = Blob::new(EMPTY_ARC.clone());
        let mut image = Image::new(data, ImageFormat::Rgba8, width.into(), height.into());
        image.x_extend = painting.inner.x_extend;
        image.y_extend = painting.inner.y_extend;
        Self { image }
    }
    pub fn brush(self) -> Brush {
        Brush::Image(self.image)
    }
    pub fn image(&self) -> &Image {
        &self.image
    }
    /// Builder method for setting a hint for the desired image [quality](ImageQuality)
    /// when rendering.
    #[must_use]
    pub fn with_quality(self, quality: ImageQuality) -> Self {
        Self {
            image: self.image.with_quality(quality),
        }
    }

    /// Returns the image with the alpha multiplier set to `alpha`.
    #[must_use]
    #[track_caller]
    pub fn with_alpha(self, alpha: f32) -> Self {
        Self {
            image: self.image.with_alpha(alpha),
        }
    }

    /// Returns the image with the alpha multiplier multiplied again by `alpha`.
    /// The behaviour of this transformation is undefined if `alpha` is negative.
    #[must_use]
    #[track_caller]
    pub fn multiply_alpha(self, alpha: f32) -> Self {
        Self {
            image: self.image.multiply_alpha(alpha),
        }
    }
}

impl From<Scene> for Canvas {
    fn from(value: Scene) -> Self {
        Self::from_scene(Box::new(value))
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

impl Canvas {
    pub fn new() -> Self {
        Self::from_scene(Box::<Scene>::default())
    }
    pub fn from_scene(scene: Box<Scene>) -> Self {
        Self {
            scene,
            paintings: HashMap::default(),
        }
    }
    pub fn new_image(&mut self, painting: Painting, width: u16, height: u16) -> PaintingConfig {
        let config = PaintingConfig::new(&painting, width, height);
        self.override_image(&config.image, painting);
        config
    }

    #[doc(alias = "image")]
    pub fn draw_painting(
        &mut self,
        painting: Painting,
        width: u16,
        height: u16,
        transform: Affine,
    ) {
        let image = self.new_image(painting, width, height);
        self.scene.draw_image(&image.image, transform);
    }

    #[deprecated(note = "Prefer `draw_painting` for greater efficiency")]
    pub fn draw_image(&mut self, image: &Image, transform: Affine) {
        self.scene.draw_image(image, transform);
    }

    pub fn override_image(&mut self, image: &Image, painting: Painting) {
        self.paintings.insert(image.data.id(), painting);
    }
}

impl Deref for Canvas {
    type Target = Scene;

    fn deref(&self) -> &Self::Target {
        &self.scene
    }
}
impl DerefMut for Canvas {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.scene
    }
}

impl Debug for Canvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Canvas")
            .field("scene", &"elided")
            .field("paintings", &self.paintings)
            .finish()
    }
}

// --- MARK: Musings ---

/// When making an image filter graph, we need to know a few things:
///
/// 1) The Scene to draw.
/// 2) The resolution of the filter target (i.e. input image).
/// 3) The resolution of the output image.
///
/// The scene to draw might be a texture from a previous step or externally provided.
/// The resolution of the input might change depending on the resolution of the
/// output, because of scaling/rotation/skew.
#[derive(Debug)]
pub struct Thinking;

/// What threading model do we want. Requirements:
/// 1) Creating scenes on different threads should be possible.
/// 2) Scenes created on different threads should be able to use filter effects.
/// 3) We should only upload each CPU side image once.
#[derive(Debug)]
pub struct Threading;

/// Question: What do we win from backpropogating render sizes?
/// Answer: Image sampling
///
/// Conclusion: Special handling of "automatic" scene sizing to
/// render multiple times if needed.
///
/// Conclusion: Two phase approach, backpropogating from every scene
/// with a defined size?
#[derive(Debug)]
pub struct ThinkingAgain;

/// Do we want custom fully graph nodes?
/// Answer for now: No?
#[derive(Debug)]
pub struct Scheduling;
