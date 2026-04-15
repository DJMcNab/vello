// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Random text stress scene.

use core::fmt;
use glifo::Glyph;
use parley::FontFamily;
use parley::{
    Alignment, AlignmentOptions, FontContext, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem, StyleProperty,
};
use vello_cpu::color::palette::css;
use vello_cpu::color::{AlphaColor, Srgb};
use vello_cpu::{Level, Pixmap, RenderContext, RenderMode, RenderSettings, Resources};

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColorBrush {
    color: AlphaColor<Srgb>,
}

impl Default for ColorBrush {
    fn default() -> Self {
        Self { color: css::WHITE }
    }
}

#[cfg(target_arch = "wasm32")]
const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");

struct Segment {
    layout: Layout<ColorBrush>,
    x: f32,
    y: f32,
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Segment")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish_non_exhaustive()
    }
}

fn new_font_context() -> FontContext {
    #[cfg(not(target_arch = "wasm32"))]
    {
        FontContext::new()
    }

    #[cfg(target_arch = "wasm32")]
    {
        let mut font_cx = FontContext::new();
        font_cx
            .collection
            .register_fonts(ROBOTO_FONT.to_vec().into(), None);
        font_cx
    }
}

fn build_segment(layout_cx: &mut LayoutContext<ColorBrush>, font_cx: &mut FontContext) -> Segment {
    let text = "Hello World";
    let color = css::YELLOW;
    let font_size = 10.;
    let x = 10.;
    let y = 20.;

    let mut builder = layout_cx.ranged_builder(font_cx, text, 1.0, true);
    builder.push_default(FontFamily::parse("Roboto").unwrap());
    builder.push_default(StyleProperty::FontSize(font_size));
    builder.push_default(StyleProperty::Brush(ColorBrush { color }));

    let mut layout: Layout<ColorBrush> = builder.build(text);
    let max_advance = Some(600.0);
    layout.break_all_lines(max_advance);
    layout.align(max_advance, Alignment::Start, AlignmentOptions::default());

    Segment { layout, x, y }
}

fn render_segment(
    ctx: &mut RenderContext,
    resources: &mut Resources,
    layout: &Layout<ColorBrush>,
    offset_x: f32,
    offset_y: f32,

    hinting_enabled: bool,
) {
    for line in layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                render_glyph_run(
                    ctx,
                    resources,
                    &glyph_run,
                    offset_x,
                    offset_y,
                    hinting_enabled,
                );
            }
        }
    }
}

fn render_glyph_run(
    ctx: &mut RenderContext,
    resources: &mut Resources,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    offset_x: f32,
    offset_y: f32,

    hinting_enabled: bool,
) {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let glyphs = glyph_run.glyphs().map(move |glyph| {
        let glyph_x = offset_x + run_x + glyph.x;
        let glyph_y = offset_y + run_y - glyph.y;
        run_x += glyph.advance;

        Glyph {
            id: glyph.id as u32,
            x: glyph_x,
            y: glyph_y,
        }
    });

    let run = glyph_run.run();
    let style = glyph_run.style();
    ctx.set_paint(style.brush.color);
    ctx.glyph_run(resources, run.font())
        .font_size(run.font_size())
        .hint(hinting_enabled)
        .atlas_cache(false)
        .fill_glyphs(glyphs);
}

fn main() {
    let settings = RenderSettings {
        level: Level::new(),
        num_threads: 0,
        // Required for Gamma Correction to be enabled.
        render_mode: RenderMode::OptimizeQuality,
    };

    let mut ctx = RenderContext::new_with(100, 100, settings);
    let mut resources = Resources::new();
    let mut layout_cx = LayoutContext::new();
    let mut font_cx = new_font_context();
    let segment = build_segment(&mut layout_cx, &mut font_cx);
    render_segment(
        &mut ctx,
        &mut resources,
        &segment.layout,
        segment.x,
        segment.y,
        false,
    );

    ctx.flush();

    let mut pixmap_1 = Pixmap::new(100, 100);
    ctx.render_to_pixmap(&mut resources, &mut pixmap_1);

    let png_1 = pixmap_1.into_png().unwrap();
    std::fs::write("example_basic1.png", png_1).unwrap();

    ctx.reset();
}
