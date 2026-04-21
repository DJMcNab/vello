// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text input value

use glifo::Glyph;
use parley::FontFamily;
use parley::fontique::{Collection, CollectionOptions, SourceCache};
use parley::{
    Alignment, AlignmentOptions, FontContext, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem, StyleProperty,
};
use std::path::Path;
use std::sync::Arc;
use vello_cpu::color::palette::css;
use vello_cpu::color::{AlphaColor, Srgb};
use vello_cpu::kurbo::{Affine, Rect, Stroke};
use vello_cpu::peniko::{Color, Extend, ImageQuality, ImageSampler};
use vello_cpu::{
    Image, ImageSource, Level, Pixmap, RenderContext, RenderMode, RenderSettings, Resources,
};

const TEXT: &str =
    "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nSed ornare arcu lectus.\nwwwwwwww";

fn main() {
    let mut layout_cx = LayoutContext::new();
    let mut font_cx = new_font_context();
    // We assume that we're on the same device.
    let outputs_folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("outputs");

    tidy_outputs(&outputs_folder);

    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo.png",
        TestCase::default(),
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_hinted.png",
        TestCase {
            hinting_enabled: true,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_24.png",
        TestCase {
            font_size: 24.,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_gamma.png",
        TestCase {
            gamma_correction: true,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_dark_bg.png",
        TestCase {
            foreground_color: css::WHITE,
            background_color: css::BLACK,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_dark_bg_gamma.png",
        TestCase {
            gamma_correction: true,
            foreground_color: css::WHITE,
            background_color: css::BLACK,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_dark_bg_24.png",
        TestCase {
            font_size: 24.,
            foreground_color: css::WHITE,
            background_color: css::BLACK,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_blue_on_green.png",
        TestCase {
            font_size: 12.,
            foreground_color: css::BLUE,
            background_color: css::LIME,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_blue_on_green_gamma.png",
        TestCase {
            font_size: 12.,
            gamma_correction: true,
            foreground_color: css::BLUE,
            background_color: css::LIME,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_red_on_green.png",
        TestCase {
            font_size: 12.,
            foreground_color: css::RED,
            background_color: css::LIME,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "arimo_red_on_green_gamma.png",
        TestCase {
            font_size: 12.,
            gamma_correction: true,
            foreground_color: css::RED,
            background_color: css::LIME,
            ..TestCase::default()
        },
    );

    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "0.5_black.png",
        TestCase {
            font_size: 12.,
            foreground_color: css::BLACK.with_alpha(0.5),
            background_color: css::WHITE,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Arimo").unwrap(),
        "0.5_black_gamma.png",
        TestCase {
            font_size: 12.,
            gamma_correction: true,
            foreground_color: css::BLACK.with_alpha(0.5),
            background_color: css::WHITE,
            ..TestCase::default()
        },
    );

    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto.png",
        TestCase::default(),
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_hinted.png",
        TestCase {
            hinting_enabled: true,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_rotated.png",
        TestCase {
            rotation: -0.5f64.to_radians(),
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_hinted_rotated.png",
        TestCase {
            hinting_enabled: true,
            rotation: -0.5f64.to_radians(),
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_24.png",
        TestCase {
            font_size: 24.,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_gamma.png",
        TestCase {
            gamma_correction: true,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_dark_bg.png",
        TestCase {
            foreground_color: css::WHITE,
            background_color: css::BLACK,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_dark_bg_gamma.png",
        TestCase {
            gamma_correction: true,
            foreground_color: css::WHITE,
            background_color: css::BLACK,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_dark_bg_24.png",
        TestCase {
            font_size: 24.,
            foreground_color: css::WHITE,
            background_color: css::BLACK,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_blue_on_green.png",
        TestCase {
            font_size: 12.,
            foreground_color: css::BLUE,
            background_color: css::LIME,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_blue_on_green_gamma.png",
        TestCase {
            font_size: 12.,
            gamma_correction: true,
            foreground_color: css::BLUE,
            background_color: css::LIME,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_red_on_green.png",
        TestCase {
            font_size: 12.,
            foreground_color: css::RED,
            background_color: css::LIME,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_red_on_green_gamma.png",
        TestCase {
            font_size: 12.,
            gamma_correction: true,
            foreground_color: css::RED,
            background_color: css::LIME,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_0.5_black.png",
        TestCase {
            font_size: 12.,
            foreground_color: css::BLACK.with_alpha(0.5),
            background_color: css::WHITE,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_0.5_black_gamma.png",
        TestCase {
            font_size: 12.,
            gamma_correction: true,
            foreground_color: css::BLACK.with_alpha(0.5),
            background_color: css::WHITE,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_stem_darkening.png",
        TestCase {
            stem_darkening: true,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_stem_darkening_gamma.png",
        TestCase {
            stem_darkening: true,
            gamma_correction: true,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_stem_darkening_24.png",
        TestCase {
            stem_darkening: true,
            font_size: 24.,
            ..TestCase::default()
        },
    );
    text_case(
        &mut layout_cx,
        &mut font_cx,
        &outputs_folder,
        FontFamily::parse("Roboto").unwrap(),
        "roboto_stem_darkening_dark_bg.png",
        TestCase {
            stem_darkening: true,
            foreground_color: css::WHITE,
            background_color: css::BLACK,
            ..TestCase::default()
        },
    );
}

struct TestCase {
    hinting_enabled: bool,
    gamma_correction: bool,
    foreground_color: Color,
    background_color: Color,
    font_size: f32,
    /// Rotation angle in radians, applied about the center of the canvas.
    rotation: f64,
    /// Emulated stem darkening: draws the glyph outline as an additional stroke.
    /// Stroke width is `min(0.3, 0.015125 * font_size)`.
    stem_darkening: bool,
}

impl Default for TestCase {
    fn default() -> Self {
        Self {
            hinting_enabled: false,
            gamma_correction: false,
            foreground_color: Color::BLACK,
            background_color: Color::WHITE,
            font_size: 12.,
            rotation: 0.,
            stem_darkening: false,
        }
    }
}

fn text_case(
    layout_cx: &mut LayoutContext<ColorBrush>,
    font_cx: &mut FontContext,
    outputs_folder: &Path,
    font: FontFamily<'static>,
    name: &str,
    args: TestCase,
) {
    let TestCase {
        hinting_enabled,
        gamma_correction,
        foreground_color,
        background_color,
        font_size,
        rotation,
        stem_darkening,
    } = args;

    let layout = build_layout(layout_cx, font_cx, font, foreground_color, font_size);
    let width = layout.width().ceil() + 20.;
    let height = layout.height().ceil() + 10.;

    // Ideally, we'd reuse the same render context, but it isn't possible to resize them.
    let settings = RenderSettings {
        level: Level::new(),
        num_threads: 0,
        // Required for gamma correction to be enabled.
        render_mode: RenderMode::OptimizeQuality,
    };
    let mut ctx = RenderContext::new_with(width as u16, height as u16, settings);
    // The docs in basic.rs require us to recreate the resources when we make a new ctx.
    let mut resources = Resources::new();
    ctx.reset();
    if gamma_correction {
        ctx.set_gamma_correction(true);
    }

    ctx.set_paint(background_color);
    ctx.fill_rect(&Rect::from_points(
        (-5., -5.),
        (width as f64 + 5., height as f64 + 10.),
    ));

    if rotation != 0. {
        let cx = width as f64 / 2.;
        let cy = height as f64 / 2.;
        ctx.set_transform(
            Affine::translate((cx, cy)) * Affine::rotate(rotation) * Affine::translate((-cx, -cy)),
        );
    }
    render_layout(
        &mut ctx,
        &mut resources,
        &layout,
        10.,
        5.,
        hinting_enabled,
        stem_darkening,
    );
    ctx.flush();

    let mut pixmap = Pixmap::new(width as u16, height as u16);
    ctx.render_to_pixmap(&mut resources, &mut pixmap);

    let path = outputs_folder.join(name);
    std::fs::write(&path, pixmap.clone().into_png().unwrap()).unwrap();
    let abs_path = path.canonicalize().unwrap();
    let url = format!("file://{}", abs_path.display());

    const SCALE: f64 = 4.;
    let scaled_width = (width as f64 * SCALE) as u16;
    let scaled_height = (height as f64 * SCALE) as u16;
    let scaled_settings = RenderSettings {
        level: Level::new(),
        num_threads: 0,
        render_mode: RenderMode::OptimizeQuality,
    };
    let mut scaled_ctx = RenderContext::new_with(scaled_width, scaled_height, scaled_settings);
    let mut scaled_resources = Resources::new();
    scaled_ctx.reset();
    let image = Image {
        image: ImageSource::Pixmap(Arc::new(pixmap)),
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality: ImageQuality::Low,
            ..Default::default()
        },
    };
    scaled_ctx.set_paint(image);
    scaled_ctx.set_paint_transform(Affine::scale(SCALE));
    scaled_ctx.fill_rect(&Rect::from_points(
        (0., 0.),
        (scaled_width as f64, scaled_height as f64),
    ));
    scaled_ctx.flush();
    let mut scaled_pixmap = Pixmap::new(scaled_width, scaled_height);
    scaled_ctx.render_to_pixmap(&mut scaled_resources, &mut scaled_pixmap);
    let scaled_path = outputs_folder.join("scaled").join(name);
    std::fs::write(&scaled_path, scaled_pixmap.into_png().unwrap()).unwrap();
    let scaled_url = format!("file://{}", scaled_path.canonicalize().unwrap().display());

    eprintln!(
        // Render as a terminal hyperlink; this avoids issues if the path to the workspace root contains a space.
        "Wrote output image for \x1b]8;;{url}\x1b\\{name}\x1b]8;;\x1b\\ [\x1b]8;;{scaled_url}\x1b\\scaled\x1b]8;;\x1b\\]. Dimensions: {}x{}",
        width, height
    );
}

fn tidy_outputs(outputs_folder: &std::path::PathBuf) {
    for entry in std::fs::read_dir(outputs_folder).unwrap().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "png") {
            std::fs::remove_file(&path).unwrap();
        }
    }
    let scaled_folder = outputs_folder.join("scaled");
    if scaled_folder.is_dir() {
        for entry in std::fs::read_dir(&scaled_folder).unwrap().flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "png") {
                std::fs::remove_file(&path).unwrap();
            }
        }
    }
}

fn new_font_context() -> FontContext {
    let mut font_cx = FontContext {
        collection: Collection::new(CollectionOptions {
            shared: false,
            system_fonts: false,
        }),
        source_cache: SourceCache::default(),
    };

    let fonts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fonts");
    for font_file in ["SourceSerif4Variable-Roman.ttf", "Arimo.ttf"] {
        let path = fonts_dir.join(font_file);
        match std::fs::read(&path) {
            Ok(data) => {
                font_cx.collection.register_fonts(data.into(), None);
            }
            Err(e) => eprintln!(
                "Warning: could not load {font_file}: {e}\n\
                 See fonts/README.md for download instructions."
            ),
        }
    }
    let roboto_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../examples/assets/roboto/Roboto-Regular.ttf");
    match std::fs::read(&roboto_path) {
        Ok(data) => {
            font_cx.collection.register_fonts(data.into(), None);
        }
        Err(e) => eprintln!("Warning: could not load Roboto-Regular.ttf: {e}"),
    }
    font_cx
}

fn build_layout(
    layout_cx: &mut LayoutContext<ColorBrush>,
    font_cx: &mut FontContext,
    font: FontFamily<'static>,
    foreground_color: Color,
    font_size: f32,
) -> Layout<ColorBrush> {
    let text = TEXT;

    let mut builder = layout_cx.ranged_builder(font_cx, text, 1.0, true);
    builder.push_default(font);
    builder.push_default(StyleProperty::FontSize(font_size));
    builder.push_default(StyleProperty::Brush(ColorBrush {
        color: foreground_color,
    }));

    let mut layout: Layout<ColorBrush> = builder.build(text);
    let max_advance = None;
    layout.break_all_lines(max_advance);
    layout.align(max_advance, Alignment::Start, AlignmentOptions::default());

    layout
}

fn render_layout(
    ctx: &mut RenderContext,
    resources: &mut Resources,
    layout: &Layout<ColorBrush>,
    offset_x: f32,
    offset_y: f32,
    hinting_enabled: bool,
    stem_darkening: bool,
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
                    stem_darkening,
                );
            } else {
                unreachable!()
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
    stem_darkening: bool,
) {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let glyphs: Vec<Glyph> = glyph_run
        .glyphs()
        .map(move |glyph| {
            let glyph_x = offset_x + run_x + glyph.x;
            let glyph_y = offset_y + run_y - glyph.y;
            run_x += glyph.advance;

            Glyph {
                id: glyph.id as u32,
                x: glyph_x,
                y: glyph_y,
            }
        })
        .collect();

    let run = glyph_run.run();
    let style = glyph_run.style();
    ctx.set_paint(style.brush.color);

    if stem_darkening {
        let stroke_width = (0.015125_f32 * run.font_size()).min(0.3);
        ctx.set_stroke(Stroke {
            width: stroke_width as f64,
            ..Default::default()
        });
        ctx.glyph_run(resources, run.font())
            .font_size(run.font_size())
            .hint(hinting_enabled)
            .atlas_cache(false)
            .stroke_glyphs(glyphs.iter().copied());
    }

    ctx.glyph_run(resources, run.font())
        .font_size(run.font_size())
        .hint(hinting_enabled)
        .atlas_cache(false)
        .fill_glyphs(glyphs.into_iter());
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColorBrush {
    color: AlphaColor<Srgb>,
}

impl Default for ColorBrush {
    fn default() -> Self {
        Self { color: css::WHITE }
    }
}
