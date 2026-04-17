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
use vello_cpu::color::palette::css;
use vello_cpu::color::{AlphaColor, Srgb};
use vello_cpu::kurbo::Rect;
use vello_cpu::peniko::Color;
use vello_cpu::{Level, Pixmap, RenderContext, RenderMode, RenderSettings, Resources};

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
}

struct TestCase {
    hinting_enabled: bool,
    gamma_correction: bool,
    foreground_color: Color,
    background_color: Color,
    font_size: f32,
}

impl Default for TestCase {
    fn default() -> Self {
        Self {
            hinting_enabled: false,
            gamma_correction: false,
            foreground_color: Color::BLACK,
            background_color: Color::WHITE,
            font_size: 12.,
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

    render_layout(&mut ctx, &mut resources, &layout, 10., 5., hinting_enabled);
    ctx.flush();

    let mut pixmap = Pixmap::new(width as u16, height as u16);
    ctx.render_to_pixmap(&mut resources, &mut pixmap);

    let png_data = pixmap.into_png().unwrap();
    let path = outputs_folder.join(name);

    std::fs::write(&path, png_data).unwrap();
    let abs_path = path.canonicalize().unwrap();
    let url = format!("file://{}", abs_path.display());
    eprintln!(
        // Render as a terminal hyperlink; this avoids issues if the path to the workspace root contains a space.
        "Wrote output image for \x1b]8;;{url}\x1b\\{name}\x1b]8;;\x1b\\. Dimensions: {}x{}",
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
    // Load Roboto from the workspace assets (used for quick comparisons, not downloaded separately)
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColorBrush {
    color: AlphaColor<Srgb>,
}

impl Default for ColorBrush {
    fn default() -> Self {
        Self { color: css::WHITE }
    }
}
