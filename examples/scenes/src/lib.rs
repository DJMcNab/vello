use vello::{
    kurbo::{Affine, PathEl, Point},
    peniko::{Brush, Color, Fill},
    Scene, SceneBuilder,
};

pub fn gen_test_scene() -> Scene {
    let mut scene = Scene::default();
    let mut builder = SceneBuilder::for_scene(&mut scene);
    let path = [
        PathEl::MoveTo(Point::new(100.0, 100.0)),
        PathEl::LineTo(Point::new(500.0, 120.0)),
        PathEl::LineTo(Point::new(300.0, 150.0)),
        PathEl::LineTo(Point::new(200.0, 260.0)),
        PathEl::LineTo(Point::new(150.0, 210.0)),
        PathEl::ClosePath,
    ];
    let brush = Brush::Solid(Color::rgb8(0x80, 0x80, 0x80));
    builder.fill(Fill::NonZero, Affine::IDENTITY, &brush, None, &path);
    scene
}
