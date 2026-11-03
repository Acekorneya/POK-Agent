use std::path::Path;

fn main() {
    embed_whatsapp_component();
    tauri_build::build();
}

/// The WhatsApp client is a separate program built by
/// `scripts/build-whatsapp-sidecar.ps1` (its own crate and Rust version). When
/// that build exists, embed it in the app, so users get one file and never
/// have to download or place a second one. A build made without it still
/// compiles; WhatsApp is then reported as unavailable in Settings.
fn embed_whatsapp_component() {
    println!("cargo:rustc-check-cfg=cfg(has_whatsapp_component)");
    let component = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default())
        .join("../../../sidecars/whatsapp/target/release/pok-ai-whatsapp.exe");
    println!("cargo:rerun-if-changed={}", component.display());
    // It is a Windows program: other targets never carry it.
    let for_windows = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows");
    if let (true, Ok(path)) = (for_windows, component.canonicalize()) {
        println!("cargo:rustc-cfg=has_whatsapp_component");
        println!("cargo:rustc-env=POK_WHATSAPP_COMPONENT={}", path.display());
    }
}
