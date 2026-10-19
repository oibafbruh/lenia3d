// Embed every presets/*.json file into the binary.
use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-changed=presets");
    let dir = Path::new("presets");
    let mut files: Vec<_> = fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect())
        .unwrap_or_default();
    files.sort();
    let mut code = String::from("pub static BUILTIN_PRESETS: &[(&str, &str)] = &[\n");
    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
        let abs = fs::canonicalize(f).unwrap();
        code.push_str(&format!(
            "    ({:?}, include_str!({:?})),\n",
            f.file_name().unwrap().to_string_lossy(),
            abs.display().to_string()
        ));
    }
    code.push_str("];\n");
    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("builtin_presets.rs"), code).unwrap();
}
