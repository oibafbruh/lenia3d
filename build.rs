// Embed every presets/*.json file into the binary.
use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-changed=presets");
    println!("cargo:rerun-if-changed=presets/flow");
    let list = |dir: &Path| -> Vec<std::path::PathBuf> {
        let mut v: Vec<_> = fs::read_dir(dir)
            .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect())
            .unwrap_or_default();
        v.sort();
        v
    };
    let mut files = list(Path::new("presets"));
    files.extend(list(Path::new("presets/flow")));
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
