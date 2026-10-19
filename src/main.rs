use lenia3d::headless::{self, Options};
use lenia3d::render::RenderMode;

fn usage() -> ! {
    eprintln!(
        "Lenia3D - real-time 3D Lenia\n\n\
         usage: lenia3d [options]\n\n\
         (no options)            start the interactive app\n\
         --size N                world size (64, 128, 256, 512)\n\
         --preset NAME           preset to load\n\
         --bench                 benchmark simulation speed and exit\n\
         --steps N               steps for --bench / --screenshot / --chan-check\n\
         --screenshot FILE.png   render one image headless and exit\n\
         --mode organic|glow|xray  render mode for --screenshot\n\
         --resolution WxH        image size for --screenshot\n\
         --scale F               creature scale factor\n\
         --chan-file FILE        Bert Chan's animals3D.json to include\n\
         --preset-file FILE      extra preset JSON (one preset or a list)\n\
         --log-every N           print population stats every N steps (Evolution mode)\n\
         --check                 verify every preset stays alive (add --chan-file for Chan's)\n\
         --list-presets          print built-in presets"
    );
    std::process::exit(2)
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("lenia3d=info,wgpu_core=warn,wgpu_hal=error"))
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut opts = Options::default();
    let mut action = "app";
    let mut out = String::new();
    let mut size_given = false;
    let mut i = 0;
    let next = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| usage())
    };
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                opts.size = next(&mut i).parse().unwrap_or_else(|_| usage());
                size_given = true;
            }
            "--steps" => opts.steps = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--preset" => opts.preset = Some(next(&mut i)),
            "--scale" => opts.scale = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--render" => opts.render = next(&mut i),
            "--preset-file" => opts.preset_file = Some(next(&mut i)),
            "--log-every" => headless::LOG_EVERY.store(next(&mut i).parse().unwrap_or_else(|_| usage()), std::sync::atomic::Ordering::Relaxed),
            "--dist" => opts.dist = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--yaw" => opts.yaw = next(&mut i).parse().unwrap_or_else(|_| usage()),
            "--chan-file" => opts.chan_file = Some(next(&mut i)),
            "--mode" => {
                opts.mode = match next(&mut i).as_str() {
                    "glow" => RenderMode::Glow,
                    "xray" => RenderMode::Xray,
                    _ => RenderMode::Organic,
                }
            }
            "--resolution" => {
                let r = next(&mut i);
                let (w, h) = r.split_once('x').unwrap_or_else(|| usage());
                opts.width = w.parse().unwrap_or_else(|_| usage());
                opts.height = h.parse().unwrap_or_else(|_| usage());
            }
            "--bench" => action = "bench",
            "--check" | "--chan-check" => action = "check",
            "--list-presets" => action = "list",
            "--screenshot" => {
                action = "screenshot";
                out = next(&mut i);
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown option {other}");
                usage()
            }
        }
        i += 1;
    }
    if !opts.size.is_power_of_two() || !(32..=512).contains(&opts.size) {
        eprintln!("--size must be a power of two between 32 and 512");
        std::process::exit(2);
    }
    let result = match action {
        "bench" => headless::bench(&opts),
        "screenshot" => headless::screenshot(&opts, &out),
        "check" => headless::check_presets(&opts),
        "list" => headless::all_presets(opts.chan_file.as_deref()).map(|ps| {
            for p in ps {
                println!("{:<32} {:>2} ch {:>2} kernels  R={:<5} {}", p.name, p.rule.channels, p.rule.kernels.len(), p.rule.R, p.description);
            }
        }),
        _ => lenia3d::app::run(opts, size_given),
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
