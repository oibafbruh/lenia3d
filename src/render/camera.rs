//! Smoothed orbit camera with pan, cursor-directed zoom and fly controls.

use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Copy, Debug)]
struct Pose {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    dist: f32,
}

pub struct OrbitCamera {
    goal: Pose,
    cur: Pose,
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        let p = Pose { target: Vec3::ZERO, yaw: 0.6, pitch: 0.38, dist: 4.1 };
        Self { goal: p, cur: p, fov_y: 45f32.to_radians() }
    }
}

impl OrbitCamera {
    fn eye_of(p: &Pose) -> Vec3 {
        let dir = Vec3::new(p.pitch.cos() * p.yaw.sin(), p.pitch.sin(), p.pitch.cos() * p.yaw.cos());
        p.target + dir * p.dist
    }

    pub fn eye(&self) -> Vec3 {
        Self::eye_of(&self.cur)
    }

    pub fn target(&self) -> Vec3 {
        self.cur.target
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.cur.target, Vec3::Y)
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        let near = (self.cur.dist * 0.01).max(0.001);
        glam::camera::rh::proj::directx::perspective(self.fov_y, aspect.max(1e-3), near, 100.0)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view()
    }

    /// Basis vectors (right, up, forward) of the current view.
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let f = (self.cur.target - self.eye()).normalize();
        let r = f.cross(Vec3::Y).normalize_or(Vec3::X);
        let u = r.cross(f);
        (r, u, f)
    }

    /// World-space ray through a pixel (`ndc` in [-1, 1], y up).
    pub fn ray(&self, ndc: Vec2, aspect: f32) -> (Vec3, Vec3) {
        let inv = self.view_proj(aspect).inverse();
        let a = inv.project_point3(ndc.extend(0.0));
        let b = inv.project_point3(ndc.extend(1.0));
        (a, (b - a).normalize())
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.goal.yaw -= dx * 0.008;
        self.goal.pitch = (self.goal.pitch + dy * 0.008).clamp(-1.55, 1.55);
    }

    pub fn pan(&mut self, dx: f32, dy: f32, viewport_h: f32) {
        let (r, u, _) = self.basis();
        let scale = 2.0 * self.goal.dist * (self.fov_y * 0.5).tan() / viewport_h.max(1.0);
        self.goal.target += (-r * dx + u * dy) * scale;
    }

    /// Zoom by wheel `steps`, moving toward the point under the cursor.
    pub fn zoom(&mut self, steps: f32, ndc: Vec2, aspect: f32) {
        let factor = (-steps * 0.12).exp();
        let new_dist = (self.goal.dist * factor).clamp(0.05, 30.0);
        let f = new_dist / self.goal.dist;
        // point under the cursor at the target's depth
        let (o, d) = self.ray(ndc, aspect);
        let (_, _, fwd) = self.basis();
        let denom = d.dot(fwd);
        if denom.abs() > 1e-4 {
            let t = (self.cur.target - o).dot(fwd) / denom;
            let p = o + d * t;
            self.goal.target += (p - self.goal.target) * (1.0 - f);
        }
        self.goal.dist = new_dist;
    }

    /// Fly relative to the view: x right, y up, z forward (units per second scaled by distance).
    pub fn fly(&mut self, dir: Vec3, dt: f32) {
        let (r, _, f) = self.basis();
        let speed = self.goal.dist.max(0.3) * 0.9 * dt;
        self.goal.target += (r * dir.x + Vec3::Y * dir.y + f * dir.z) * speed;
    }

    pub fn frame_box(&mut self, center: Vec3, radius: f32) {
        self.goal.target = center;
        self.goal.dist = radius / (self.fov_y * 0.5).sin() * 1.05;
    }

    pub fn reset(&mut self) {
        let fov = self.fov_y;
        *self = Self::default();
        self.fov_y = fov;
    }

    /// Advance smoothing; returns true while still moving.
    pub fn update(&mut self, dt: f32) -> bool {
        let k = 1.0 - (-dt * 14.0).exp();
        let g = self.goal;
        let c = &mut self.cur;
        c.target += (g.target - c.target) * k;
        c.yaw += (g.yaw - c.yaw) * k;
        c.pitch += (g.pitch - c.pitch) * k;
        c.dist += (g.dist - c.dist) * k;
        (g.target - c.target).length() > 1e-4 || (g.yaw - c.yaw).abs() > 1e-4 || (g.pitch - c.pitch).abs() > 1e-4 || (g.dist - c.dist).abs() > 1e-4
    }

    /// Jump immediately to the goal (no smoothing).
    pub fn snap(&mut self) {
        self.cur = self.goal;
    }
}
