// Hot loop over typed arrays; indices are always in range.
// @ts-nocheck
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

const N = 2000;
const HYPE_RATIO = 0.22;
const DRIFT_SECONDS = 3.5;
const SORT_SECONDS = 3;
const REPEL_RADIUS = 0.9;

export interface HoverInfo { index: number; hype: boolean; x: number; y: number }

function makeSprite() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.35, "rgba(255,255,255,0.75)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function Particles({ onHover }: { onHover: (h: HoverInfo | null) => void }) {
  const ref = useRef<THREE.Points>(null);
  const { camera, size, pointer } = useThree();
  const sprite = useMemo(makeSprite, []);
  const data = useMemo(() => {
    const pos = new Float32Array(N * 3);
    const start = new Float32Array(N * 3);
    const target = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    const hype = new Uint8Array(N);
    const seed = new Float32Array(N);
    const truth = new THREE.Color("#B6D3C1");
    const emerald = new THREE.Color("#267559");
    const red = new THREE.Color("#AB403A");
    for (let i = 0; i < N; i++) {
      const h = Math.random() < HYPE_RATIO ? 1 : 0;
      hype[i] = h;
      seed[i] = Math.random() * 100;
      const r = 3 + Math.random() * 4;
      const a = Math.random() * Math.PI * 2;
      start[i * 3] = Math.cos(a) * r * 1.6;
      start[i * 3 + 1] = (Math.random() - 0.5) * 6;
      start[i * 3 + 2] = Math.sin(a) * r - 2;
      const tx = (Math.random() - 0.5) * 16;
      target[i * 3] = tx;
      target[i * 3 + 1] = h ? -1.6 + (Math.random() - 0.5) * 0.9 : 1.4 + (Math.random() - 0.5) * 0.6;
      target[i * 3 + 2] = (Math.random() - 0.5) * 1.5;
      const c = h ? red : truth.clone().lerp(emerald, Math.random() * 0.6);
      col.set([c.r, c.g, c.b], i * 3);
    }
    pos.set(start);
    return { pos, start, target, col, hype, seed };
  }, []);

  const offset = useRef(new Float32Array(N * 3));
  const clock = useRef(0);
  const lastHover = useRef(0);
  const v = useMemo(() => new THREE.Vector3(), []);

  useEffect(() => {
    document.querySelector("[data-hero-canvas]")?.setAttribute("data-scene-ready", "true");
  }, []);

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.05);
    clock.current += dt;
    const t = clock.current;
    const k = Math.min(1, Math.max(0, (t - DRIFT_SECONDS) / SORT_SECONDS));
    const ease = k * k * (3 - 2 * k);
    // cursor in world at z=0
    v.set(pointer.x, pointer.y, 0.5).unproject(camera);
    const dir = v.sub(camera.position).normalize();
    const dist = -camera.position.z / dir.z;
    const mx = camera.position.x + dir.x * dist;
    const my = camera.position.y + dir.y * dist;
    const { pos, start, target, hype, seed } = data;
    const off = offset.current;
    let best = -1, bestD = 0.35;
    for (let i = 0; i < N; i++) {
      const i3 = i * 3, s = seed[i];
      const fx = start[i3] + Math.sin(t * 0.25 + s) * 0.6;
      const fy = start[i3 + 1] + Math.cos(t * 0.2 + s * 1.3) * 0.5;
      const fz = start[i3 + 2] + Math.sin(t * 0.15 + s * 0.7) * 0.6;
      const flow = ((target[i3] + t * 0.35 + 8) % 16) - 8;
      let tx = flow, ty = target[i3 + 1], tz = target[i3 + 2];
      if (hype[i]) {
        const glitch = Math.sin(t * 13 + s * 7) > 0.92 ? 0.25 : 0;
        tx += Math.sin(t * 9 + s) * 0.04 + glitch;
        ty += Math.cos(t * 11 + s * 3) * 0.06;
      } else {
        ty += Math.sin(t * 0.8 + tx * 0.6) * 0.08;
      }
      let x = fx + (tx - fx) * ease, y = fy + (ty - fy) * ease, z = fz + (tz - fz) * ease;
      const dx = x - mx, dy = y - my, d = Math.hypot(dx, dy);
      let ox = off[i3] * 0.9, oy = off[i3 + 1] * 0.9;
      if (d < REPEL_RADIUS && d > 0.0001) {
        const f = (1 - d / REPEL_RADIUS) * 0.12;
        ox += (dx / d) * f; oy += (dy / d) * f;
      }
      off[i3] = ox; off[i3 + 1] = oy;
      x += ox; y += oy;
      pos[i3] = x; pos[i3 + 1] = y; pos[i3 + 2] = z;
      if (d < bestD && ease > 0.9) { bestD = d; best = i; }
    }
    if (ref.current) (ref.current.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    // gentle parallax
    camera.position.x += (pointer.x * 0.8 - camera.position.x) * 0.03;
    camera.position.y += (pointer.y * 0.5 - camera.position.y) * 0.03;
    camera.lookAt(0, 0, 0);
    if (t - lastHover.current > 0.12) {
      lastHover.current = t;
      if (best >= 0) {
        onHover({ index: best, hype: !!hype[best], x: ((pointer.x + 1) / 2) * size.width, y: ((1 - pointer.y) / 2) * size.height });
      } else onHover(null);
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[data.pos, 3]} />
        <bufferAttribute attach="attributes-color" args={[data.col, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.11} map={sprite} vertexColors transparent depthWrite={false} sizeAttenuation opacity={0.45} />
    </points>
  );
}

export default function ParticleField({ onHover }: { onHover: (h: HoverInfo | null) => void }) {
  return (
    <div data-hero-canvas className="absolute inset-0">
      <Canvas camera={{ position: [0, 0, 9], fov: 50 }} dpr={[1, 1.75]} gl={{ antialias: false, alpha: true }}>
        <Particles onHover={onHover} />
      </Canvas>
    </div>
  );
}
