import { Canvas, useFrame } from "@react-three/fiber";
import { Html, Line, OrbitControls } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { EDGES, STAGES } from "@/lib/arena";

const POS: [number, number, number][] = [
  [-7, 0, 0], [-4.5, 1.2, 0], [-2, 0, 0], [0.5, 2.2, -1], [1, -0.6, 0.5],
  [3.5, 0, 0], [6, 0.8, 0], [4, 3.2, -1.5],
];

function readCss(name: string) {
  if (typeof window === "undefined") return "#fff";
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const c = document.createElement("canvas").getContext("2d")!;
  c.fillStyle = v || "#fff";
  return c.fillStyle as string;
}

function Node({ i, active, onSelect, selected }: { i: number; active: boolean; selected: boolean; onSelect: (i: number) => void }) {
  const ref = useRef<THREE.Mesh>(null);
  const colors = useMemo(() => ({ a: readCss("--node-active"), b: readCss("--node") }), []);
  useFrame((s) => {
    if (!ref.current) return;
    ref.current.rotation.y += 0.01;
    const k = active ? 1 + Math.sin(s.clock.elapsedTime * 6 + i) * 0.12 : 1;
    ref.current.scale.setScalar(selected ? 1.3 : k);
  });
  return (
    <group position={POS[i]!}>
      <mesh ref={ref} onClick={() => onSelect(i)}>
        <icosahedronGeometry args={[0.45, 1]} />
        <meshStandardMaterial color={active || selected ? colors.a : colors.b} flatShading emissive={active ? colors.a : "#000"} emissiveIntensity={active ? 0.6 : 0} />
      </mesh>
      <Html center position={[0, -0.85, 0]} distanceFactor={9}>
        <div className="diagram-label">{STAGES[i]?.name}</div>
      </Html>
    </group>
  );
}

function Packets({ running }: { running: boolean }) {
  const refs = useRef<THREE.Mesh[]>([]);
  const color = useMemo(() => readCss("--packet"), []);
  useFrame((s) => {
    refs.current.forEach((m, idx) => {
      if (!m) return;
      const [a, b] = EDGES[idx % EDGES.length]!;
      const t = ((s.clock.elapsedTime * (running ? 0.9 : 0.25)) + idx * 0.37) % 1;
      m.position.lerpVectors(new THREE.Vector3(...POS[a]!), new THREE.Vector3(...POS[b]!), t);
    });
  });
  return (
    <>
      {Array.from({ length: EDGES.length * 2 }).map((_, i) => (
        <mesh key={i} ref={(m) => { if (m) refs.current[i] = m; }}>
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshBasicMaterial color={color} />
        </mesh>
      ))}
    </>
  );
}

export default function Diagram3D({ running, selected, onSelect }: { running: boolean; selected: number | null; onSelect: (i: number) => void }) {
  const edgeColor = useMemo(() => readCss("--edge"), []);
  return (
    <Canvas camera={{ position: [0, 1.5, 12], fov: 50 }} dpr={[1, 2]}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 8, 5]} intensity={1.2} />
      <pointLight position={[-6, -3, 4]} intensity={20} />
      {EDGES.map(([a, b], i) => (
        <Line key={i} points={[POS[a]!, POS[b]!]} color={edgeColor} lineWidth={1.5} transparent opacity={0.6} />
      ))}
      {STAGES.map((_, i) => (
        <Node key={i} i={i} active={running} selected={selected === i} onSelect={onSelect} />
      ))}
      <Packets running={running} />
      <gridHelper args={[30, 30, edgeColor, edgeColor]} position={[0, -3, 0]} material-opacity={0.15} material-transparent />
      <OrbitControls enablePan={false} minDistance={7} maxDistance={20} autoRotate={!running} autoRotateSpeed={0.3} />
    </Canvas>
  );
}
