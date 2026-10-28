import { motion, useInView, useReducedMotion, animate } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Reveal({ children, className, delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.7, delay, ease: [0.2, 0.7, 0.2, 1] }}
    >
      {children}
    </motion.div>
  );
}

export function SectionHead({ eyebrow, a, b, className }: { eyebrow: string; a: string; b: string; className?: string }) {
  return (
    <Reveal className={cn("mb-12", className)}>
      <p className="label mb-5 text-muted-foreground">{eyebrow}</p>
      <h2 className="text-4xl font-light md:text-6xl">
        <span className="block">{a}</span>
        <span className="headline-ghost block">{b}</span>
      </h2>
    </Reveal>
  );
}

export function BracketButton({ children, className, ...p }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button {...p} className={cn("btn-bracket", className)}>
      [ {children} ]
    </button>
  );
}

export function Counter({ to, decimals = 0, suffix = "" }: { to: number; decimals?: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!inView) return;
    const c = animate(0, to, { duration: 1.8, ease: "easeOut", onUpdate: setV });
    return () => c.stop();
  }, [inView, to]);
  return <span ref={ref}>{v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}</span>;
}

export function VerdictChip({ label }: { label: "true" | "false" | "unsure" }) {
  const cls = label === "true" ? "bg-truth/40 text-foreground border-truth" : label === "false" ? "bg-hype text-primary-foreground border-hype" : "bg-muted text-muted-foreground border-border";
  return <span className={cn("label inline-flex rounded-full border px-2.5 py-1 text-[9px]", cls)}>{label}</span>;
}

export function ConfBar({ value }: { value: number | null }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(1, (value - 0.5) / 0.5)) * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-muted">
        <motion.div className="h-full rounded-full bg-primary" initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: 0.8 }} />
      </div>
      <span className="font-label text-[10px] tabular-nums text-muted-foreground">{value == null ? "—" : value.toFixed(2)}</span>
    </div>
  );
}

export function Cursor() {
  const dot = useRef<HTMLDivElement>(null);
  const [big, setBig] = useState(false);
  const [on, setOn] = useState(false);
  useEffect(() => {
    if (!window.matchMedia("(pointer: fine)").matches) return;
    setOn(true);
    const move = (e: PointerEvent) => {
      if (dot.current) dot.current.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
      const el = e.target as HTMLElement;
      setBig(!!el.closest?.("a,button,[role=button],input,textarea,select,label"));
    };
    window.addEventListener("pointermove", move);
    return () => window.removeEventListener("pointermove", move);
  }, []);
  if (!on) return null;
  return (
    <div ref={dot} className="pointer-events-none fixed left-0 top-0 z-[70]" aria-hidden>
      <div className={cn("-translate-x-1/2 -translate-y-1/2 rounded-full bg-primary mix-blend-multiply transition-all duration-200", big ? "h-9 w-9 opacity-25" : "h-2 w-2 opacity-90")} />
    </div>
  );
}
