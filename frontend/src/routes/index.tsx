import { createFileRoute } from "@tanstack/react-router";
import { Hero, Nav } from "@/components/hot/Hero";
import { Architecture } from "@/components/hot/Architecture";
import { LiveRun } from "@/components/hot/LiveRun";
import { Claims } from "@/components/hot/Claims";
import { Evaluation, Footer } from "@/components/hot/Evaluation";
import { Cursor } from "@/components/hot/ui";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Hype or Truth — a fact-checker with no LLM inside" },
      { name: "description", content: "Reads every sentence about a company, checks it against the record, and tells you how sure it is. Run it live." },
      { property: "og:title", content: "Hype or Truth — Less hype. More truth." },
      { property: "og:description", content: "A calibrated fact-checking system for company claims. 88.8% accurate on unseen pages, no LLM inside." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <div className="grain min-h-screen">
      <Cursor />
      <Nav />
      <main>
        <Hero />
        <Architecture />
        <LiveRun />
        <Claims />
        <Evaluation />
      </main>
      <Footer />
    </div>
  );
}
