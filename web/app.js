(function () {
  "use strict";
  var DEMO = new URLSearchParams(location.search).get("demo") === "fixture";
  var STAGES = ["extract", "evidence", "decide", "generate", "output"];
  var TIMING = { extract: "extract_ms", evidence: "evidence_ms", decide: "decide_ms", generate: "generate_ms" };
  var $ = function (id) { return document.getElementById(id); };
  var csv = $("csv"), runBtn = $("run"), statusEl = $("status"), errEl = $("error");
  var rowsEl = $("rows"), claimsEl = $("claims"), detailEl = $("detail");
  var data = null, timer = null;

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function stageEl(name) { return document.querySelector('.stage[data-stage="' + name + '"]'); }
  function setStatus(t) { statusEl.textContent = t; }
  function showError(msg) { errEl.hidden = !msg; errEl.textContent = msg || ""; }
  function pct(x) { return Math.round((x || 0) * 100); }
  function pctText(x) { return x == null ? "\u2013" : pct(x) + "%"; }
  function fmtValue(v) {
    if (v == null) return "";
    if (typeof v !== "object") return String(v);
    return Object.keys(v).map(function (k) { return k + " " + v[k]; }).join(", ");
  }

  function resetStages() {
    STAGES.forEach(function (s) {
      var g = stageEl(s); g.classList.remove("active", "done");
      g.querySelector(".ms").textContent = "";
    });
  }
  function startAnimation() {
    resetStages();
    var i = 0;
    stageEl(STAGES[0]).classList.add("active");
    timer = setInterval(function () {
      stageEl(STAGES[i]).classList.remove("active");
      i = (i + 1) % STAGES.length;
      stageEl(STAGES[i]).classList.add("active");
    }, 380);
  }
  function stopAnimation(timings) {
    clearInterval(timer);
    STAGES.forEach(function (s) {
      var g = stageEl(s); g.classList.remove("active");
      if (timings) {
        g.classList.add("done");
        var k = TIMING[s];
        if (k && typeof timings[k] === "number") g.querySelector(".ms").textContent = timings[k].toFixed(1) + " ms";
      }
    });
  }

  function renderRows(rows) {
    rowsEl.replaceChildren();
    if (!rows.length) {
      var r0 = el("tr", "empty"); var c0 = el("td", null, "No rows returned."); c0.colSpan = 4;
      r0.appendChild(c0); rowsEl.appendChild(r0); return;
    }
    rows.forEach(function (r, i) {
      var tr = el("tr"); tr.dataset.i = i; tr.tabIndex = 0;
      tr.appendChild(el("td", "id", r.id));
      var t = el("td", "text", r.text); t.title = r.text; tr.appendChild(t);
      var lt = el("td"); lt.appendChild(el("span", "chip " + r.label, r.label)); tr.appendChild(lt);
      var ct = el("td"); var m = el("span", "meter"); var f = el("i");
      f.style.width = (r.confidence == null ? 0 : pct(r.confidence)) + "%"; m.appendChild(f);
      ct.appendChild(m); ct.appendChild(el("span", "pct", pctText(r.confidence))); tr.appendChild(ct);
      tr.addEventListener("click", function () { select(i); });
      tr.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(i); }
      });
      rowsEl.appendChild(tr);
    });
  }
  function renderClaims(claims) {
    claimsEl.replaceChildren();
    if (!claims.length) { claimsEl.appendChild(el("li", "empty", "No claims returned.")); return; }
    claims.slice(0, 20).forEach(function (c) {
      var li = el("li");
      li.appendChild(el("span", "ty", c.type));
      li.appendChild(el("span", null, c.text));
      li.appendChild(el("span", "cf", pctText(c.confidence)));
      claimsEl.appendChild(li);
    });
  }
  function select(i) {
    Array.prototype.forEach.call(rowsEl.children, function (tr) {
      tr.classList.toggle("sel", tr.dataset.i === String(i));
    });
    var r = data.rows[i];
    detailEl.replaceChildren();
    detailEl.appendChild(el("h2", null, r.id + " is " + r.label + " (" + pctText(r.confidence) + ")"));
    var cols = el("div", "cols");
    var a = el("div"); a.appendChild(el("h3", null, "Facts extracted"));
    var ul = el("ul");
    (r.facts || []).forEach(function (f) {
      var li = el("li");
      li.appendChild(el("code", null, f.subject + " / " + f.attribute));
      li.appendChild(document.createTextNode(" = " + fmtValue(f.value)));
      ul.appendChild(li);
    });
    if (!ul.children.length) ul.appendChild(el("li", "empty", "None found."));
    a.appendChild(ul);
    var b = el("div"); b.appendChild(el("h3", null, "Checks against evidence"));
    var ul2 = el("ul");
    (r.checks || []).forEach(function (c) {
      var li = el("li");
      li.appendChild(el("code", null, c.state));
      li.appendChild(document.createTextNode(" "));
      li.appendChild(el("span", "st " + c.status, c.status));
      var extra = ", support " + (typeof c.support === "number" ? c.support.toFixed(2) : c.support);
      if (c.days != null) extra += ", " + c.days + " d old";
      if (c.trusted === false) extra += ", untrusted page";
      li.appendChild(document.createTextNode(extra));
      ul2.appendChild(li);
    });
    if (!ul2.children.length) ul2.appendChild(el("li", "empty", "No checks ran."));
    b.appendChild(ul2);
    cols.appendChild(a); cols.appendChild(b); detailEl.appendChild(cols);
  }

  function tab(which) {
    ["rows", "claims"].forEach(function (n) {
      $("tab-" + n).setAttribute("aria-selected", String(n === which));
      $("pane-" + n).hidden = n !== which;
    });
  }
  $("tab-rows").addEventListener("click", function () { tab("rows"); });
  $("tab-claims").addEventListener("click", function () { tab("claims"); });

  function fixture() { return fetch("/static/fixtures/run.json").then(function (r) { return r.json(); }); }
  function loadSample() {
    var p = DEMO
      ? fixture().then(function (d) {
          return "id,text,date,label\n" + d.rows.map(function (r) {
            return r.id + ',"' + r.text.replace(/"/g, '""') + '",' + r.date + ",";
          }).join("\n") + "\n";
        })
      : fetch("/api/sample").then(function (r) { return r.json(); }).then(function (d) { return d.csv; });
    p.then(function (t) { csv.value = t; runBtn.disabled = false; setStatus("Sample loaded. Press Run."); })
     .catch(function () { runBtn.disabled = false; setStatus("Could not load the sample. Paste your own CSV."); });
  }
  function run() {
    showError(""); runBtn.disabled = true; setStatus("Running…"); startAnimation();
    var t0 = performance.now();
    var req = DEMO
      ? new Promise(function (res) { setTimeout(res, 1600); }).then(fixture).then(function (b) { return { ok: true, body: b }; })
      : fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ csv: csv.value }) })
          .then(function (r) {
            return r.json().catch(function () { return {}; }).then(function (b) { return { ok: r.ok, body: b }; });
          });
    req.then(function (res) {
      if (!res.ok) {
        stopAnimation(null); showError(res.body.detail || "The server rejected the CSV."); setStatus("Failed."); return;
      }
      data = res.body;
      stopAnimation(data.timings || {});
      renderRows(data.rows || []); renderClaims(data.claims || []);
      detailEl.replaceChildren(el("p", "empty", "Select a row to see the facts and checks behind its label."));
      var n = (data.rows || []).length;
      setStatus(n + " sentences labelled in " + Math.round(performance.now() - t0) + " ms.");
      if (n) select(0);
    }).catch(function () {
      stopAnimation(null); showError("Could not reach the server. Check that it is running and try again."); setStatus("Failed.");
    }).then(function () { runBtn.disabled = false; });
  }
  runBtn.addEventListener("click", run);
  loadSample();
})();
