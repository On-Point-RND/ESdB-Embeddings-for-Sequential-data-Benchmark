(() => {
  "use strict";

  const data = window.BENCHMARK_RESULTS;
  if (!data) throw new Error("results.js must load before app.js");

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const escapeHtml = (value) => String(value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");

  const metricColumns = [
    ["regression", "Regression"], ["classification", "Classification"],
    ["forecasting", "Forecasting"], ["anomaly", "Anomaly"], ["overall", "Overall"]
  ];

  function renderRanking() {
    const best = Object.fromEntries(metricColumns.map(([key]) => [key, Math.min(...data.rankings.map((row) => row[key]))]));
    const second = Object.fromEntries(metricColumns.map(([key]) => {
      const values = [...new Set(data.rankings.map((row) => row[key]))].sort((a, b) => a - b);
      return [key, values[1]];
    }));

    $("#ranking-body").innerHTML = data.rankings.map((row, index) => `
      <tr><td class="rank-number">${index + 1}</td><th scope="row">${escapeHtml(row.method)}</th>
        ${metricColumns.map(([key]) => `<td class="${row[key] === best[key] ? "best" : row[key] === second[key] ? "second" : ""}">${row[key].toFixed(2)}</td>`).join("")}
      </tr>`).join("");

    $("#overall-ranks").innerHTML = data.rankings.map((row) => `
      <div class="rank-bar"><span>${escapeHtml(row.method)}</span><div aria-hidden="true"><i style="--rank:${row.overall}"></i></div><strong>${row.overall.toFixed(2)}</strong></div>
    `).join("");
  }

  let activeDomain = "all";

  function renderDatasets() {
    const visible = activeDomain === "all" ? data.datasets : data.datasets.filter((dataset) => dataset.domainKey === activeDomain);
    $("#dataset-grid").innerHTML = visible.map((dataset) => `
      <article class="dataset-card" data-domain="${escapeHtml(dataset.domainKey)}">
        <div class="dataset-card-top"><span class="domain-pill ${escapeHtml(dataset.domainKey)}">${escapeHtml(dataset.domain)}</span><span>${dataset.targets.length}/4 tasks</span></div>
        <h3>${escapeHtml(dataset.name)}</h3><p>${escapeHtml(dataset.summary)}</p>
        <dl><div><dt>Sequences</dt><dd>${escapeHtml(dataset.sequences)}</dd></div><div><dt>Events</dt><dd>${escapeHtml(dataset.events)}</dd></div></dl>
        <button class="text-button" type="button" data-dataset="${escapeHtml(dataset.id)}" aria-haspopup="dialog">View targets <span aria-hidden="true">↗</span></button>
      </article>`).join("");
    $$('[data-dataset]').forEach((button) => button.addEventListener("click", () => openDataset(button.dataset.dataset)));
    $("#dataset-count").textContent = `${visible.length} dataset${visible.length === 1 ? "" : "s"}`;
  }

  function openDataset(id) {
    const dataset = data.datasets.find((item) => item.id === id);
    if (!dataset) return;
    $("#dataset-dialog-content").innerHTML = `
      <header class="dialog-heading"><div><span class="domain-pill ${escapeHtml(dataset.domainKey)}">${escapeHtml(dataset.domain)}</span><h2 id="dataset-dialog-title">${escapeHtml(dataset.name)}</h2><p>${escapeHtml(dataset.summary)}</p></div><button class="dialog-close" type="button" data-close-dialog aria-label="Close dataset details">×</button></header>
      <div class="dataset-facts">
        <div><span>Structure</span><strong>${escapeHtml(dataset.structure)}</strong></div><div><span>Cardinality</span><strong>${escapeHtml(dataset.cardinality)}</strong></div>
        <div><span>Mean length</span><strong>${escapeHtml(dataset.meanLength)}</strong></div><div><span>Local / global tasks</span><strong>${dataset.localTasks} / ${dataset.globalTasks}</strong></div>
        <div><span>Max shifts</span><strong>${escapeHtml(dataset.shifts)}</strong></div><div><span>Global train / test</span><strong>${escapeHtml(dataset.globalSplit)}</strong></div>
      </div>
      <div class="table-scroll compact-table"><table><thead><tr><th>Task</th><th>Constructed target</th><th>Scope / horizon</th><th>Balance / metric</th></tr></thead><tbody>
        ${dataset.targets.map((target) => `<tr>${target.map((value, index) => index === 0 ? `<th scope="row">${escapeHtml(value)}</th>` : `<td>${escapeHtml(value)}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div><p class="dialog-source">Dataset structure and targets: Tables 7–9 and 11 of the preprint.</p>`;
    const dialog = $("#dataset-dialog");
    $("[data-close-dialog]", dialog).addEventListener("click", () => dialog.close());
    dialog.showModal();
  }

  function renderSimpleTable(target, rows, columns) {
    $(target).innerHTML = rows.map((row) => `<tr>${columns.map(([key, header], index) => index === 0 ? `<th scope="row">${escapeHtml(row[key])}</th>` : `<td data-label="${escapeHtml(header)}">${escapeHtml(row[key])}</td>`).join("")}</tr>`).join("");
  }

  function renderFusionTable(target, table) {
    const numericRows = table.rows.map((row) => row.slice(1).map((value) => {
      const number = Number.parseFloat(String(value).replace("−", "-"));
      return Number.isFinite(number) ? number : Number.NEGATIVE_INFINITY;
    }));
    const bestByColumn = [0, 1, 2, 3].map((column) => Math.max(...numericRows.map((row) => row[column])));
    $(target).innerHTML = table.rows.map((row, rowIndex) => `<tr><th scope="row">${escapeHtml(row[0])}</th>${row.slice(1).map((value, column) => `<td class="${numericRows[rowIndex][column] === bestByColumn[column] ? "best" : ""}">${escapeHtml(value)}</td>`).join("")}</tr>`).join("");
  }

  function renderFusion() {
    $("#fusion-main-label").textContent = data.fusion.main.label;
    renderFusionTable("#fusion-main-body", data.fusion.main);
    $("#fusion-tabs").innerHTML = data.fusion.studies.map((study, index) => `<button type="button" role="tab" aria-selected="${index === 0}" data-fusion-tab="${escapeHtml(study.id)}">${escapeHtml(study.label.split(" · ")[0])}</button>`).join("");
    selectFusion(data.fusion.studies[0].id);
    $$('[data-fusion-tab]').forEach((button) => button.addEventListener("click", () => selectFusion(button.dataset.fusionTab)));
  }

  function selectFusion(id) {
    const study = data.fusion.studies.find((item) => item.id === id);
    if (!study) return;
    $$('[data-fusion-tab]').forEach((button) => button.setAttribute("aria-selected", String(button.dataset.fusionTab === id)));
    $("#fusion-study-title").textContent = study.label;
    $("#fusion-study-note").textContent = study.note;
    renderFusionTable("#fusion-study-body", study);
  }

  function renderMethods() {
    $("#methods-grid").innerHTML = data.methods.map((method, index) => `
      <details class="method-card" ${index === 0 ? "open" : ""}><summary><span><b>${escapeHtml(method.name)}</b><small>${escapeHtml(method.family)}</small></span><i aria-hidden="true">+</i></summary><p>${escapeHtml(method.description)}</p><span class="method-origin">${escapeHtml(method.origin)}</span></details>
    `).join("");
  }

  function renderAuthors() {
    $("#authors-list").innerHTML = data.authors.map(([name, affiliation]) => `<li><strong>${escapeHtml(name)}</strong><span>${escapeHtml(affiliation)}</span></li>`).join("");
  }

  function setupDialogs() {
    $$('dialog').forEach((dialog) => dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); }));
    $$('[data-figure]').forEach((button) => button.addEventListener("click", () => {
      const dialog = $("#figure-dialog");
      const image = $("img", dialog);
      image.src = button.dataset.figure;
      image.alt = button.dataset.alt || "Expanded research figure";
      $("figcaption", dialog).textContent = button.dataset.caption || "";
      dialog.showModal();
    }));
    $("#figure-dialog-close").addEventListener("click", () => $("#figure-dialog").close());
  }

  function setupNavigation() {
    const links = $$('.topbar a[href^="#"]');
    const sections = links.map((link) => $(link.getAttribute("href"))).filter(Boolean);
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`));
    }, { rootMargin: "-30% 0px -60%", threshold: [0, 0.2, 0.6] });
    sections.forEach((section) => observer.observe(section));
  }

  function setupDomainFilters() {
    $$('[data-domain-filter]').forEach((button) => button.addEventListener("click", () => {
      activeDomain = button.dataset.domainFilter;
      $$('[data-domain-filter]').forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
      renderDatasets();
    }));
  }

  renderRanking();
  renderDatasets();
  renderFusion();
  renderMethods();
  renderAuthors();
  renderSimpleTable("#validator-body", data.validators, [["scope", "Scope"], ["lightgbm", "LightGBM"], ["mlp", "MLP"], ["linear", "Linear probe"]]);
  renderSimpleTable("#multi-target-body", data.multiTarget, [["dataset", "Dataset"], ["model", "Model"], ["regime", "HPO"], ["regression", "Regression"], ["classification", "Classification"], ["forecasting", "Forecasting"], ["anomaly", "Anomaly"]]);
  setupDomainFilters();
  setupDialogs();
  setupNavigation();
})();
