const state = {
  data: null,
  selected: null
};

const selectors = {
  overviewTaskFilter: document.querySelector("#overview-task-filter"),
  resetView: document.querySelector("#reset-view"),
  datasetDonut: document.querySelector("#dataset-donut"),
  taskAvailabilityDonut: document.querySelector("#task-availability-donut"),
  sourceDonut: document.querySelector("#source-donut"),
  scoreHistogram: document.querySelector("#score-histogram"),
  scoreHistogramLabel: document.querySelector("#score-histogram-label"),
  compositeHistogram: document.querySelector("#composite-histogram"),
  winnerChart: document.querySelector("#winner-chart"),
  datasetFilter: document.querySelector("#dataset-filter"),
  sortFilter: document.querySelector("#sort-filter"),
  search: document.querySelector("#method-search"),
  summary: document.querySelector("#task-summary"),
  compositeChart: document.querySelector("#composite-chart"),
  taskChart: document.querySelector("#task-chart"),
  methodLandscape: document.querySelector("#method-landscape"),
  datasetHeatmap: document.querySelector("#dataset-heatmap"),
  metricDistribution: document.querySelector("#metric-distribution"),
  additionalSummary: document.querySelector("#additional-summary"),
  ntpDatasetFilter: document.querySelector("#ntp-dataset-filter"),
  ntpTaskFilter: document.querySelector("#ntp-task-filter"),
  ntpChart: document.querySelector("#ntp-chart"),
  ntpBody: document.querySelector("#ntp-body"),
  archiveChart: document.querySelector("#archive-chart"),
  sheetFilter: document.querySelector("#sheet-filter"),
  sheetSearch: document.querySelector("#sheet-search"),
  sheetSummary: document.querySelector("#sheet-summary"),
  sheetChart: document.querySelector("#sheet-chart"),
  sheetHead: document.querySelector("#sheet-head"),
  sheetBody: document.querySelector("#sheet-body"),
  tableStatus: document.querySelector("#table-status"),
  body: document.querySelector("#results-body"),
  cards: document.querySelector("#dataset-cards"),
  datasetModal: document.querySelector("#dataset-modal"),
  datasetModalContent: document.querySelector("#dataset-modal-content"),
  datasetModalClose: document.querySelector("#dataset-modal-close")
};

const chartPalette = [
  "#66D92C",
  "#80E054",
  "#98DB2C",
  "#2CD996",
  "#2BD6D6",
  "#30C0F0",
  "#308FF0",
  "#4872F0",
  "#6533FF",
  "#411BD6",
  "#D2DE26",
  "#2EE4B6"
];

const datasetInfo = {
  "30music": {
    summary: "Music listening sequences from Last.fm-style interaction logs.",
    domain: "Music recommendation / listening behavior",
    scale: "Local notes: 45k users, more than 31M listening events, 4.1M likes. RecSys'24 preprocessing stats: 43,762 users, 822,507 items, 22.6M interactions.",
    featureCount: "2 event features: 1 cat, 1 num",
    features: [
      { name: "client_id", type: "id", cardinality: "43,762 users after RecSys'24 preprocessing", description: "Sequence identifier." },
      { name: "timestamp", type: "time", cardinality: "event-level timestamp", description: "Ordering field for user events." },
      { name: "track_id", type: "cat", cardinality: "822,507 items after RecSys'24 preprocessing; 5.6M raw tracks in local notes", description: "Listened music track." },
      { name: "play_duration", type: "num", cardinality: "continuous", description: "Listening duration for the event." }
    ],
    targets: [
      "diversity quartile classification",
      "play-duration regression",
      "same-day event forecasting",
      "anomaly from high diversity and low mean duration"
    ],
    preprocessing: [
      "frequency-code categorical columns",
      "collect user-level sequences",
      "use a 3-day local horizon",
      "sample shift positions"
    ]
  },
  age: {
    summary: "Anonymized bank-card transaction sequences for age-group modeling.",
    domain: "Banking / card transactions",
    scale: "44M transactions; 50k clients in the classification setup; 30k sequences in the distribution-forecasting setup.",
    featureCount: "2 event features: 1 cat, 1 num",
    features: [
      { name: "client_id", type: "id", cardinality: "50k clients in the classification setup", description: "Sequence identifier." },
      { name: "trans_date", type: "time", cardinality: "daily transaction date", description: "Ordering field." },
      { name: "small_group", type: "cat", cardinality: "203 transaction categories", description: "Transaction category such as spending or merchant group." },
      { name: "amount_rur", type: "num", cardinality: "continuous", description: "Transaction amount in rubles." },
      { name: "age", type: "label", cardinality: "age groups", description: "Target/grouping field, not an event input feature." }
    ],
    targets: [
      "age classification",
      "30-day amount regression",
      "same-date event forecasting",
      "amount coefficient-of-variation anomaly"
    ],
    preprocessing: [
      "frequency-code categorical columns",
      "collect client-level sequences",
      "filter short histories",
      "use a 30-day local horizon",
      "sample shift positions"
    ]
  },
  alphabattle: {
    summary: "Bank customer transaction histories used for product/default-related behavioral tasks.",
    domain: "Banking / customer transactions",
    scale: "1,466,527 sequences and 343M events in HT-Transformer statistics.",
    featureCount: "17 event features: 14 cat, 3 num",
    features: [
      { name: "client_id", type: "id", cardinality: "1,466,527 sequences in HT-Transformer statistics", description: "Customer sequence identifier." },
      { name: "time_from_first_trn", type: "time", cardinality: "hour offset", description: "Ordering field derived from transaction timing." },
      { name: "mcc_category", type: "cat", cardinality: "not reported in current notes", description: "Merchant category group." },
      { name: "currency", type: "cat", cardinality: "not reported in current notes", description: "Transaction currency code." },
      { name: "operation_kind", type: "cat", cardinality: "not reported in current notes", description: "High-level operation kind." },
      { name: "card_type", type: "cat", cardinality: "not reported in current notes", description: "Bank card type." },
      { name: "operation_type", type: "cat", cardinality: "not reported in current notes", description: "Detailed transaction operation type." },
      { name: "operation_type_group", type: "cat", cardinality: "not reported in current notes", description: "Grouped transaction operation type." },
      { name: "ecommerce_flag", type: "cat", cardinality: "binary or flag-coded", description: "Marks e-commerce transactions." },
      { name: "payment_system", type: "cat", cardinality: "not reported in current notes", description: "Payment system identifier." },
      { name: "income_flag", type: "cat", cardinality: "binary or flag-coded", description: "Marks income-like transactions." },
      { name: "mcc", type: "cat", cardinality: "not reported in current notes", description: "Merchant category code." },
      { name: "country", type: "cat", cardinality: "not reported in current notes", description: "Transaction country." },
      { name: "city", type: "cat", cardinality: "not reported in current notes", description: "Transaction city." },
      { name: "weekofyear", type: "cat", cardinality: "calendar week code", description: "Calendar week of transaction." },
      { name: "day_of_week", type: "cat", cardinality: "7 values", description: "Day of week." },
      { name: "amnt", type: "num", cardinality: "continuous", description: "Transaction amount." },
      { name: "hour", type: "num", cardinality: "0-23 when complete", description: "Hour of day." },
      { name: "hour_diff", type: "num", cardinality: "integer gap", description: "Elapsed hours between transactions." }
    ],
    targets: [
      "product classification",
      "transaction-amount regression",
      "event-count forecasting",
      "fraud/default-style anomaly flag"
    ],
    preprocessing: [
      "join transactions with product and flag targets",
      "frequency-code categorical columns",
      "order by transaction_number and derive time_from_first_trn",
      "collect client-level sequences",
      "use a 30-day local horizon",
      "sample shift positions"
    ]
  },
  "x5-retail": {
    summary: "Retail purchase histories from X5/RetailHero-style customer behavior data.",
    domain: "Retail / grocery purchases",
    scale: "45.8M purchases from 400k clients in the appendix description; 40k sequences in the distribution-forecasting setup.",
    featureCount: "17 event features: 8 cat, 9 num",
    features: [
      { name: "client_id", type: "id", cardinality: "400k clients in appendix stats", description: "Customer sequence identifier." },
      { name: "transaction_datetime", type: "time", cardinality: "event-level timestamp", description: "Ordering field." },
      { name: "age", type: "label", cardinality: "bucketized into age groups", description: "Target/grouping field, not an event input feature." },
      { name: "product_id", type: "cat", cardinality: "not reported in current notes", description: "Purchased product identifier." },
      { name: "is_own_trademark", type: "cat", cardinality: "2 values", description: "Private-label product flag." },
      { name: "is_alcohol", type: "cat", cardinality: "2 values", description: "Alcohol product flag." },
      { name: "level_1", type: "cat", cardinality: "not reported in current notes", description: "Product hierarchy level 1." },
      { name: "level_2", type: "cat", cardinality: "not reported in current notes", description: "Product hierarchy level 2." },
      { name: "level_3", type: "cat", cardinality: "not reported in current notes", description: "Product hierarchy level 3." },
      { name: "level_4", type: "cat", cardinality: "not reported in current notes", description: "Product hierarchy level 4." },
      { name: "segment_id", type: "cat", cardinality: "not reported in current notes", description: "Customer segment identifier." },
      { name: "purchase_sum", type: "num", cardinality: "continuous", description: "Purchase value." },
      { name: "trn_sum_from_red", type: "num", cardinality: "continuous", description: "Transaction sum redeemed from red/loyalty points." },
      { name: "trn_sum_from_iss", type: "num", cardinality: "continuous", description: "Transaction sum issued from the loyalty system." },
      { name: "netto", type: "num", cardinality: "continuous", description: "Product size or weight proxy." },
      { name: "regular_points_received", type: "num", cardinality: "continuous", description: "Regular loyalty points earned." },
      { name: "express_points_received", type: "num", cardinality: "continuous", description: "Express loyalty points earned." },
      { name: "product_quantity", type: "num", cardinality: "continuous/count", description: "Purchased quantity." },
      { name: "regular_points_spent", type: "num", cardinality: "continuous", description: "Regular loyalty points spent." },
      { name: "express_points_spent", type: "num", cardinality: "continuous", description: "Express loyalty points spent." }
    ],
    targets: [
      "age-bucket classification",
      "10-day purchase-sum regression",
      "same-hour purchase forecasting",
      "anomaly from loyalty-point redemption"
    ],
    preprocessing: [
      "merge purchases with products and clients",
      "filter client ages to the valid range used by preprocessing",
      "fill missing numeric columns with 0",
      "frequency-code categorical columns",
      "collect client-level sequences",
      "use a 10-day local horizon",
      "sample shift positions"
    ]
  },
  ett: {
    summary: "Regular electricity transformer temperature time series.",
    domain: "Energy / transformer monitoring",
    scale: "ETTh files: 17,420 rows each. ETTm files: 69,680 rows each. No missing points in the listed files.",
    featureCount: "7 numerical time-series features",
    features: [
      { name: "week_id", type: "id", cardinality: "weekly sequence groups", description: "Sequence identifier used by preprocessing." },
      { name: "time", type: "time", cardinality: "hourly or 15-minute grid", description: "Regular ordering field." },
      { name: "LUFL", type: "num", cardinality: "continuous", description: "Low-level useful load feature." },
      { name: "MUFL", type: "num", cardinality: "continuous", description: "Middle-level useful load feature." },
      { name: "MULL", type: "num", cardinality: "continuous", description: "Middle-level load feature." },
      { name: "LULL", type: "num", cardinality: "continuous", description: "Low-level load feature." },
      { name: "HULL", type: "num", cardinality: "continuous", description: "High-level load feature." },
      { name: "HUFL", type: "num", cardinality: "continuous", description: "High-level useful load feature." },
      { name: "OT", type: "num", cardinality: "continuous", description: "Oil temperature target/measurement." }
    ],
    targets: [
      "local forecasting target"
    ],
    preprocessing: [
      "keep regular time ordering",
      "group by week_id",
      "sample shift positions"
    ]
  },
  favorita: {
    summary: "Daily retail sales data with product-class and store-level structure.",
    domain: "Retail / store sales",
    scale: "125,497,040 train rows; 54 stores; 4,100 items in local notes.",
    featureCount: "dynamic numerical sales features after class pivot",
    features: [
      { name: "store_nbr", type: "id", cardinality: "54 stores", description: "Store-level sequence identifier." },
      { name: "date", type: "time", cardinality: "1,684 unique train dates in local notes", description: "Daily ordering field." },
      { name: "class_id", type: "cat", cardinality: "not reported in current notes", description: "Item class used before pivoting." },
      { name: "class_<id>_sales", type: "num", cardinality: "one numerical column per retained class_id after pivot", description: "Daily unit sales for each item class at each store." }
    ],
    targets: [
      "store-type classification",
      "rare-city anomaly",
      "30-day sales regression",
      "next-step sales forecasting"
    ],
    preprocessing: [
      "join train rows with item classes",
      "aggregate unit_sales by store, class and date",
      "zero-fill the full store-class-date grid",
      "pivot class sales into sequence columns",
      "merge store metadata for targets",
      "sample shift positions"
    ]
  },
  rossman: {
    summary: "Daily store sales observations with promotion and holiday indicators.",
    domain: "Retail / store sales",
    scale: "2013-2015 daily observations in local notes.",
    featureCount: "7 event features: 5 cat, 2 num",
    features: [
      { name: "Store", type: "id", cardinality: "store count not reported in current notes", description: "Store-level sequence identifier." },
      { name: "Date", type: "time", cardinality: "daily grid", description: "Ordering field." },
      { name: "DayOfWeek", type: "cat", cardinality: "7 values", description: "Calendar weekday." },
      { name: "Open", type: "cat", cardinality: "2 values", description: "Open/closed indicator." },
      { name: "Promo", type: "cat", cardinality: "2 values", description: "Promotion indicator." },
      { name: "StateHoliday", type: "cat", cardinality: "4 values in local notes: 0, a, b, c", description: "State holiday type." },
      { name: "SchoolHoliday", type: "cat", cardinality: "2 values", description: "School holiday indicator." },
      { name: "Sales", type: "num", cardinality: "continuous/count", description: "Daily sales." },
      { name: "Customers", type: "num", cardinality: "count", description: "Daily customer count." }
    ],
    targets: [
      "store-type classification",
      "60-day sales regression",
      "sales forecasting",
      "spike-ratio anomaly"
    ],
    preprocessing: [
      "frequency-code categorical columns",
      "collect store-level sequences",
      "use a global time split",
      "use a 60-day horizon",
      "sample shift positions"
    ]
  },
  taobao: {
    summary: "E-commerce user behavior sequences from Taobao.",
    domain: "E-commerce / clickstream",
    scale: "10k users; 9,904 sequences and 5M events in HT-Transformer statistics.",
    featureCount: "2 event features: 2 cat",
    features: [
      { name: "client_id", type: "id", cardinality: "9,904 sequences in HT-Transformer statistics", description: "User sequence identifier." },
      { name: "time", type: "time", cardinality: "event timestamp", description: "Ordering field." },
      { name: "behavior_type", type: "cat", cardinality: "4 action types in local notes", description: "User action such as view, cart, favorite or purchase." },
      { name: "item_category", type: "cat", cardinality: "about 8k categories in local notes", description: "Product category." }
    ],
    targets: [
      "purchase/view ratio classification over a 48-hour future window",
      "48-hour event-count regression",
      "same-hour forecasting",
      "item-action anomaly"
    ],
    preprocessing: [
      "frequency-code categorical columns",
      "collect client-level sequences",
      "use a 48-hour local horizon",
      "sample shift positions",
      "use item_id internally for anomaly construction"
    ]
  },
  twitter: {
    summary: "Tweet text converted into character-level sequences.",
    domain: "Social media / text",
    scale: "Uses training.1600000.processed.noemoticon.csv.",
    featureCount: "1 event feature: 1 cat",
    features: [
      { name: "tweet_id", type: "id", cardinality: "one sequence per retained tweet", description: "Tweet sequence identifier." },
      { name: "char_number", type: "time", cardinality: "character position", description: "Ordering field within a tweet." },
      { name: "char", type: "cat", cardinality: "observed cleaned character vocabulary; exact size depends on the loaded file", description: "Cleaned tweet character after frequency coding." }
    ],
    targets: [
      "sentiment classification",
      "mention-count regression",
      "punctuation-ratio anomaly"
    ],
    preprocessing: [
      "drop empty text and missing sentiment",
      "remove @ symbols while counting mentions",
      "frequency-code characters",
      "use a stratified sentiment split"
    ]
  },
  yambda: {
    summary: "Large-scale Yandex Music interaction data for recommendation and sequence modeling.",
    domain: "Music recommendation / streaming",
    scale: "Yambda-5B: 1M users, 9.39M items, 4.65B listens. Smaller 500M and 50M samples are also released.",
    featureCount: "4 event features: 3 cat, 1 num",
    features: [
      { name: "client_id", type: "id", cardinality: "up to 1M users in Yambda-5B", description: "User sequence identifier." },
      { name: "timestamp", type: "time", cardinality: "event timestamp", description: "Ordering field." },
      { name: "item_id", type: "cat", cardinality: "9.39M items in Yambda-5B", description: "Music item." },
      { name: "is_organic", type: "cat", cardinality: "2 values", description: "Organic versus system-driven exposure flag." },
      { name: "event_type", type: "cat", cardinality: "not reported in current notes", description: "Interaction type." },
      { name: "track_length_seconds", type: "num", cardinality: "continuous", description: "Track duration in seconds." }
    ],
    targets: [
      "organic-mode classification",
      "10-day listened-duration regression",
      "same-hour event forecasting",
      "dislike/listen-ratio anomaly"
    ],
    preprocessing: [
      "frequency-code categorical columns",
      "collect client-level sequences",
      "use a 10-day local horizon",
      "sample shift positions"
    ]
  },
  zvuk: {
    summary: "Music listening event sequences with track and cluster information.",
    domain: "Music recommendation / streaming",
    scale: "Raw/app stats: 244.7M events, 12.6M sessions, 382k users, 1.5M tracks. RecSys'24 preprocessing stats: 19,267 users, 146,894 items, 8.09M interactions.",
    featureCount: "3 event features: 2 cat, 1 num",
    features: [
      { name: "client_id", type: "id", cardinality: "19,267 users after RecSys'24 preprocessing", description: "User sequence identifier." },
      { name: "datetime", type: "time", cardinality: "event timestamp", description: "Ordering field." },
      { name: "track_id", type: "cat", cardinality: "146,894 items after RecSys'24 preprocessing; 1.5M raw tracks in app stats", description: "Listened track." },
      { name: "cluster_id", type: "cat", cardinality: "not reported in current notes", description: "Track/content cluster joined from track metadata." },
      { name: "play_duration", type: "num", cardinality: "continuous", description: "Listening duration." }
    ],
    targets: [
      "future dominant cluster classification",
      "40-day play-duration regression",
      "same-day event forecasting",
      "play-duration diversity anomaly"
    ],
    preprocessing: [
      "join interactions with track cluster metadata",
      "frequency-code categorical columns",
      "collect client-level sequences",
      "use a 40-day local horizon",
      "sample shift positions"
    ]
  },
  "electric-devices": {
    summary: "Electric device measurement sequences loaded from the UCR-style ElectricDevices train file.",
    domain: "Sensor / electric device measurements",
    scale: "Loaded from ElectricDevices_TRAIN.txt.",
    featureCount: "1 event feature: 1 num",
    features: [
      { name: "sequence_id", type: "id", cardinality: "one sequence per row in the train file", description: "Generated sequence identifier." },
      { name: "time", type: "time", cardinality: "generated integer position", description: "Ordering field." },
      { name: "sequence", type: "num", cardinality: "continuous", description: "Raw numerical measurement value at the generated time position." }
    ],
    targets: [
      "device-class classification",
      "next-step value-difference forecasting"
    ],
    preprocessing: [
      "load label from the first column",
      "use remaining columns as a numerical sequence",
      "use a global time split",
      "sample shift positions"
    ]
  }
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function format(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return Number(value).toFixed(digits);
}

function formatCompact(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  const number = Number(value);
  if (Math.abs(number) >= 1000) return number.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return number.toLocaleString("en-US", {
    minimumFractionDigits: Math.abs(number) < 10 ? digits : 2,
    maximumFractionDigits: Math.abs(number) < 10 ? digits : 2
  });
}

function percentFrom(value, min = 0, max = 1) {
  if (value === null || value === undefined || Number.isNaN(value) || max === min) return 0;
  return Math.max(0, Math.min(100, Math.round(((value - min) / (max - min)) * 100)));
}

function barScale(left = "0", middle = "50", right = "100") {
  return `
    <div class="bar-scale" aria-hidden="true">
      <span>${escapeHtml(left)}</span>
      <i></i>
      <span>${escapeHtml(middle)}</span>
      <i></i>
      <span>${escapeHtml(right)}</span>
    </div>
  `;
}

function bar(value) {
  const percent = Math.max(0, Math.min(100, Math.round((value || 0) * 100)));
  return `
    <span class="score-bar">
      <span class="mono">${percent}%</span>
      <span class="score-track"><span class="score-fill" style="--fill: ${percent}%"></span></span>
    </span>
  `;
}

async function loadData() {
  const response = await fetch("data/sber-bench.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load benchmark data: ${response.status}`);
  state.data = await response.json();
}

function hydrateStats() {
  document.querySelector("#stat-datasets").textContent = state.data.meta.datasets;
  document.querySelector("#stat-methods").textContent = state.data.meta.methods;
  document.querySelector("#stat-records").textContent = state.data.meta.records;
  document.querySelector("#stat-updated").textContent = state.data.meta.lastVerified;
}

function defaultDatasetId() {
  return state.data.datasets.find((dataset) => dataset.id === "age")?.id
    ?? state.data.datasets[0]?.id
    ?? "";
}

function setupControls() {
  selectors.datasetFilter.innerHTML = state.data.datasets
    .map((dataset) => `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.name)}</option>`)
    .join("");
  selectors.datasetFilter.value = defaultDatasetId();

  selectors.datasetFilter.addEventListener("change", () => {
    state.selected = null;
    render();
  });
  selectors.sortFilter.addEventListener("change", render);
  selectors.search.addEventListener("input", render);
}

function setupOverviewControls() {
  selectors.overviewTaskFilter.innerHTML = [
    `<option value="all">All tasks</option>`,
    ...state.data.tasks.map((task) => `<option value="${escapeHtml(task.id)}">${escapeHtml(task.label)}</option>`)
  ].join("");
  selectors.overviewTaskFilter.addEventListener("change", renderOverviewCharts);
  selectors.resetView.addEventListener("click", () => {
    selectors.datasetFilter.value = defaultDatasetId();
    selectors.sortFilter.value = "rank";
    selectors.search.value = "";
    state.selected = null;
    render();
    document.querySelector("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function setupAdditionalControls() {
  const ntpRecords = state.data.additional?.ntpLgbm?.records ?? [];
  const ntpDatasets = [...new Set(ntpRecords.map((record) => record.dataset))].sort();
  const ntpTasks = [...new Set(ntpRecords.map((record) => record.task))].sort();
  selectors.ntpDatasetFilter.innerHTML = [
    `<option value="all">All datasets</option>`,
    ...ntpDatasets.map((dataset) => `<option value="${escapeHtml(dataset)}">${escapeHtml(dataset)}</option>`)
  ].join("");
  selectors.ntpTaskFilter.innerHTML = [
    `<option value="all">All tasks</option>`,
    ...ntpTasks.map((task) => `<option value="${escapeHtml(task)}">${escapeHtml(task)}</option>`)
  ].join("");
  selectors.ntpDatasetFilter.addEventListener("change", renderNtpSection);
  selectors.ntpTaskFilter.addEventListener("change", renderNtpSection);

  selectors.sheetFilter.innerHTML = state.data.rawSheets
    .map((sheet) => `<option value="${sheet.id}">${escapeHtml(sheet.name)}</option>`)
    .join("");
  selectors.sheetFilter.value = state.data.rawSheets.find((sheet) => sheet.name === "RESULTS NEW from March")?.id
    ?? state.data.rawSheets[0]?.id
    ?? "";
  selectors.sheetFilter.addEventListener("change", renderRawSheet);
  selectors.sheetSearch.addEventListener("input", renderRawSheet);
}

function setupNavigationState() {
  const links = [...document.querySelectorAll(".topbar nav a")];
  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if (!("IntersectionObserver" in window)) return;

  const observer = new IntersectionObserver((entries) => {
    const active = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]?.target;
    if (!active) return;
    links.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${active.id}`);
    });
  }, {
    rootMargin: "-28% 0px -58% 0px",
    threshold: [0.04, 0.18, 0.36]
  });

  sections.forEach((section) => observer.observe(section));
}

function setupInteractiveSurfaces() {
  const surfaces = document.querySelectorAll(".chart-card, .summary-card, .dataset-card");
  surfaces.forEach((surface) => {
    if (surface.dataset.surfaceReady === "true") return;
    surface.dataset.surfaceReady = "true";
    surface.addEventListener("pointermove", (event) => {
      const rect = surface.getBoundingClientRect();
      surface.style.setProperty("--mx", `${event.clientX - rect.left}px`);
      surface.style.setProperty("--my", `${event.clientY - rect.top}px`);
    });
  });
}

function rowsForDataset() {
  const datasetId = selectors.datasetFilter.value;
  const query = selectors.search.value.trim().toLowerCase();
  return state.data.records
    .filter((record) => record.datasetId === datasetId)
    .filter((record) => !query || record.method.toLowerCase().includes(query));
}

function taskWinners(rows) {
  const winners = {};
  state.data.tasks.forEach((task) => {
    const winner = rows
      .filter((row) => row.scores[task.id] !== null && row.scores[task.id] !== undefined)
      .sort((a, b) => b.scores[task.id] - a.scores[task.id])[0];
    winners[task.id] = winner?.method ?? "";
  });
  return winners;
}

function taskRangesForRows(rows) {
  const ranges = {};
  state.data.tasks.forEach((task) => {
    const values = rows
      .map((record) => record.scores[task.id])
      .filter((value) => value !== null && value !== undefined);
    ranges[task.id] = {
      min: values.length ? Math.min(...values) : 0,
      max: values.length ? Math.max(...values) : 1
    };
  });
  return ranges;
}

function withComposite(rows) {
  const enriched = rows.map((row) => ({ ...row, composite: 0, compositeTasks: 0 }));
  state.data.tasks.forEach((task) => {
    const ranked = enriched
      .filter((row) => row.scores[task.id] !== null && row.scores[task.id] !== undefined)
      .sort((a, b) => b.scores[task.id] - a.scores[task.id]);
    const denominator = Math.max(1, ranked.length - 1);
    ranked.forEach((row, index) => {
      row.composite += ranked.length === 1 ? 1 : 1 - index / denominator;
      row.compositeTasks += 1;
    });
  });
  enriched.forEach((row) => {
    row.composite = row.compositeTasks ? row.composite / row.compositeTasks : 0;
  });
  return enriched;
}

function sortRows(rows) {
  const sort = selectors.sortFilter.value;
  if (sort === "rank") return rows.sort((a, b) => b.composite - a.composite);
  return rows.sort((a, b) => {
    const left = a.scores[sort] ?? Number.NEGATIVE_INFINITY;
    const right = b.scores[sort] ?? Number.NEGATIVE_INFINITY;
    return right - left;
  });
}

function globalTaskRanges() {
  const ranges = {};
  state.data.tasks.forEach((task) => {
    const values = state.data.records
      .map((record) => record.scores[task.id])
      .filter((value) => value !== null && value !== undefined);
    ranges[task.id] = {
      min: values.length ? Math.min(...values) : 0,
      max: values.length ? Math.max(...values) : 1
    };
  });
  return ranges;
}

function normalizeScore(value, range) {
  if (value === null || value === undefined || range.max === range.min) return null;
  return (value - range.min) / (range.max - range.min);
}

function scrollToResults() {
  document.querySelector("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function applyDatasetFilter(datasetId) {
  selectors.datasetFilter.value = datasetId;
  selectors.search.value = "";
  state.selected = null;
  render();
  scrollToResults();
}

function applyMethodSearch(methodName) {
  selectors.search.value = methodName;
  state.selected = null;
  render();
  scrollToResults();
}

function renderSummary(rows, winners) {
  const dataset = state.data.datasets.find((item) => item.id === selectors.datasetFilter.value);
  const datasetCard = `
    <article class="summary-card">
      <span>Dataset</span>
      <strong>${dataset?.name ?? "Unknown"}</strong>
      <small>${rows.length} visible rows</small>
    </article>
  `;
  const taskCards = state.data.tasks.map((task) => {
    const winner = winners[task.id];
    const row = rows.find((item) => item.method === winner);
    return `
      <article class="summary-card">
        <span>${task.label}</span>
        <strong>${winner || "No value"}</strong>
        <small>${task.metric}: ${row ? format(row.scores[task.id]) : "--"}</small>
      </article>
    `;
  });
  selectors.summary.innerHTML = datasetCard + taskCards.join("");
}

function scoreCell(record, taskId, winners, ranges) {
  const value = record.scores[taskId];
  if (value === null || value === undefined) return `<span class="score-missing">--</span>`;
  const fill = percentFrom(value, ranges[taskId]?.min, ranges[taskId]?.max);
  const winnerClass = winners[taskId] === record.method ? " metric-winner" : "";
  return `
    <span class="metric-cell${winnerClass}" style="--metric-fill: ${Math.max(4, fill)}%">
      <span class="metric-value mono">${format(value)}</span>
      <span class="metric-track" aria-hidden="true"><i></i></span>
      ${winnerClass ? `<small>best</small>` : ""}
    </span>
  `;
}

function renderTable(rows, winners) {
  const ranges = taskRangesForRows(rows);
  const dataset = state.data.datasets.find((item) => item.id === selectors.datasetFilter.value);
  const query = selectors.search.value.trim();
  selectors.tableStatus.innerHTML = `
    <span>${escapeHtml(dataset?.name ?? "Dataset")}</span>
    <strong>${rows.length} rows</strong>
    <small>${query ? `filtered by "${escapeHtml(query)}"` : "click a row to inspect exact hyperparameters"}</small>
  `;

  selectors.body.innerHTML = rows.map((record, index) => `
    <tr data-index="${index}" class="${state.selected?.method === record.method ? "active" : ""}">
      <td class="rank-cell mono">${index + 1}</td>
      <td>
        <span class="method-cell">
          <strong>${record.method}</strong>
          <small class="mono">${record.methodId}</small>
        </span>
      </td>
      <td>${bar(record.composite)}</td>
      <td>${scoreCell(record, "regression", winners, ranges)}</td>
      <td>${scoreCell(record, "classification", winners, ranges)}</td>
      <td>${scoreCell(record, "forecasting", winners, ranges)}</td>
      <td>${scoreCell(record, "anomaly", winners, ranges)}</td>
    </tr>
  `).join("");

  selectors.body.querySelectorAll("tr").forEach((element, index) => {
    element.addEventListener("click", () => {
      state.selected = rows[index];
      renderDetails(rows[index], winners);
      renderTable(rows, winners);
    });
  });
}

function renderCharts(rows, winners) {
  renderCompositeChart(rows);
  renderTaskChart(rows, winners);
}

function renderGlobalCharts() {
  const ranges = globalTaskRanges();
  renderMethodLandscape(ranges);
  renderDatasetHeatmap(ranges);
  renderMetricDistribution();
}

function renderOverviewCharts() {
  const taskId = selectors.overviewTaskFilter.value;
  renderDatasetDonut();
  renderTaskAvailabilityDonut();
  renderSourceDonut();
  renderScoreHistogram(taskId);
  renderCompositeHistogram();
  renderWinnerChart(taskId);
}

function renderDatasetDonut() {
  const entries = state.data.datasets
    .map((dataset) => ({
      label: dataset.name,
      value: dataset.records,
      datasetId: dataset.id
    }))
    .sort((a, b) => b.value - a.value);
  renderDonut(selectors.datasetDonut, entries, "rows");
}

function renderTaskAvailabilityDonut() {
  const entries = state.data.tasks.map((task) => ({
    label: task.label,
    value: state.data.records.filter((record) => record.scores[task.id] !== null && record.scores[task.id] !== undefined).length,
    taskId: task.id
  }));
  renderDonut(selectors.taskAvailabilityDonut, entries, "scores");
}

function renderSourceDonut() {
  const additional = state.data.additional;
  const notesCount = Object.values(additional.notes ?? {}).flat().length;
  const entries = [
    { label: "Main leaderboard", value: state.data.records.length, targetId: "results" },
    { label: "NTP_LGBM", value: additional.ntpLgbm.records.length, targetId: "additional" },
    { label: "Archive tables", value: additional.wideResults.records.length, targetId: "additional" },
    { label: "BERT_CORR trials", value: additional.validatorTrials.records.length, targetId: "additional" },
    { label: "Progress", value: additional.progress.records.length, targetId: "additional" },
    { label: "Papers / notes", value: additional.papers.records.length + notesCount, targetId: "additional" }
  ].filter((entry) => entry.value > 0);
  renderDonut(selectors.sourceDonut, entries, "rows");
}

function renderDonut(element, entries, unit) {
  const total = entries.reduce((sum, entry) => sum + entry.value, 0);
  let cursor = 0;
  const gradient = entries.map((entry, index) => {
    const start = total ? (cursor / total) * 360 : 0;
    cursor += entry.value;
    const end = total ? (cursor / total) * 360 : 360;
    return `${chartPalette[index % chartPalette.length]} ${start.toFixed(2)}deg ${end.toFixed(2)}deg`;
  }).join(", ");

  element.innerHTML = `
    <div
      class="donut"
      style="--donut-bg: conic-gradient(${gradient || "rgba(255,255,255,0.12) 0deg 360deg"})"
      aria-label="${total} ${unit}"
    >
      <span class="mono">${total}</span>
      <small>${escapeHtml(unit)}</small>
    </div>
    <div class="donut-legend">
      ${entries.map((entry, index) => {
        const percent = total ? Math.round((entry.value / total) * 100) : 0;
        const attrs = [
          entry.datasetId ? `data-dataset-id="${escapeHtml(entry.datasetId)}"` : "",
          entry.taskId ? `data-task-id="${escapeHtml(entry.taskId)}"` : "",
          entry.targetId ? `data-target-id="${escapeHtml(entry.targetId)}"` : ""
        ].filter(Boolean).join(" ");
        return `
          <button class="legend-item" type="button" ${attrs}>
            <i style="--swatch: ${chartPalette[index % chartPalette.length]}"></i>
            <span>${escapeHtml(entry.label)}</span>
            <b class="mono">${entry.value}</b>
            <small>${percent}%</small>
          </button>
        `;
      }).join("")}
    </div>
  `;

  element.querySelectorAll("[data-dataset-id]").forEach((button) => {
    button.addEventListener("click", () => applyDatasetFilter(button.dataset.datasetId));
  });
  element.querySelectorAll("[data-task-id]").forEach((button) => {
    button.addEventListener("click", () => {
      selectors.overviewTaskFilter.value = button.dataset.taskId;
      renderOverviewCharts();
    });
  });
  element.querySelectorAll("[data-target-id]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelector(`#${button.dataset.targetId}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

function scoreValuesForTask(taskId) {
  if (taskId === "all") {
    const ranges = globalTaskRanges();
    const values = [];
    state.data.records.forEach((record) => {
      state.data.tasks.forEach((task) => {
        const normalized = normalizeScore(record.scores[task.id], ranges[task.id]);
        if (normalized !== null) values.push(normalized);
      });
    });
    selectors.scoreHistogramLabel.textContent = "normalized all tasks";
    return values;
  }

  const task = state.data.tasks.find((item) => item.id === taskId);
  selectors.scoreHistogramLabel.textContent = `${task?.label ?? taskId} ${task?.metric ?? ""}`.trim();
  return state.data.records
    .map((record) => record.scores[taskId])
    .filter((value) => value !== null && value !== undefined);
}

function renderScoreHistogram(taskId) {
  renderHistogram(selectors.scoreHistogram, scoreValuesForTask(taskId), {
    bins: 12,
    precision: taskId === "all" ? 2 : 3,
    empty: "No scores for this task."
  });
}

function allCompositeRows() {
  return state.data.datasets.flatMap((dataset) => {
    const rows = state.data.records.filter((record) => record.datasetId === dataset.id);
    return withComposite(rows).filter((row) => row.compositeTasks > 0);
  });
}

function renderCompositeHistogram() {
  renderHistogram(selectors.compositeHistogram, allCompositeRows().map((row) => row.composite), {
    bins: 10,
    precision: 2,
    empty: "No composite values."
  });
}

function renderHistogram(element, values, options) {
  if (!values.length) {
    element.innerHTML = `<p class="empty-note">${escapeHtml(options.empty)}</p>`;
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binCount = max === min ? 1 : Math.max(1, options.bins);
  const width = max === min ? 1 : (max - min) / binCount;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    start: max === min ? min : min + width * index,
    end: max === min ? max : min + width * (index + 1),
    count: 0
  }));

  values.forEach((value) => {
    const index = max === min ? 0 : Math.min(binCount - 1, Math.floor((value - min) / width));
    bins[index].count += 1;
  });

  const maxCount = Math.max(...bins.map((bin) => bin.count), 1);
  const median = quantile(values, 0.5);
  const q1 = quantile(values, 0.25);
  const q3 = quantile(values, 0.75);
  element.innerHTML = `
    <div class="chart-stats">
      <span><small>min</small><b class="mono">${format(min, options.precision)}</b></span>
      <span><small>median</small><b class="mono">${format(median, options.precision)}</b></span>
      <span><small>max</small><b class="mono">${format(max, options.precision)}</b></span>
      <span><small>iqr</small><b class="mono">${format(q1, options.precision)}..${format(q3, options.precision)}</b></span>
    </div>
    <div class="histogram-bars">
      ${bins.map((bin) => {
        const height = Math.max(3, Math.round((bin.count / maxCount) * 100));
        const range = `${format(bin.start, options.precision)} .. ${format(bin.end, options.precision)}`;
        return `
          <button class="histogram-bin" type="button" style="--height: ${height}%" title="${range}: ${bin.count} values">
            <i></i>
            <span class="mono">${bin.count}</span>
            <small>${format(bin.start, options.precision)}</small>
          </button>
        `;
      }).join("")}
    </div>
    <div class="histogram-axis">
      <span class="mono">${format(min, options.precision)}</span>
      <span>${values.length} values</span>
      <span class="mono">${format(max, options.precision)}</span>
    </div>
  `;
}

function renderWinnerChart(taskId) {
  const wins = new Map();
  const tasks = taskId === "all"
    ? state.data.tasks
    : state.data.tasks.filter((task) => task.id === taskId);

  state.data.datasets.forEach((dataset) => {
    const rows = state.data.records.filter((record) => record.datasetId === dataset.id);
    tasks.forEach((task) => {
      const winner = rows
        .filter((row) => row.scores[task.id] !== null && row.scores[task.id] !== undefined)
        .sort((a, b) => b.scores[task.id] - a.scores[task.id])[0];
      if (!winner) return;
      const current = wins.get(winner.method) ?? { method: winner.method, count: 0, contexts: [] };
      current.count += 1;
      current.contexts.push(`${dataset.name} / ${task.label}`);
      wins.set(winner.method, current);
    });
  });

  const rows = [...wins.values()].sort((a, b) => b.count - a.count || a.method.localeCompare(b.method)).slice(0, 14);
  const max = Math.max(...rows.map((row) => row.count), 1);
  const body = rows.map((row) => {
    const percent = Math.round((row.count / max) * 100);
    return `
      <button class="bar-row interactive-row" type="button" data-method="${escapeHtml(row.method)}" title="${escapeHtml(row.contexts.join("\n"))}">
        <span>${escapeHtml(row.method)}</span>
        <div class="bar-track"><i style="--fill: ${percent}%"></i></div>
        <b class="mono">${row.count}</b>
      </button>
    `;
  }).join("");
  selectors.winnerChart.innerHTML = body
    ? `${barScale("0", String(Math.round(max / 2)), String(max))}${body}`
    : `<p class="empty-note">No winners for this task focus.</p>`;

  selectors.winnerChart.querySelectorAll("[data-method]").forEach((button) => {
    button.addEventListener("click", () => applyMethodSearch(button.dataset.method));
  });
}

function renderMethodLandscape(ranges) {
  const maxCoverage = Math.max(...state.data.methods.map((method) => method.records), 1);
  const methodStats = state.data.methods.map((method) => {
    const records = state.data.records.filter((record) => record.method === method.name);
    const datasets = new Set(records.map((record) => record.datasetId)).size;
    const normalizedScores = [];
    records.forEach((record) => {
      state.data.tasks.forEach((task) => {
        const normalized = normalizeScore(record.scores[task.id], ranges[task.id]);
        if (normalized !== null) normalizedScores.push(normalized);
      });
    });
    const score = normalizedScores.length
      ? normalizedScores.reduce((sum, value) => sum + value, 0) / normalizedScores.length
      : 0;
    return { ...method, datasets, score };
  }).sort((a, b) => b.score - a.score || b.records - a.records);

  selectors.methodLandscape.innerHTML = `
    <span class="axis-tick x-tick" style="--pos: 8%">0</span>
    <span class="axis-tick x-tick" style="--pos: 50%">coverage</span>
    <span class="axis-tick x-tick" style="--pos: 92%">${maxCoverage}</span>
    <span class="axis-tick y-tick" style="--pos: 10%">1.0</span>
    <span class="axis-tick y-tick" style="--pos: 49%">0.5</span>
    <span class="axis-tick y-tick" style="--pos: 88%">0</span>
    <span class="axis-label x-axis">coverage</span>
    <span class="axis-label y-axis">normalized score</span>
    ${methodStats.map((method, index) => {
      const x = 8 + (method.records / maxCoverage) * 84;
      const y = 10 + method.score * 78;
      const size = 0.78 + Math.sqrt(method.records) * 0.18;
      return `
        <button
          class="scatter-dot${index < 6 ? " prominent" : ""}"
          type="button"
          data-method="${escapeHtml(method.name)}"
          style="--x: ${x}%; --y: ${y}%; --size: ${size}rem; --tone: ${index % 6};"
          title="${escapeHtml(method.name)}: ${method.records} rows, ${method.datasets} datasets, normalized score ${format(method.score)}"
        >
          <span>${escapeHtml(method.name)}</span>
        </button>
      `;
    }).join("")}
  `;

  selectors.methodLandscape.querySelectorAll("[data-method]").forEach((button) => {
    button.addEventListener("click", () => applyMethodSearch(button.dataset.method));
  });
}

function renderDatasetHeatmap(ranges) {
  const header = state.data.tasks
    .map((task) => `<span class="heat-head">${task.label}</span>`)
    .join("");
  const body = state.data.datasets.map((dataset) => {
    const records = state.data.records.filter((record) => record.datasetId === dataset.id);
    const cells = state.data.tasks.map((task) => {
      const candidates = records
        .filter((record) => record.scores[task.id] !== null && record.scores[task.id] !== undefined)
        .sort((a, b) => b.scores[task.id] - a.scores[task.id]);
      const best = candidates[0];
      const normalized = best ? normalizeScore(best.scores[task.id], ranges[task.id]) : 0;
      const heat = Math.max(0.07, (normalized ?? 0) * 0.46).toFixed(3);
      return `
        <span
          class="heat-cell"
          style="--heat: ${heat}"
          title="${dataset.name} / ${task.label}: ${best ? best.method : "no value"}"
        >
          ${best ? format(best.scores[task.id]) : "--"}
        </span>
      `;
    }).join("");
    return `<span class="heat-method" title="${dataset.name}">${dataset.name}</span>${cells}`;
  }).join("");
  selectors.datasetHeatmap.innerHTML = `
    <div class="heatmap-scale">
      <span>low</span>
      <i></i>
      <span>high</span>
    </div>
    <div class="heatmap-grid dataset-heatmap-grid">
      <span class="heat-corner">Dataset</span>${header}${body}
    </div>
  `;
}

function quantile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  return sorted[base + 1] === undefined
    ? sorted[base]
    : sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}

function renderMetricDistribution() {
  selectors.metricDistribution.innerHTML = state.data.tasks.map((task) => {
    const values = state.data.records
      .map((record) => record.scores[task.id])
      .filter((value) => value !== null && value !== undefined)
      .sort((a, b) => a - b);
    if (!values.length) return "";
    const min = values[0];
    const max = values[values.length - 1];
    const q1 = quantile(values, 0.25);
    const median = quantile(values, 0.5);
    const q3 = quantile(values, 0.75);
    const scale = (value) => max === min ? 50 : ((value - min) / (max - min)) * 100;
    return `
      <div class="box-row">
        <span>${task.label}</span>
        <div class="box-track" title="${task.label}: ${values.length} values">
          <i class="box-whisker" style="--left: ${scale(min)}%; --width: ${scale(max) - scale(min)}%"></i>
          <i class="box-range" style="--left: ${scale(q1)}%; --width: ${Math.max(1, scale(q3) - scale(q1))}%"></i>
          <b style="--left: ${scale(median)}%"></b>
        </div>
        <small class="mono">${format(min)} .. ${format(max)}</small>
      </div>
    `;
  }).join("");
}

function renderAdditionalSections() {
  renderAdditionalSummary();
  renderNtpSection();
  renderArchiveChart();
  renderRawSheet();
}

function renderAdditionalSummary() {
  const meta = state.data.meta;
  const additional = state.data.additional;
  const cards = [
    ["Workbook sheets", meta.workbookSheets ?? state.data.rawSheets.length, `${meta.rawRows ?? 0} raw rows`],
    ["NTP_LGBM rows", additional.ntpLgbm.records.length, "classifier-level values"],
    ["Archive records", additional.wideResults.records.length, "wide result tables"],
    ["BERT_CORR trials", additional.validatorTrials.records.length, "validator trial rows"],
    ["Papers / notes", additional.papers.records.length + Object.values(additional.notes).flat().length, "reference rows"]
  ];
  selectors.additionalSummary.innerHTML = cards.map(([label, value, note]) => `
    <article class="summary-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </article>
  `).join("");
}

function renderNtpSection() {
  const dataset = selectors.ntpDatasetFilter.value;
  const task = selectors.ntpTaskFilter.value;
  const records = (state.data.additional.ntpLgbm.records ?? [])
    .filter((record) => dataset === "all" || record.dataset === dataset)
    .filter((record) => task === "all" || record.task === task)
    .sort((a, b) => b.value - a.value);
  const values = records.map((record) => record.value);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const body = records.slice(0, 18).map((record) => {
    const normalized = max === min ? 1 : (record.value - min) / (max - min);
    const percent = Math.max(3, Math.round(normalized * 100));
    const label = `${record.dataset} / ${record.method} / ${record.classifier}`;
    return `
      <div class="bar-row dense-row">
        <span title="${escapeHtml(label)}">${escapeHtml(label)}</span>
        <div class="bar-track"><i style="--fill: ${percent}%"></i></div>
        <b class="mono">${format(record.value)}</b>
      </div>
    `;
  }).join("");
  selectors.ntpChart.innerHTML = body
    ? `${barScale(formatCompact(min), formatCompact((min + max) / 2), formatCompact(max))}${body}`
    : `<p class="empty-note">No NTP_LGBM rows match these filters.</p>`;

  selectors.ntpBody.innerHTML = records.map((record) => `
    <tr>
      <td>${escapeHtml(record.dataset)}</td>
      <td>${escapeHtml(record.method)}</td>
      <td>${escapeHtml(record.task)}</td>
      <td>${escapeHtml(record.classifier)}</td>
      <td class="mono">${format(record.value)}</td>
    </tr>
  `).join("");
}

function renderArchiveChart() {
  const counts = new Map();
  (state.data.additional.wideResults.records ?? []).forEach((record) => {
    counts.set(record.sourceSheet, (counts.get(record.sourceSheet) ?? 0) + 1);
  });
  const rows = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const max = Math.max(...rows.map(([, count]) => count), 1);
  const body = rows.map(([sheet, count]) => {
    const percent = Math.round((count / max) * 100);
    return `
      <div class="bar-row">
        <span title="${escapeHtml(sheet)}">${escapeHtml(sheet)}</span>
        <div class="bar-track"><i style="--fill: ${percent}%"></i></div>
        <b class="mono">${count}</b>
      </div>
    `;
  }).join("");
  selectors.archiveChart.innerHTML = body
    ? `${barScale("0", String(Math.round(max / 2)), String(max))}${body}`
    : `<p class="empty-note">No archive records.</p>`;
}

function columnName(index) {
  let number = index + 1;
  let name = "";
  while (number > 0) {
    const remainder = (number - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    number = Math.floor((number - 1) / 26);
  }
  return name;
}

function visibleSheetRows(sheet) {
  const query = selectors.sheetSearch.value.trim().toLowerCase();
  if (!query) return sheet.rows;
  return sheet.rows.filter((row) => row.some((value) => String(value).toLowerCase().includes(query)));
}

function renderRawSheet() {
  const sheet = state.data.rawSheets.find((item) => item.id === selectors.sheetFilter.value) ?? state.data.rawSheets[0];
  if (!sheet) return;
  const rows = visibleSheetRows(sheet);
  const columnCount = Math.max(sheet.columnCount, ...rows.map((row) => row.length), 0);
  selectors.sheetSummary.innerHTML = `
    <span>${escapeHtml(sheet.name)}</span>
    <strong>${rows.length} / ${sheet.rowCount} rows</strong>
    <small>${columnCount} columns preserved from workbook</small>
  `;
  renderRawSheetChart(rows, columnCount);
  selectors.sheetHead.innerHTML = `
    <tr>
      <th>#</th>
      ${Array.from({ length: columnCount }, (_, index) => `<th>${columnName(index)}</th>`).join("")}
    </tr>
  `;
  selectors.sheetBody.innerHTML = rows.map((row, rowIndex) => `
    <tr>
      <td class="mono">${rowIndex + 1}</td>
      ${Array.from({ length: columnCount }, (_, index) => `<td>${escapeHtml(row[index] ?? "")}</td>`).join("")}
    </tr>
  `).join("");
}

function renderRawSheetChart(rows, columnCount) {
  const numericColumns = Array.from({ length: columnCount }, (_, index) => {
    const values = rows
      .map((row) => Number(String(row[index] ?? "").replace(",", ".")))
      .filter((value) => Number.isFinite(value));
    return {
      index,
      count: values.length,
      min: values.length ? Math.min(...values) : null,
      max: values.length ? Math.max(...values) : null
    };
  }).filter((column) => column.count > 0).sort((a, b) => b.count - a.count).slice(0, 12);

  const maxCount = Math.max(...numericColumns.map((column) => column.count), 1);
  const body = numericColumns.map((column) => {
    const percent = Math.round((column.count / maxCount) * 100);
    return `
      <div class="bar-row dense-row">
        <span>Column ${columnName(column.index)}</span>
        <div class="bar-track"><i style="--fill: ${percent}%"></i></div>
        <b class="mono">${column.count}</b>
        <small class="range-note">${format(column.min)} .. ${format(column.max)}</small>
      </div>
    `;
  }).join("");
  selectors.sheetChart.innerHTML = body
    ? `${barScale("0", String(Math.round(maxCount / 2)), String(maxCount))}${body}`
    : `<p class="empty-note">No numeric columns in visible rows.</p>`;
}

function renderCompositeChart(rows) {
  const topRows = rows.slice(0, 12);
  const body = topRows.map((record, index) => {
    const percent = Math.round(record.composite * 100);
    return `
      <button class="bar-row ranked-row" type="button" data-method="${escapeHtml(record.method)}">
        <em class="mono">${index + 1}</em>
        <span title="${escapeHtml(record.method)}">${escapeHtml(record.method)}</span>
        <div class="bar-track">
          <i style="--fill: ${percent}%"></i>
        </div>
        <b class="mono">${percent}%</b>
      </button>
    `;
  }).join("");
  selectors.compositeChart.innerHTML = body
    ? `${barScale("0%", "50%", "100%")}${body}`
    : `<p class="empty-note">No rows match the current filters.</p>`;
  selectors.compositeChart.querySelectorAll("[data-method]").forEach((button) => {
    button.addEventListener("click", () => applyMethodSearch(button.dataset.method));
  });
}

function renderTaskChart(rows, winners) {
  const topRows = rows.slice(0, 9);
  const ranges = {};
  state.data.tasks.forEach((task) => {
    const values = rows
      .map((record) => record.scores[task.id])
      .filter((value) => value !== null && value !== undefined);
    ranges[task.id] = {
      min: values.length ? Math.min(...values) : 0,
      max: values.length ? Math.max(...values) : 1
    };
  });
  const headers = state.data.tasks
    .map((task) => `<span class="heat-head">${task.label}</span>`)
    .join("");
  const body = topRows.map((record) => {
    const cells = state.data.tasks.map((task) => {
      const value = record.scores[task.id];
      const range = ranges[task.id];
      const normalized = value === null || value === undefined || range.max === range.min
        ? 0
        : (value - range.min) / (range.max - range.min);
      const heat = Math.max(0.07, normalized * 0.46).toFixed(3);
      const winnerClass = winners[task.id] === record.method ? " winner" : "";
      return `
        <span
          class="heat-cell${winnerClass}"
          style="--heat: ${heat}"
          title="${record.method} / ${task.label}: ${format(value)}"
        >${format(value)}</span>
      `;
    }).join("");
    return `<span class="heat-method" title="${record.method}">${record.method}</span>${cells}`;
  }).join("");
  selectors.taskChart.innerHTML = `
    <div class="heatmap-scale">
      <span>low</span>
      <i></i>
      <span>best</span>
    </div>
    <div class="heatmap-grid task-heatmap-grid">
      <span class="heat-corner">Method</span>
      ${headers}
      ${body || `<span class="empty-note">No rows</span>`}
    </div>
  `;
}

function detailMap(record, source) {
  const entries = Object.entries(source ?? {});
  if (!entries.length) return "--";
  return entries.map(([task, value]) => `${task}: ${value}`).join("\n");
}

function scoreMap(record) {
  return state.data.tasks
    .map((task) => `${task.label} ${task.metric}: ${format(record.scores[task.id])}`)
    .join("\n");
}

function renderDetails(record, winners) {
  if (!record) return;
  state.selected = record;
  const bestTasks = Object.entries(winners)
    .filter(([, method]) => method === record.method)
    .map(([task]) => task)
    .join(", ");
  document.querySelector("#detail-method").textContent = record.method;
  document.querySelector("#detail-dataset").textContent = record.dataset;
  document.querySelector("#detail-best").textContent = bestTasks || "not best on visible rows";
  document.querySelector("#detail-scores").textContent = scoreMap(record);
  document.querySelector("#detail-params").textContent = detailMap(record, record.params);
  document.querySelector("#detail-masking").textContent = detailMap(record, record.masking);
}

function datasetTypeCounts(features = []) {
  return features.reduce((acc, feature) => {
    acc[feature.type] = (acc[feature.type] ?? 0) + 1;
    return acc;
  }, {});
}

function renderTypePills(features = []) {
  const counts = datasetTypeCounts(features);
  return ["cat", "num", "time", "id", "label"]
    .filter((type) => counts[type])
    .map((type) => `
      <span class="type-pill type-${escapeHtml(type)}">
        ${escapeHtml(type)} <b>${counts[type]}</b>
      </span>
    `).join("");
}

function renderCompactList(items = []) {
  if (!items.length) return `<p class="muted-text">No curated notes yet.</p>`;
  return `
    <ul>
      ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ul>
  `;
}

function renderFeatureTable(features = []) {
  if (!features.length) {
    return `<p class="muted-text">Feature metadata is not registered for this dataset yet.</p>`;
  }
  return `
    <div class="dataset-feature-shell">
      <table class="dataset-feature-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Type</th>
            <th>Cardinality</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          ${features.map((feature) => `
            <tr>
              <td><code>${escapeHtml(feature.name)}</code></td>
              <td><span class="type-pill type-${escapeHtml(feature.type)}">${escapeHtml(feature.type)}</span></td>
              <td>${escapeHtml(feature.cardinality)}</td>
              <td>${escapeHtml(feature.description)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function openDatasetModal(datasetId) {
  const dataset = state.data.datasets.find((item) => item.id === datasetId);
  if (!dataset || !selectors.datasetModal || !selectors.datasetModalContent) return;
  const info = datasetInfo[dataset.id] ?? {};
  const features = info.features ?? [];
  selectors.datasetModalContent.innerHTML = `
    <header class="dataset-dialog-header">
      <div>
        <p class="kicker">${escapeHtml(dataset.id)}</p>
        <h3 id="dataset-modal-title">${escapeHtml(dataset.name)}</h3>
        <p>${escapeHtml(info.summary ?? "Benchmark dataset used in the current result table.")}</p>
      </div>
      <div class="dataset-dialog-stats" aria-label="Dataset quick statistics">
        <span><b>${escapeHtml(dataset.methods)}</b> methods</span>
        <span><b>${escapeHtml(dataset.records)}</b> rows</span>
      </div>
    </header>

    <div class="dataset-dialog-grid">
      <section>
        <h4>Domain</h4>
        <p>${escapeHtml(info.domain ?? "Not registered yet.")}</p>
      </section>
      <section>
        <h4>Scale</h4>
        <p>${escapeHtml(info.scale ?? "Not registered yet.")}</p>
      </section>
      <section>
        <h4>Feature Set</h4>
        <p>${escapeHtml(info.featureCount ?? "Not registered yet.")}</p>
        <div class="dataset-type-strip">${renderTypePills(features)}</div>
      </section>
    </div>

    <section class="dataset-dialog-section">
      <h4>Selected Features</h4>
      ${renderFeatureTable(features)}
    </section>

    <div class="dataset-dialog-columns">
      <section class="dataset-dialog-section">
        <h4>Targets</h4>
        ${renderCompactList(info.targets)}
      </section>
      <section class="dataset-dialog-section">
        <h4>Preprocessing</h4>
        ${renderCompactList(info.preprocessing)}
      </section>
    </div>
  `;

  if (selectors.datasetModal.open && typeof selectors.datasetModal.close === "function") {
    selectors.datasetModal.close();
  }
  if (typeof selectors.datasetModal.showModal === "function") {
    selectors.datasetModal.showModal();
  } else {
    selectors.datasetModal.setAttribute("open", "");
  }
}

function closeDatasetModal() {
  if (!selectors.datasetModal) return;
  if (typeof selectors.datasetModal.close === "function") {
    selectors.datasetModal.close();
  } else {
    selectors.datasetModal.removeAttribute("open");
  }
}

function setupDatasetDialog() {
  if (!selectors.datasetModal) return;
  selectors.datasetModalClose?.addEventListener("click", closeDatasetModal);
  selectors.datasetModal.addEventListener("click", (event) => {
    if (event.target === selectors.datasetModal) closeDatasetModal();
  });
}

function renderDatasetCards() {
  selectors.cards.innerHTML = state.data.datasets.map((dataset) => {
    const info = datasetInfo[dataset.id];
    return `
      <article class="dataset-card">
        <button class="dataset-card-main" type="button" data-dataset-id="${escapeHtml(dataset.id)}">
          <span class="dataset-kicker">${escapeHtml(dataset.id)}</span>
          <h3>${escapeHtml(dataset.name)}</h3>
          <p>${escapeHtml(info?.summary ?? "Benchmark dataset used in the current result table.")}</p>
          <div class="dataset-card-tags">
            <span>${escapeHtml(info?.domain ?? "Domain pending")}</span>
            <span>${escapeHtml(info?.featureCount ?? "Feature notes pending")}</span>
          </div>
        </button>
        <div class="dataset-card-actions">
          <button class="dataset-detail-button" type="button" data-dataset-details-id="${escapeHtml(dataset.id)}">
            Open dataset card
          </button>
          <span>${escapeHtml(dataset.methods)} methods / ${escapeHtml(dataset.records)} rows</span>
        </div>
      </article>
    `;
  }).join("");

  selectors.cards.querySelectorAll("[data-dataset-id]").forEach((button) => {
    button.addEventListener("click", () => applyDatasetFilter(button.dataset.datasetId));
  });
  selectors.cards.querySelectorAll("[data-dataset-details-id]").forEach((button) => {
    button.addEventListener("click", () => openDatasetModal(button.dataset.datasetDetailsId));
  });
}

function render() {
  const rows = sortRows(withComposite(rowsForDataset()));
  const winners = taskWinners(rows);
  renderSummary(rows, winners);
  renderCharts(rows, winners);
  renderTable(rows, winners);
  const selectedStillVisible = state.selected && rows.some((row) => row.method === state.selected.method);
  renderDetails(selectedStillVisible ? state.selected : rows[0], winners);
  setupInteractiveSurfaces();
}

async function init() {
  try {
    await loadData();
    hydrateStats();
    setupControls();
    setupOverviewControls();
    setupAdditionalControls();
    setupDatasetDialog();
    setupNavigationState();
    renderDatasetCards();
    renderOverviewCharts();
    renderGlobalCharts();
    renderAdditionalSections();
    render();
    setupInteractiveSurfaces();
  } catch (error) {
    document.querySelector("main").innerHTML = `
      <section class="panel">
        <p class="kicker">Data error</p>
        <h2>Could not load benchmark results.</h2>
        <p class="panel-note">${error.message}</p>
      </section>
    `;
  }
}

init();
