/*
 * Editable content layer for the project page.
 *
 * Tables below are transcribed from the 2026 preprint. Change values here;
 * app.js is only responsible for presentation and interaction.
 */
window.BENCHMARK_RESULTS = {
  rankings: [
    { method: "CoLES", regression: 2.10, classification: 2.73, forecasting: 3.73, anomaly: 2.10, overall: 2.69 },
    { method: "Slice-InfoNCE", regression: 3.15, classification: 2.55, forecasting: 4.77, anomaly: 3.20, overall: 3.43 },
    { method: "NTP-GRU", regression: 3.80, classification: 4.00, forecasting: 4.14, anomaly: 4.40, overall: 4.08 },
    { method: "NTP-GPT", regression: 5.20, classification: 3.95, forecasting: 3.64, anomaly: 5.25, overall: 4.48 },
    { method: "Mask-InfoNCE", regression: 4.75, classification: 4.45, forecasting: 5.18, anomaly: 3.95, overall: 4.60 },
    { method: "JEPA", regression: 5.70, classification: 5.09, forecasting: 4.14, anomaly: 4.05, overall: 4.74 },
    { method: "MLM", regression: 6.30, classification: 5.86, forecasting: 5.32, anomaly: 5.50, overall: 5.74 },
    { method: "TSFresh", regression: 5.00, classification: 7.36, forecasting: 5.09, anomaly: 7.55, overall: 6.25 }
  ],

  datasets: [
    {
      id: "age", name: "AGE", domain: "Transactions", domainKey: "trx",
      structure: "Irregular · categorical + numerical", cardinality: "50–1k categories",
      sequences: "29,964", events: "26,420,179", meanLength: "881.73",
      localTasks: 2, globalTasks: 2, shifts: "9 train / 1 test", globalSplit: "27,000 / 3,000",
      summary: "Anonymised bank-card histories with transaction category and payment amount.",
      targets: [
        ["Classification", "Age group · 4 classes", "Global", "24.8 / 25.0 / 25.2 / 25.0%"],
        ["Regression", "Future purchase amount", "30 days", "R²"],
        ["Forecasting", "Next-window event count", "1 day", "R²"],
        ["Anomaly", "Payment-amount coefficient of variation above P95", "Global", "5.00%"]
      ]
    },
    {
      id: "alphabattle", name: "AlphaBattle", domain: "Transactions", domainKey: "trx",
      structure: "Irregular · categorical + numerical", cardinality: "50–1k categories",
      sequences: "1,457,032", events: "443,407,241", meanLength: "304.32",
      localTasks: 2, globalTasks: 2, shifts: "9 train / 1 test", globalSplit: "861,481 / 95,721",
      summary: "Large-scale bank transaction histories with product and client-default labels.",
      targets: [
        ["Classification", "Application product · 5 product IDs", "Global", "51.3 / 29.8 / 6.2 / 7.9 / 4.7%"],
        ["Regression", "Future transaction amount", "30 days", "R²"],
        ["Forecasting", "Next-window event count", "1 hour", "R²"],
        ["Anomaly", "Client default", "Global", "2.83%"]
      ]
    },
    {
      id: "retail", name: "x5-Retail", domain: "Transactions", domainKey: "trx",
      structure: "Irregular · categorical + numerical", cardinality: "50–1k categories",
      sequences: "212,322", events: "33,490,385", meanLength: "157.73",
      localTasks: 3, globalTasks: 1, shifts: "9 train / 1 test", globalSplit: "318,074 / 35,342",
      summary: "Retail purchase histories combining products, amounts and loyalty activity.",
      targets: [
        ["Classification", "Age quantile · 4 classes", "Global", "26.8 / 22.1 / 26.9 / 24.1%"],
        ["Regression", "Future purchase amount", "10 days", "R²"],
        ["Forecasting", "Next-window event count", "1 hour", "R²"],
        ["Anomaly", "Loyalty-point payment greater than 10", "10 days", "11.15%"]
      ]
    },
    {
      id: "taobao", name: "Taobao", domain: "Recommender systems", domainKey: "recsys",
      structure: "Irregular · categorical + numerical", cardinality: "100k–10M categories",
      sequences: "7,437", events: "11,289,272", meanLength: "1,517.99",
      localTasks: 3, globalTasks: 1, shifts: "9 train / 1 test", globalSplit: "8,749 / 973",
      summary: "Dense e-commerce clickstream with view, cart, favourite and purchase events.",
      targets: [
        ["Classification", "Purchase/view ratio · 2 quantiles", "48 hours", "75 / 25%"],
        ["Regression", "Future purchase amount", "48 hours", "R²"],
        ["Forecasting", "Next-window event count", "1 hour", "R²"],
        ["Anomaly", "Purchase immediately after liking an item", "Global", "4.59%"]
      ]
    },
    {
      id: "zvuk", name: "Zvuk", domain: "Recommender systems", domainKey: "recsys",
      structure: "Irregular · categorical + numerical", cardinality: "100k–10M categories",
      sequences: "62,753", events: "143,643,940", meanLength: "2,289.04",
      localTasks: 4, globalTasks: 0, shifts: "9 train / 1 test", globalSplit: "—",
      summary: "Music listening histories with tracks, clusters and play duration.",
      targets: [
        ["Classification", "Future listening genre", "40 days", "Top-5 classes"],
        ["Regression", "Future listening volume", "40 days", "R²"],
        ["Forecasting", "Next-window event count", "1 day", "R²"],
        ["Anomaly", "Listening-duration coefficient of variation above P95", "40 days", "5.00%"]
      ]
    },
    {
      id: "30music", name: "30Music", domain: "Recommender systems", domainKey: "recsys",
      structure: "Irregular · categorical + numerical", cardinality: "100k–10M categories",
      sequences: "21,572", events: "22,734,503", meanLength: "1,053.89",
      localTasks: 2, globalTasks: 2, shifts: "9 train / 1 test", globalSplit: "35,934 / 3,993",
      summary: "Long-form music interactions with track identity and listening duration.",
      targets: [
        ["Classification", "Listening-time variability · 4 classes", "Global", "25 / 25 / 25 / 25%"],
        ["Regression", "Future listening duration", "3 days", "R²"],
        ["Forecasting", "Next-window event count", "1 day", "R²"],
        ["Anomaly", "Track diversity > 0.95 and mean duration < 0.05", "Global", "0.24%"]
      ]
    },
    {
      id: "yambda", name: "Yambda", domain: "Recommender systems", domainKey: "recsys",
      structure: "Irregular · categorical + numerical", cardinality: "100k–10M categories",
      sequences: "8,280", events: "46,355,169", meanLength: "5,598.45",
      localTasks: 2, globalTasks: 2, shifts: "9 train / 1 test", globalSplit: "8,277 / 920",
      summary: "Recommendation logs separating organic and recommended listening events.",
      targets: [
        ["Classification", "Recommended vs. organic listening", "Global", "53 / 47%"],
        ["Regression", "Future listening volume", "10 days", "R²"],
        ["Forecasting", "Next-window event count", "1 hour", "R²"],
        ["Anomaly", "Dislike-to-listen ratio above P95", "Global", "5.01%"]
      ]
    },
    {
      id: "electric-devices", name: "Electric Devices", domain: "Time series", domainKey: "ts",
      structure: "Regular · numerical", cardinality: "Univariate",
      sequences: "8,926", events: "856,896", meanLength: "96.00",
      localTasks: 1, globalTasks: 1, shifts: "27 train / 3 test", globalSplit: "8,033 / 893",
      summary: "Fixed-length electricity-consumption sequences from seven device classes.",
      targets: [
        ["Classification", "Device type", "Global", "7 classes"],
        ["Forecasting", "Energy change", "1 step", "R²"]
      ]
    },
    {
      id: "ett", name: "ETT", domain: "Time series", domainKey: "ts",
      structure: "Regular · numerical", cardinality: "Multivariate",
      sequences: "206", events: "138,432", meanLength: "672.00",
      localTasks: 1, globalTasks: 0, shifts: "90 train / 10 test", globalSplit: "—",
      summary: "Electricity transformer temperature and load measurements on a regular grid.",
      targets: [["Forecasting", "Oil-temperature change", "1 step", "R²"]]
    },
    {
      id: "favorita", name: "Favorita", domain: "Time series", domainKey: "ts",
      structure: "Regular · numerical", cardinality: "Multivariate",
      sequences: "54", events: "90,936", meanLength: "1,684.00",
      localTasks: 2, globalTasks: 2, shifts: "90 train / 10 test", globalSplit: "48 / 6",
      summary: "Daily item-class demand grouped into store-level multivariate sequences.",
      targets: [
        ["Classification", "Store type · 5 classes", "Global", "33.96 / 28.30 / 15.09 / 15.09 / 7.55%"],
        ["Regression", "Future item demand", "30 days", "R²"],
        ["Forecasting", "Next-window sales demand", "1 day", "R²"],
        ["Anomaly", "Only store in its city", "Global", "21.00%"]
      ]
    },
    {
      id: "rossmann", name: "Rossmann", domain: "Time series", domainKey: "ts",
      structure: "Regular · numerical", cardinality: "Multivariate",
      sequences: "1,115", events: "1,017,209", meanLength: "912.30",
      localTasks: 3, globalTasks: 1, shifts: "90 train / 10 test", globalSplit: "1,003 / 112",
      summary: "Daily store sales with promotions, customers and holiday indicators.",
      targets: [
        ["Classification", "Store type", "Global", "Default target"],
        ["Regression", "Future sales volume", "60 days", "R²"],
        ["Forecasting", "Next-window sales demand", "1 day", "R²"],
        ["Anomaly", "Future max-to-median sales ratio", "60 days", "5.00%"]
      ]
    },
    {
      id: "twitter", name: "Twitter", domain: "Text", domainKey: "text",
      structure: "No timestamps · categorical", cardinality: "30–100 categories",
      sequences: "1,600,000", events: "117,745,496", meanLength: "73.59",
      localTasks: 0, globalTasks: 3, shifts: "0", globalSplit: "1,440,000 / 160,000",
      summary: "Character-level tweet sequences with sentiment and behavioural targets.",
      targets: [
        ["Classification", "Sentiment · 4 classes", "Global", "54.03 / 31.13 / 13.30 / 1.54%"],
        ["Regression", "Mention count", "Global", "R²"],
        ["Anomaly", "Special-character count normalised by length", "Global", "2.00%"]
      ]
    }
  ],

  validators: [
    { scope: "Average", lightgbm: 1.652, mlp: 2.000, linear: 2.043 },
    { scope: "Classification", lightgbm: 1.167, mlp: 2.167, linear: 2.500 },
    { scope: "Forecasting", lightgbm: 2.000, mlp: 1.400, linear: 1.800 },
    { scope: "Regression", lightgbm: 1.333, mlp: 2.333, linear: 2.333 },
    { scope: "Anomaly", lightgbm: 2.167, mlp: 2.000, linear: 1.500 }
  ],

  multiTarget: [
    { dataset: "x5-Retail", model: "NTP-GRU", regime: "Single", regression: "0.160", classification: "0.454", forecasting: "0.431", anomaly: "0.683" },
    { dataset: "x5-Retail", model: "NTP-GRU", regime: "Multi", regression: "0.121", classification: "0.443", forecasting: "0.428", anomaly: "0.663" },
    { dataset: "x5-Retail", model: "JEPA", regime: "Single", regression: "0.125", classification: "0.463", forecasting: "0.325", anomaly: "0.691" },
    { dataset: "x5-Retail", model: "JEPA", regime: "Multi", regression: "0.146", classification: "0.441", forecasting: "0.336", anomaly: "0.692" },
    { dataset: "Twitter", model: "NTP-GRU", regime: "Single", regression: "0.685", classification: "0.744", forecasting: "—", anomaly: "0.997" },
    { dataset: "Twitter", model: "NTP-GRU", regime: "Multi", regression: "0.677", classification: "0.751", forecasting: "—", anomaly: "0.997" },
    { dataset: "Twitter", model: "JEPA", regime: "Single", regression: "0.450", classification: "0.647", forecasting: "—", anomaly: "0.998" },
    { dataset: "Twitter", model: "JEPA", regime: "Multi", regression: "0.480", classification: "0.646", forecasting: "—", anomaly: "0.994" }
  ],

  fusion: {
    main: {
      label: "Table 6 · AGE · Slice-InfoNCE + NTP-GRU · Linear probe",
      rows: [
        ["Slice-InfoNCE (classification)", "0.630 ± 0.000", "0.459 ± 0.002", "0.244 ± 0.001", "0.745 ± 0.002"],
        ["NTP-GRU (forecasting)", "0.602 ± 0.003", "0.491 ± 0.000", "0.411 ± 0.002", "0.673 ± 0.006"],
        ["Concatenation", "0.630 ± 0.002", "0.498 ± 0.002", "0.411 ± 0.001", "0.741 ± 0.006"],
        ["PCA", "0.627 ± 0.002", "0.494 ± 0.001", "0.404 ± 0.000", "0.757 ± 0.010"],
        ["CCA", "0.620 ± 0.003", "0.486 ± 0.000", "0.321 ± 0.001", "0.725 ± 0.005"],
        ["KrossFuse", "0.621 ± 0.003", "0.454 ± 0.001", "0.358 ± 0.001", "0.685 ± 0.005"],
        ["TuckerFactorConcat", "0.628 ± 0.001", "0.499 ± 0.001", "0.414 ± 0.001", "0.746 ± 0.007"]
      ]
    },
    studies: [
      {
        id: "table20", label: "Table 20 · AGE · CoLES + NTP-GRU",
        note: "Regression + forecasting specialists · linear probe",
        rows: [
          ["CoLES", "0.597", "0.511", "0.385", "0.777"], ["NTP-GRU", "0.602", "0.491", "0.411", "0.673"],
          ["CCA", "0.590", "0.510", "0.412", "0.764"], ["KrossFuse", "0.608", "0.500", "0.407", "0.749"],
          ["PCA", "0.617", "0.512", "0.418", "0.785"], ["TuckerFactorConcat", "0.620", "0.513", "0.420", "0.799"],
          ["Concatenation", "0.610", "0.511", "0.419", "0.780"]
        ]
      },
      {
        id: "table21", label: "Table 21 · AGE · Slice-InfoNCE + JEPA",
        note: "Classification + anomaly specialists · linear probe",
        rows: [
          ["Slice-InfoNCE", "0.630", "0.459", "0.244", "0.745"], ["JEPA", "0.442", "0.378", "−0.011", "0.982"],
          ["CCA", "0.597", "—", "—", "0.932"], ["KrossFuse", "0.627", "0.424", "0.226", "0.868"],
          ["PCA", "0.623", "0.460", "0.230", "0.982"], ["TuckerFactorConcat", "0.623", "0.459", "0.223", "0.988"],
          ["Concatenation", "0.620", "0.448", "0.228", "0.981"]
        ]
      },
      {
        id: "table28", label: "Table 28 · Rossmann · CoLES + Slice-InfoNCE",
        note: "Regression + classification specialists · linear probe",
        rows: [
          ["Slice-InfoNCE", "0.729", "0.951", "−0.606", "0.901"], ["CoLES", "0.637", "0.972", "−1.032", "0.747"],
          ["CCA", "0.208", "0.573", "−2.062", "0.722"], ["KrossFuse", "0.583", "0.933", "−1.615", "0.716"],
          ["PCA", "0.604", "0.964", "−2.062", "0.638"], ["TuckerFactorConcat", "0.601", "0.974", "−1.903", "0.737"],
          ["Concatenation", "0.688", "0.966", "−0.163", "0.713"]
        ]
      }
    ]
  },

  methods: [
    { name: "CoLES", family: "Contrastive learning", description: "Contrasts random slices from the same entity against slices from other entities using the original margin loss.", origin: "Event sequences / transactions" },
    { name: "Slice-InfoNCE", family: "Contrastive learning", description: "Uses the same slice-based views as CoLES, replacing the margin objective with InfoNCE.", origin: "Event sequences + vision/text contrastive learning" },
    { name: "Mask-InfoNCE", family: "Contrastive learning", description: "Creates two independently masked sequence views and aligns them through InfoNCE.", origin: "Masked contrastive learning" },
    { name: "MLM", family: "Masked modelling", description: "Reconstructs masked categorical and numerical event fields with a bidirectional encoder.", origin: "Text, RecSys, transactions and time series" },
    { name: "JEPA", family: "Latent prediction", description: "Predicts masked target representations from visible context using an EMA target encoder and a latent-space loss.", origin: "Vision, RecSys and language modelling" },
    { name: "NTP-GPT", family: "Next-token prediction", description: "A causal Transformer predicts the next event representation and exposes its hidden states as sequence embeddings.", origin: "Text, RecSys, transactions and time series" },
    { name: "NTP-GRU", family: "Next-token prediction", description: "An autoregressive GRU counterpart to NTP-GPT, trained with the same next-event objective.", origin: "Sequential autoregression" },
    { name: "TSFresh", family: "Statistical baseline", description: "A classical library of temporal, distributional and spectral sequence descriptors; no neural pretraining.", origin: "Time series and event sequences" },
    { name: "R-RNN", family: "Random baseline", description: "An untrained GRU encoder used to isolate the value added by self-supervised pretraining.", origin: "Random recurrent features" },
    { name: "R-Tr", family: "Random baseline", description: "An untrained GPT-style Transformer used as a strong random-feature reference.", origin: "Random Transformer features" }
  ],

  authors: [
    ["Egor Surkov", "Applied AI"], ["Andrey Savochkin", "Russian State Medical University"],
    ["Grigorii Gulii", "Applied AI"], ["Konstantin Sozykin", "Applied AI"],
    ["Dmitry Osin", "Applied AI"], ["Albert Bogdanov", "Applied AI"],
    ["Aleksei Shestov", "SB AI Lab"], ["Artem Sakhno", "SB AI Lab"],
    ["Maksim Makarenko", "SB AI Lab"], ["Andrey Savchenko", "SB AI Lab"],
    ["Evgeny Burnaev", "Applied AI"], ["Egor Shvetsov", "Applied AI"]
  ]
};
