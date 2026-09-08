const API_URL = "https://housescope-api.onrender.com";

const menuToggle = document.querySelector(".menu-toggle");
const navigation = document.getElementById("menu-navigation");
if (menuToggle && navigation) {
  menuToggle.addEventListener("click", () => {
    const expanded = menuToggle.getAttribute("aria-expanded") === "true";
    menuToggle.setAttribute("aria-expanded", String(!expanded));
    menuToggle.setAttribute("aria-label", expanded ? "Open navigation" : "Close navigation");
    navigation.classList.toggle("open", !expanded);
  });
  navigation.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      menuToggle.setAttribute("aria-expanded", "false");
      menuToggle.setAttribute("aria-label", "Open navigation");
      navigation.classList.remove("open");
    });
  });
}

async function readResponse(response) {
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || "The request could not be completed.");
  }
  return body;
}

const predictForm = document.getElementById("predictForm");
if (predictForm) {
predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button[type=submit]");
  const resultEmpty = document.getElementById("resultEmpty");
  const resultContent = document.getElementById("resultContent");
  const resultError = document.getElementById("resultError");
  const features = {};
  new FormData(event.target).forEach((value, key) => {
    features[key] = Number(value);
  });

  button.disabled = true;
  button.classList.add("loading");
  button.firstElementChild.textContent = "Calculating...";
  resultError.hidden = true;
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features)
    });
    const result = await readResponse(response);
    resultEmpty.hidden = true;
    resultContent.hidden = false;
    document.getElementById("predictedPrice").textContent = formatCurrency(result.predicted_price);
    document.getElementById("priceLow").textContent = formatCurrency(result.price_low);
    document.getElementById("priceHigh").textContent = formatCurrency(result.price_high);
    document.getElementById("rangeText").textContent =
      `${formatCurrency(result.price_low)} - ${formatCurrency(result.price_high)}`;
    const rangeSize = result.price_high - result.price_low;
    const markerPosition = rangeSize > 0
      ? ((result.predicted_price - result.price_low) / rangeSize) * 100
      : 50;
    document.getElementById("rangeMarker").style.left =
      `${Math.min(100, Math.max(0, markerPosition))}%`;
    const tierValue = document.getElementById("tierValue");
    tierValue.textContent = result.tier;
    tierValue.className = `tier-${result.tier_code.toLowerCase()}`;
    document.getElementById("segmentMedian").textContent = formatCurrency(result.tier_median_price);
    const difference = result.price_vs_tier_median_pct;
    const differenceElement = document.getElementById("segmentDifference");
    differenceElement.textContent = `${difference >= 0 ? "+" : ""}${difference}%`;
    differenceElement.className = difference >= 0 ? "value-positive" : "value-negative";
    document.getElementById("resultNote").textContent =
      `${result.assumptions[0].replace("were set to the training median or most common value.", "use typical training values.")}`;
  } catch (error) {
    resultError.textContent = error.message;
    resultError.hidden = false;
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
    button.firstElementChild.textContent = "Generate prediction";
  }
});
}

const templateButton = document.getElementById("templateButton");
if (templateButton) templateButton.addEventListener("click", () => {
  window.location.href = `${API_URL}/template`;
});
const exampleTemplateButton = document.getElementById("exampleTemplateButton");
if (exampleTemplateButton) exampleTemplateButton.addEventListener("click", () => {
  window.location.href = `${API_URL}/template?example=true`;
});

const csvFile = document.getElementById("csvFile");
if (csvFile) csvFile.addEventListener("change", (event) => {
  const label = event.target.closest(".file-label");
  if (event.target.files[0]) {
    label.firstChild.textContent = event.target.files[0].name;
  }
});

const uploadForm = document.getElementById("uploadForm");
if (uploadForm) uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const resultElement = document.getElementById("uploadResult");
  const file = document.getElementById("csvFile").files[0];
  const formData = new FormData();
  formData.append("file", file);

  try {
    event.target.classList.add("loading");
    const response = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData
    });

    const result = await readResponse(response);
    resultElement.className = "message success";
    const predictions = result.predictions;
    const values = predictions.map((item) => item.predicted_price);
    const average = values.reduce((sum, value) => sum + value, 0) / values.length;
    const preview = predictions.slice(0, 8);
    const batchOutput = document.getElementById("batchOutput");
    batchOutput.hidden = false;
    batchOutput.innerHTML = `
      <div class="batch-output-heading">
        <div>
          <span class="result-label">BATCH PREDICTIONS READY</span>
          <strong>${predictions.length.toLocaleString()} properties scored</strong>
        </div>
        <button id="downloadPredictions" class="download-button" type="button">Download all predictions <span aria-hidden="true">↓</span></button>
      </div>
      <div class="batch-metrics">
        <div><span>AVERAGE ESTIMATE</span><strong>${formatCurrency(average)}</strong></div>
        <div><span>LOWEST ESTIMATE</span><strong>${formatCurrency(Math.min(...values))}</strong></div>
        <div><span>HIGHEST ESTIMATE</span><strong>${formatCurrency(Math.max(...values))}</strong></div>
      </div>
      <p class="batch-preview-label">Preview of the first ${preview.length} rows</p>
      <div class="batch-table-wrap"><table class="batch-table">
        <thead><tr><th>Row</th><th>Predicted sale price</th></tr></thead>
        <tbody>${preview.map((item) => `<tr><td>${item.row + 1}</td><td>${formatCurrency(item.predicted_price)}</td></tr>`).join("")}</tbody>
      </table></div>`;
    document.getElementById("downloadPredictions").addEventListener("click", () => {
      const csv = `Row,PredictedSalePrice\n${predictions.map((item) => `${item.row + 1},${item.predicted_price}`).join("\n")}`;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
      link.download = "housescope-predictions.csv";
      link.click();
      URL.revokeObjectURL(link.href);
    });
    resultElement.textContent = "Predictions are ready below.";
  } catch (error) {
    resultElement.className = "message error";
    resultElement.textContent = `Error: ${error.message}`;
  } finally {
    event.target.classList.remove("loading");
  }
});

const contactForm = document.getElementById("contactForm");
if (contactForm) contactForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = document.getElementById("contactStatus");
  const button = contactForm.querySelector("button[type=submit]");
  button.disabled = true;
  button.textContent = "Sending...";
  status.textContent = "";
  try {
    const response = await fetch(contactForm.action, {
      method: "POST",
      body: new FormData(contactForm),
      headers: { Accept: "application/json" }
    });
    if (!response.ok) {
      throw new Error("Formspree rejected the message.");
    }
    contactForm.reset();
    status.textContent = "Message sent successfully. Thank you for reaching out.";
    status.className = "message success";
  } catch (error) {
    status.textContent = "Could not send the message. Please try again.";
    status.className = "message error";
  } finally {
    button.disabled = false;
    button.textContent = "Send message ->";
  }
});

function formatCurrency(value) {
  return `$${Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

if (document.getElementById("scatterPlot")) {
  loadSegmentation();
}

async function loadSegmentation() {
  try {
    const response = await fetch(`${API_URL}/segmentation`);
    const data = await readResponse(response);
    document.getElementById("clusterCount").textContent = data.n_clusters;
    document.getElementById("silhouetteScore").textContent = Number(data.silhouette_score).toFixed(3);
    document.getElementById("qualityBar").style.width = `${Math.max(0, Math.min(100, data.silhouette_score * 100))}%`;
    document.getElementById("featureCount").textContent = data.features.length;
    renderScatter(data.points, data.cluster_to_tier);
    renderDendrogram(data.dendrogram);
    renderSegmentCards(data.segment_stats);
  } catch (error) {
    document.getElementById("scatterPlot").textContent = `Unable to load segmentation: ${error.message}`;
  }
}

function renderDendrogram(dendrogram) {
  const target = document.getElementById("agglomerativePlot");
  if (!target || !dendrogram?.lines?.length) return;
  const paths = dendrogram.lines.map((points, index) => {
    const color = index % 3 === 0 ? "#7fa58a" : index % 3 === 1 ? "#4e8790" : "#d99a5b";
    return `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2" />`;
  }).join("");
  target.innerHTML = `<svg viewBox="0 0 ${dendrogram.width} ${dendrogram.height}" preserveAspectRatio="none" aria-hidden="true"><line x1="0" y1="${dendrogram.cut_y}" x2="${dendrogram.width}" y2="${dendrogram.cut_y}" stroke="#d99a5b" stroke-width="2" stroke-dasharray="7 5" /><text x="12" y="16">MERGE DISTANCE</text><text x="12" y="${dendrogram.cut_y - 7}">CUT FOR ${dendrogram.cut_clusters} GROUPS</text>${paths}</svg>`;
}

function renderScatter(points, clusterToTier) {
  const width = 900;
  const height = 390;
  const padding = 48;
  const maxX = Math.max(...points.map((point) => Number(point.lot_area)));
  const maxY = Math.max(...points.map((point) => Number(point.sale_price)));
  const colors = { Low: "#7fa58a", Medium: "#4e8790", High: "#d99a5b" };
  const circles = points.map((point) => {
    const x = padding + (Number(point.lot_area) / maxX) * (width - padding * 2);
    const y = height - padding - (Number(point.sale_price) / maxY) * (height - padding * 2);
    const tier = clusterToTier[String(point.cluster)];
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="${colors[tier] || "#8c6bb1"}" opacity=".65"><title>${tier}: ${formatCurrency(point.sale_price)}</title></circle>`;
  }).join("");
  document.getElementById("scatterPlot").innerHTML = `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true"><line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" /><line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" />${circles}<text x="${width / 2}" y="${height - 10}">LOT AREA</text><text x="16" y="${height / 2}" transform="rotate(-90 16 ${height / 2})">SALE PRICE</text></svg>`;
  document.getElementById("plotLegend").innerHTML = Object.entries(colors).map(([tier, color]) => `<span><i style="background:${color}"></i>${tier}</span>`).join("");
}

function renderSegmentCards(stats) {
  const descriptions = { Low: "Entry-level homes with smaller footprints and lower observed prices.", Medium: "Mid-market properties balancing space, quality, and age.", High: "Premium homes with larger living areas and stronger quality scores." };
  document.getElementById("segmentCards").innerHTML = Object.entries(stats).map(([tier, value]) => `<article class="segment-card tier-${tier.toLowerCase()}"><p class="eyebrow">${tier} TIER</p><strong>${formatCurrency(value.median_price)}</strong><p>${descriptions[tier]}</p><small>${value.count} properties / avg ${formatCurrency(value.mean_price)}</small></article>`).join("");
}

const segmentForm = document.getElementById("segmentForm");
if (segmentForm) {
  const resultElement = document.getElementById("segmentResult");
  segmentForm.addEventListener("input", () => {
    resultElement.textContent = "Ready to analyze the updated property.";
    resultElement.className = "segment-result";
  });
  segmentForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const values = {};
    new FormData(event.target).forEach((value, key) => { values[key] = Number(value); });
    resultElement.textContent = "Analyzing this property...";
    resultElement.className = "segment-result";
    try {
      const response = await fetch(`${API_URL}/segment`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values) });
      const result = await readResponse(response);
      resultElement.innerHTML = `
        <div class="segment-result-heading">Property classification</div>
        <table class="segment-result-table">
          <caption>Inputs analyzed and assigned market segment</caption>
          <tbody>
            <tr><th>Lot area</th><td>${values.LotArea.toLocaleString()} sq ft</td><th>Year built</th><td>${values.YearBuilt}</td></tr>
            <tr><th>Living area</th><td>${values.GrLivArea.toLocaleString()} sq ft</td><th>First floor</th><td>${values["1stFlrSF"].toLocaleString()} sq ft</td></tr>
            <tr><th>Overall quality</th><td>${values.OverallQual}/10</td><th>Cluster</th><td>${result.cluster + 1}</td></tr>
            <tr class="segment-result-tier"><th>Market tier</th><td colspan="3">${result.tier} <span>(${result.tier_code})</span></td></tr>
            <tr><th>Segment profile</th><td colspan="3">${result.insight}</td></tr>
          </tbody>
        </table>`;
      resultElement.className = `segment-result tier-${result.tier_code.toLowerCase()}`;
    } catch (error) {
      const message = error instanceof TypeError && error.message === "Failed to fetch"
        ? "Backend unavailable. Start FastAPI with `python -m uvicorn app:app --reload` from the backend folder, then refresh this page."
        : `Error: ${error.message}`;
      resultElement.textContent = message;
      resultElement.className = "segment-result error";
    }
  });
}
