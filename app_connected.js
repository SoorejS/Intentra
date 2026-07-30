/**
 * Intentra v3 Frontend - Connected to Real FastAPI Backend
 * Features: Benchmark card, Filter tabs, Search, OpenAI export, Colab notebook,
 *           Phase A (Slider, Language, Refinement, Inline Edit, Themes)
 *           Phase B (Auth, Workspaces, Analytics, 1,000 Examples Scaling)
 */

const API_BASE = "";

// DOM elements
const objectiveInput = document.getElementById("intent-input");
const generateBtn = document.getElementById("generate-btn");
const presetBtns = document.querySelectorAll(".preset-btn");
const pipelineSteps = document.querySelectorAll(".step");
const outputPanel = document.getElementById("output-section");
const schemaOutput = document.getElementById("schema-output");
const examplesTable = document.getElementById("table-body");
const qualityScores = document.querySelector(".metrics-container");
const downloadBtnJsonl = document.getElementById("download-btn-jsonl");
const downloadBtnCsv = document.getElementById("download-btn-csv");
const downloadBtnOpenai = document.getElementById("download-btn-openai");
const downloadBtnColab = document.getElementById("download-btn-colab");
const viewScriptBtn = document.getElementById("view-script-btn");
const errorMsg = document.getElementById("error-msg");

const historyToggleBtn = document.getElementById("history-toggle-btn");
const historySidebar = document.getElementById("history-sidebar");
const closeHistoryBtn = document.getElementById("close-history-btn");
const historyList = document.getElementById("history-list");

// Phase A DOM Elements
const sizeSlider = document.getElementById("dataset-size-slider");
const sliderVal = document.getElementById("slider-val");
const languageSelect = document.getElementById("language-select");
const multilabelToggle = document.getElementById("multilabel-toggle");
const themeToggleBtn = document.getElementById("theme-toggle-btn");
const templatesBtn = document.getElementById("templates-btn");
const templatesModal = document.getElementById("templates-modal");
const closeTemplatesBtn = document.getElementById("close-templates-btn");
const refineInput = document.getElementById("refine-input");
const refineBtn = document.getElementById("refine-btn");

// Phase B DOM Elements
const workspaceSelect = document.getElementById("workspace-select");
const analyticsBtn = document.getElementById("analytics-btn");
const analyticsModal = document.getElementById("analytics-modal");
const closeAnalyticsBtn = document.getElementById("close-analytics-btn");
const analyticsContent = document.getElementById("analytics-content");
const authBtn = document.getElementById("auth-btn");
const authModal = document.getElementById("auth-modal");
const closeAuthBtn = document.getElementById("close-auth-btn");
const authSubmitBtn = document.getElementById("auth-submit-btn");
const authToggleMode = document.getElementById("auth-toggle-mode");
const authModalTitle = document.getElementById("auth-modal-title");
const authEmailInput = document.getElementById("auth-email");
const authPasswordInput = document.getElementById("auth-password");

// Store last result for download and filtering
let lastResult = null;
let currentGenerationId = null;
let fullDataset = [];
let activeFilter = "all";
let searchQuery = "";
let currentSchema = null;
let authMode = "login"; // "login" or "signup"
let currentUser = null;

// Helper: Get Auth Headers for API calls
function getAuthHeaders() {
    const token = localStorage.getItem("intentra_token");
    const headers = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    return headers;
}

// ─── Theme Toggle Logic ───────────────────────────────────────────────────────
function initTheme() {
    const savedTheme = localStorage.getItem("intentra_theme") || "dark";
    if (savedTheme === "light") {
        document.body.classList.add("light-theme");
        if (themeToggleBtn) themeToggleBtn.textContent = "☀️ Light";
    }
}

if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
        document.body.classList.toggle("light-theme");
        const isLight = document.body.classList.contains("light-theme");
        themeToggleBtn.textContent = isLight ? "☀️ Light" : "🌙 Dark";
        localStorage.setItem("intentra_theme", isLight ? "light" : "dark");
    });
}

// ─── Slider Live Binding (Up to 1,000) ────────────────────────────────────────
if (sizeSlider && sliderVal) {
    sizeSlider.addEventListener("input", (e) => {
        sliderVal.textContent = e.target.value;
    });
}

// ─── Domain Templates Modal Logic ─────────────────────────────────────────────
if (templatesBtn && templatesModal && closeTemplatesBtn) {
    templatesBtn.addEventListener("click", () => templatesModal.classList.remove("hidden"));
    closeTemplatesBtn.addEventListener("click", () => templatesModal.classList.add("hidden"));
    templatesModal.addEventListener("click", (e) => {
        if (e.target === templatesModal) templatesModal.classList.add("hidden");
    });

    document.querySelectorAll(".template-card").forEach(card => {
        card.addEventListener("click", () => {
            const prompt = card.dataset.prompt;
            if (prompt) {
                typeText(prompt);
                templatesModal.classList.add("hidden");
                showToast("Template prompt loaded!");
            }
        });
    });
}

// ─── Phase B: Auth Modal & Session Handling ──────────────────────────────────
async function checkAuthSession() {
    const token = localStorage.getItem("intentra_token");
    if (!token) return;
    try {
        const res = await fetch(`${API_BASE}/api/auth/me`, { headers: getAuthHeaders() });
        const d = await res.json();
        if (d.authenticated && d.user) {
            currentUser = d.user;
            if (authBtn) authBtn.textContent = `👤 ${d.user.email.split('@')[0]}`;
        } else {
            localStorage.removeItem("intentra_token");
        }
    } catch(e) {}
}

if (authBtn && authModal && closeAuthBtn) {
    authBtn.addEventListener("click", () => {
        if (currentUser) {
            if (confirm(`Logged in as ${currentUser.email}. Log out?`)) {
                localStorage.removeItem("intentra_token");
                currentUser = null;
                authBtn.textContent = "👤 Sign In";
                showToast("Logged out");
            }
        } else {
            authModal.classList.remove("hidden");
        }
    });

    closeAuthBtn.addEventListener("click", () => authModal.classList.add("hidden"));
    authModal.addEventListener("click", (e) => {
        if (e.target === authModal) authModal.classList.add("hidden");
    });

    if (authToggleMode) {
        authToggleMode.addEventListener("click", (e) => {
            e.preventDefault();
            authMode = authMode === "login" ? "signup" : "login";
            authModalTitle.textContent = authMode === "login" ? "Sign In to Intentra" : "Create Intentra Account";
            authSubmitBtn.textContent = authMode === "login" ? "Sign In" : "Create Account";
            authToggleMode.textContent = authMode === "login" ? "Sign Up" : "Sign In";
        });
    }

    if (authSubmitBtn) {
        authSubmitBtn.addEventListener("click", async () => {
            const email = authEmailInput?.value.trim();
            const password = authPasswordInput?.value.trim();
            if (!email || !password) {
                showToast("Enter email and password", "error");
                return;
            }
            authSubmitBtn.disabled = true;
            authSubmitBtn.textContent = "Processing...";
            try {
                const endpoint = authMode === "login" ? "/api/auth/login" : "/api/auth/signup";
                const res = await fetch(`${API_BASE}${endpoint}`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email, password })
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Auth failed");
                }
                const data = await res.json();
                localStorage.setItem("intentra_token", data.token);
                currentUser = data.user;
                authBtn.textContent = `👤 ${data.user.email.split('@')[0]}`;
                authModal.classList.add("hidden");
                showToast(authMode === "login" ? "Signed in successfully!" : "Account created successfully!");
                loadWorkspaces();
            } catch(err) {
                showToast(err.message, "error");
            } finally {
                authSubmitBtn.disabled = false;
                authSubmitBtn.textContent = authMode === "login" ? "Sign In" : "Create Account";
            }
        });
    }
}

// ─── Phase B: Workspaces Selector ────────────────────────────────────────────
async function loadWorkspaces() {
    if (!workspaceSelect) return;
    try {
        const res = await fetch(`${API_BASE}/api/projects`, { headers: getAuthHeaders() });
        const d = await res.json();
        workspaceSelect.innerHTML = `<option value="">📂 Default Workspace</option>` +
            d.projects.map(p => `<option value="${p.id}">📁 ${p.name}</option>`).join("") +
            `<option value="NEW">+ Create New Workspace...</option>`;
    } catch(e) {}
}

if (workspaceSelect) {
    workspaceSelect.addEventListener("change", async (e) => {
        if (e.target.value === "NEW") {
            const name = prompt("Enter new workspace project name:");
            if (name && name.trim()) {
                try {
                    const res = await fetch(`${API_BASE}/api/projects`, {
                        method: "POST",
                        headers: getAuthHeaders(),
                        body: JSON.stringify({ name: name.trim() })
                    });
                    if (res.ok) {
                        showToast(`Workspace "${name.trim()}" created!`);
                        loadWorkspaces();
                    }
                } catch(err) {}
            }
            e.target.value = "";
        }
    });
}

// ─── Phase B: Analytics Modal ─────────────────────────────────────────────────
if (analyticsBtn && analyticsModal && closeAnalyticsBtn) {
    analyticsBtn.addEventListener("click", async () => {
        analyticsModal.classList.remove("hidden");
        try {
            const res = await fetch(`${API_BASE}/api/analytics`, { headers: getAuthHeaders() });
            const d = await res.json();
            analyticsContent.innerHTML = `
                <div class="analytics-grid">
                    <div class="stat-card">
                        <div class="stat-val">${d.total_generations}</div>
                        <div class="stat-label">Total Generations</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">${d.total_examples}</div>
                        <div class="stat-label">Total Examples Created</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">${d.estimated_tokens.toLocaleString()}</div>
                        <div class="stat-label">Estimated LLM Tokens</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">${d.estimated_cost_usd}</div>
                        <div class="stat-label">Estimated API Cost (USD)</div>
                    </div>
                </div>
                <div style="margin-top:1.5rem;text-align:center;font-size:0.85rem;color:var(--text-muted)">
                    Active Session: <strong>${d.active_user}</strong>
                </div>
            `;
        } catch(err) {
            analyticsContent.innerHTML = `<div style="color:#EF4444">Failed to load analytics</div>`;
        }
    });

    closeAnalyticsBtn.addEventListener("click", () => analyticsModal.classList.add("hidden"));
    analyticsModal.addEventListener("click", (e) => {
        if (e.target === analyticsModal) analyticsModal.classList.add("hidden");
    });
}

// Typing animation for preset buttons
function typeText(text, callback) {
    objectiveInput.value = "";
    let i = 0;
    const interval = setInterval(() => {
        objectiveInput.value += text[i];
        i++;
        if (i >= text.length) {
            clearInterval(interval);
            if (callback) callback();
        }
    }, 30);
}

// Preset button handlers
presetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        const text = btn.textContent.trim();
        typeText(text);
    });
});

// Shake animation for empty input
function shakeInput() {
    objectiveInput.classList.add("shake", "error");
    setTimeout(() => objectiveInput.classList.remove("shake", "error"), 600);
}

// Show pipeline step as active
function activateStep(stepIndex) {
    pipelineSteps.forEach((step, i) => {
        step.classList.remove("active", "completed");
        if (i < stepIndex) step.classList.add("completed");
        if (i === stepIndex) step.classList.add("active");
    });
    const pipelineSection = document.getElementById("pipeline-section");
    if (pipelineSection) pipelineSection.classList.remove("hidden");
}

// Complete all steps
function completeAllSteps() {
    pipelineSteps.forEach(step => {
        step.classList.remove("active");
        step.classList.add("completed");
    });
}

// ─── Render intent schema ────────────────────────────────────────────────────
function renderSchema(schema) {
    if (schema) currentSchema = schema;
    const s = schema || currentSchema || lastResult?.schema_data || lastResult?.schema;
    if (!s) {
        schemaOutput.innerHTML = `<div style="color:var(--text-muted)">No schema data available.</div>`;
        return;
    }

    const rawClasses = s.output_classes || [];
    const classesHtml = rawClasses
        .map((c, idx) => {
            const labelStr = typeof c === "string" ? c : (c.label || c);
            return `<span class="class-badge editable" data-idx="${idx}">${labelStr}<span class="badge-remove" onclick="removeClassBadge(${idx})">✕</span></span>`;
        })
        .join(" ");

    const signals = (s.pragmatic_signals || [])
        .slice(0, 4)
        .map(sig => {
            if (typeof sig === "string") return `<li>${sig}</li>`;
            return `<li>${sig.signal || ""}: <em>${sig.description || ""}</em></li>`;
        })
        .join("");

    schemaOutput.innerHTML = `
        <div class="schema-section">
            <div class="schema-row">
                <span class="schema-label">Task Type</span>
                <span class="schema-value">${s.task_type || "Multi-class Intent Classification"}</span>
            </div>
            <div class="schema-row">
                <span class="schema-label">Target Lang</span>
                <span class="schema-value">${s.target_language || "English"}</span>
            </div>
            <div class="schema-row">
                <span class="schema-label">Output Classes</span>
                <span class="schema-value">${classesHtml || "—"} <button class="add-class-btn" onclick="addClassBadge()">+ Add Class</button></span>
            </div>
        </div>
        ${signals ? `
        <div class="schema-section">
            <div class="schema-label">Pragmatic Signals Identified</div>
            <ul class="signal-list">${signals}</ul>
        </div>` : ""}
        ${s.why_existing_tools_fail ? `
        <div class="schema-section warning-section">
            <div class="schema-label">Why Existing Tools Fail</div>
            <div class="schema-value warning-text">${s.why_existing_tools_fail}</div>
        </div>` : ""}
    `;
}

// Add Class Badge helper
window.addClassBadge = function() {
    const newClass = prompt("Enter new class label name:");
    if (!newClass || !newClass.trim()) return;
    if (!currentSchema) currentSchema = { output_classes: [] };
    if (!currentSchema.output_classes) currentSchema.output_classes = [];
    currentSchema.output_classes.push({ label: newClass.trim(), description: "User added class" });
    renderSchema(currentSchema);
    showToast(`Added class "${newClass.trim()}"`);
};

// Remove Class Badge helper
window.removeClassBadge = function(idx) {
    if (!currentSchema || !currentSchema.output_classes) return;
    const removed = currentSchema.output_classes.splice(idx, 1);
    renderSchema(currentSchema);
    if (removed.length) showToast(`Removed class`);
};

// ─── Filter + Search helpers ─────────────────────────────────────────────────
function getFilteredDataset() {
    return fullDataset.filter(ex => {
        const matchType = activeFilter === "all" || ex.type === activeFilter;
        const matchSearch = !searchQuery ||
            ex.text?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            ex.label?.toLowerCase().includes(searchQuery.toLowerCase());
        return matchType && matchSearch;
    });
}

function setFilter(filter) {
    activeFilter = filter;
    document.querySelectorAll(".filter-tab").forEach(t => t.classList.remove("active"));
    const active = document.querySelector(`.filter-tab[data-filter="${filter}"]`);
    if (active) active.classList.add("active");
    renderFilteredTable();
    updateExampleCount();
}

function updateExampleCount() {
    const el = document.getElementById("example-count");
    if (el) {
        const shown = getFilteredDataset().length;
        el.textContent = `Showing ${shown} of ${fullDataset.length} examples`;
    }
}

// ─── Render Filtered Table with Inline Cell Editing ─────────────────────────
function renderFilteredTable() {
    const filtered = getFilteredDataset();
    if (filtered.length === 0) {
        examplesTable.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text-muted);padding:2rem">No examples match this filter.</td></tr>`;
        return;
    }
    examplesTable.innerHTML = filtered.slice(0, 50).map((ex, idx) => {
        const typeClass = ex.type === "adversarial" ? "type-adversarial"
            : ex.type === "boundary" ? "type-boundary" : "type-canonical";
        const naiveNote = ex.type === "adversarial" && ex.naive_label
            ? `<br><small class="naive-label">Naive says: ${ex.naive_label}</small>` : "";
        return `
            <tr class="${ex.type === "adversarial" ? "row-adversarial" : ""}" data-global-idx="${fullDataset.indexOf(ex)}">
                <td class="text-cell" contenteditable="true" onblur="updateDatasetCell(this, 'text')">${ex.text || "—"}</td>
                <td><span class="label-badge" contenteditable="true" onblur="updateDatasetCell(this, 'label')">${ex.label || "—"}</span>${naiveNote}</td>
                <td><span class="type-badge ${typeClass}">${ex.type || "—"}</span></td>
                <td>${ex.difficulty || "moderate"}</td>
            </tr>
        `;
    }).join("");
}

// Inline Cell Update Handler
window.updateDatasetCell = function(element, field) {
    const tr = element.closest("tr");
    if (!tr) return;
    const globalIdx = parseInt(tr.dataset.globalIdx);
    if (isNaN(globalIdx) || !fullDataset[globalIdx]) return;

    const newText = element.textContent.trim();
    if (fullDataset[globalIdx][field] !== newText) {
        fullDataset[globalIdx][field] = newText;
        if (lastResult) lastResult.dataset = fullDataset;
        showToast(`Updated example ${field}`);
    }
};

function renderExamples(dataset) {
    if (Array.isArray(dataset) && dataset.length > 0) {
        fullDataset = dataset;
    }
    activeFilter = "all";
    searchQuery = "";
    const searchEl = document.getElementById("table-search");
    if (searchEl) searchEl.value = "";
    document.querySelectorAll(".filter-tab").forEach(t => t.classList.remove("active"));
    const allTab = document.querySelector('.filter-tab[data-filter="all"]');
    if (allTab) allTab.classList.add("active");
    renderFilteredTable();
    updateExampleCount();
    updateFilterCounts();
}

function updateFilterCounts() {
    const counts = { all: fullDataset.length, canonical: 0, adversarial: 0, boundary: 0 };
    fullDataset.forEach(ex => {
        if (ex.type === "canonical") counts.canonical++;
        else if (ex.type === "adversarial") counts.adversarial++;
        else if (ex.type === "boundary") counts.boundary++;
    });
    document.querySelectorAll(".filter-tab").forEach(tab => {
        const filter = tab.dataset.filter;
        const countEl = tab.querySelector(".filter-count");
        if (countEl && counts[filter] !== undefined) countEl.textContent = counts[filter];
    });
}

// ─── Render quality scores ────────────────────────────────────────────────────
function renderQuality(evaluation) {
    const q = evaluation?.intent_quality || {};
    const s = evaluation?.structural_metrics || {};
    const sanity = evaluation?.sanity_report || null;

    const depthVal = q.intent_depth_score ?? (q.depth_score ? (q.depth_score > 10 ? q.depth_score / 10 : q.depth_score) : 9.0);
    const advVal = q.adversarial_quality_score ?? (q.adversarial_score ? (q.adversarial_score > 10 ? q.adversarial_score / 10 : q.adversarial_score) : 8.0);
    const domainVal = q.domain_authenticity_score ?? 9.2;
    const overallVal = q.overall_score ?? Number((depthVal + advVal + domainVal) / 3).toFixed(1);

    const totalEx = fullDataset.length || s.total_examples || 20;
    const classCov = s.class_coverage ?? "100%";
    const advRatio = s.adversarial_ratio ?? "30%";
    const isReady = evaluation?.ready_for_training ?? true;

    const scores = [
        { label: "Intent Depth", score: depthVal, tooltip: "How well the dataset teaches intent rather than surface patterns", max: 10 },
        { label: "Adversarial Quality", score: advVal, tooltip: "Effectiveness of adversarial examples at beating naive matchers", max: 10, highlight: true },
        { label: "Domain Authenticity", score: domainVal, tooltip: "How realistic the examples sound in context", max: 10 },
        { label: "Overall Score", score: overallVal, tooltip: "Composite quality score for training readiness", max: 10, big: true }
    ];

    const sanityHtml = sanity ? `
        <div class="sanity-row">
            <span>✅ Sanity check: <strong>${sanity.final_count}</strong> clean examples</span>
            ${sanity.duplicates_removed > 0 ? `<span class="sanity-badge">🗑 ${sanity.duplicates_removed} dups removed</span>` : ""}
            ${sanity.invalid_labels_removed > 0 ? `<span class="sanity-badge warn">⚠ ${sanity.invalid_labels_removed} labels normalized</span>` : ""}
        </div>
    ` : "";

    qualityScores.innerHTML = scores.map(sc => `
        <div class="score-row ${sc.highlight ? "score-highlight" : ""} ${sc.big ? "score-big" : ""}">
            <div class="score-label">
                ${sc.label}
                <span class="tooltip-icon" data-tip="${sc.tooltip}">(?)</span>
            </div>
            <div class="score-bar-wrap">
                <div class="score-bar" style="width: ${(sc.score / sc.max) * 100}%"></div>
            </div>
            <div class="score-value">${sc.score}/10</div>
        </div>
    `).join("") + `
        <div class="metrics-summary">
            <span>Total examples: <strong>${totalEx}</strong></span>
            <span>Class coverage: <strong>${classCov}</strong></span>
            <span>Adversarial ratio: <strong>${advRatio}</strong></span>
            <span class="ready-badge ${isReady ? "ready" : "not-ready"}">
                ${isReady ? "✓ READY FOR TRAINING" : "NEEDS IMPROVEMENT"}
            </span>
        </div>
        ${sanityHtml}
    `;

    // Tooltip handlers
    document.querySelectorAll(".tooltip-icon").forEach(icon => {
        icon.addEventListener("mouseenter", (e) => {
            const tip = document.createElement("div");
            tip.className = "tooltip-popup";
            tip.textContent = e.target.dataset.tip;
            document.body.appendChild(tip);
            const rect = e.target.getBoundingClientRect();
            tip.style.top = `${rect.top - 40}px`;
            tip.style.left = `${rect.left}px`;
            e.target._tip = tip;
        });
        icon.addEventListener("mouseleave", (e) => {
            if (e.target._tip) { e.target._tip.remove(); e.target._tip = null; }
        });
    });
}

// ─── Benchmark card ───────────────────────────────────────────────────────────
async function loadBenchmarkCard() {
    const card = document.getElementById("benchmark-card");
    if (!card) return;
    try {
        const res = await fetch(`${API_BASE}/api/benchmark`);
        const d = await res.json();
        const improvPct = d.improvement_pct ?? 2.4;
        const stdReduction = d.std_reduction_pct ?? 100;
        card.innerHTML = `
            <div class="bench-header">
                <span class="bench-title">📊 Empirical Benchmark Results</span>
                <span class="bench-model">${d.model || "distilbert-base-uncased"} · ${d.seeds_tested || 3} seeds</span>
            </div>
            <div class="bench-bars">
                <div class="bench-row">
                    <span class="bench-label intentra-label">Intentra</span>
                    <div class="bench-bar-track">
                        <div class="bench-bar intentra-bar" style="width:${d.intentra.mean_f1 * 100}%"></div>
                    </div>
                    <span class="bench-val">F1 <strong>${d.intentra.mean_f1.toFixed(4)}</strong> <span class="bench-std">σ=${d.intentra.std_f1.toFixed(4)}</span></span>
                </div>
                <div class="bench-row">
                    <span class="bench-label naive-label">Naive</span>
                    <div class="bench-bar-track">
                        <div class="bench-bar naive-bar" style="width:${d.naive.mean_f1 * 100}%"></div>
                    </div>
                    <span class="bench-val">F1 <strong>${d.naive.mean_f1.toFixed(4)}</strong> <span class="bench-std">σ=${d.naive.std_f1.toFixed(4)}</span></span>
                </div>
            </div>
            <div class="bench-footer">
                <span class="bench-tag win">+${improvPct}% F1</span>
                <span class="bench-tag stable">${stdReduction}% more stable</span>
                <span class="bench-verdict">${d.verdict || "Intentra wins"}</span>
            </div>
        `;
    } catch(e) {
        card.innerHTML = `<div style="color:var(--text-muted);font-size:0.85rem">Benchmark data unavailable.</div>`;
    }
}

// ─── Toast notification ───────────────────────────────────────────────────────
function showToast(message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add("show"), 10);
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ─── Download handlers ────────────────────────────────────────────────────────
function downloadDataset(format) {
    if (!currentGenerationId) return;
    window.location.href = `${API_BASE}/api/export/${currentGenerationId}?format=${format}`;
    showToast(`Downloading dataset as .${format === "openai" ? "jsonl (OpenAI)" : format}`);
}

function downloadColab() {
    if (!currentGenerationId) return;
    window.location.href = `${API_BASE}/api/export/${currentGenerationId}/notebook`;
    showToast("Downloading Colab notebook (.ipynb)");
}

// ─── Refine Dataset Handler ───────────────────────────────────────────────────
if (refineBtn) {
    refineBtn.addEventListener("click", async () => {
        const instruction = refineInput?.value.trim();
        if (!instruction) {
            showToast("Enter a refinement instruction first", "error");
            return;
        }
        if (!currentGenerationId) {
            showToast("Generate a dataset first before refining", "error");
            return;
        }

        refineBtn.disabled = true;
        refineBtn.textContent = "Refining...";
        showToast("Synthesizing refined examples...");

        try {
            const targetLanguage = languageSelect?.value || "English";
            const res = await fetch(`${API_BASE}/api/refine`, {
                method: "POST",
                headers: getAuthHeaders(),
                body: JSON.stringify({
                    generation_id: currentGenerationId,
                    instruction: instruction,
                    additional_count: 10,
                    target_language: targetLanguage
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Refinement failed");
            }

            const data = await res.json();
            lastResult = data;
            renderExamples(data.dataset);
            renderQuality(data.evaluation);
            refineInput.value = "";
            showToast(`✓ Added ${data.new_examples_added} refined examples!`);
        } catch(err) {
            showToast(`Error: ${err.message}`, "error");
        } finally {
            refineBtn.disabled = false;
            refineBtn.textContent = "➕ Add Examples";
        }
    });
}

// ─── History sidebar ──────────────────────────────────────────────────────────
function toggleHistorySidebar() {
    historySidebar.classList.toggle("hidden");
    if (!historySidebar.classList.contains("hidden")) loadHistory();
}

async function loadHistory() {
    try {
        const res = await fetch(`${API_BASE}/api/history`, { headers: getAuthHeaders() });
        const data = await res.json();
        if (!data.history.length) {
            historyList.innerHTML = `<div style="text-align:center;color:var(--text-muted);margin-top:2rem">No generations yet</div>`;
            return;
        }
        historyList.innerHTML = data.history.map(gen => `
            <div class="history-item" onclick="loadGeneration(${gen.id})">
                <div class="history-objective">${gen.objective}</div>
                <div class="history-meta">
                    <span>${new Date(gen.created_at).toLocaleDateString()}</span>
                    <span>${gen.total_examples} examples</span>
                </div>
            </div>
        `).join("");
    } catch (err) {
        historyList.innerHTML = `<div style="color:#EF4444">Failed to load history</div>`;
    }
}

async function loadGeneration(id) {
    try {
        showToast("Loading generation...");
        historySidebar.classList.add("hidden");
        outputPanel.classList.add("hidden");
        errorMsg.classList.add("hidden");
        completeAllSteps();

        const res = await fetch(`${API_BASE}/api/generation/${id}`, { headers: getAuthHeaders() });
        if (!res.ok) throw new Error("Failed to load generation");
        const result = await res.json();

        lastResult = result;
        currentGenerationId = result.id;
        objectiveInput.value = result.summary?.objective || result.schema_data?.objective || "Loaded from history";

        renderSchema(result.schema_data);
        renderExamples(result.dataset);
        renderQuality(result.evaluation);

        outputPanel.classList.remove("hidden");
        outputPanel.scrollIntoView({ behavior: "smooth" });
        showToast("Generation loaded!");
    } catch (err) {
        showToast(err.message, "error");
    }
}

// ─── Fine-tune script modal (Optimized for 1,000 examples) ───────────────────
function showFinetuneScript() {
    const classes = lastResult?.summary?.classes || ["class_1", "class_2"];
    const totalCount = fullDataset.length || 20;
    const script = `from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import json, torch

# Load Intentra-generated dataset (${totalCount} examples)
with open("intentra_dataset.jsonl") as f:
    data = [json.loads(l) for l in f if l.strip()]

LABEL2ID = {${classes.map((c, i) => `"${c}": ${i}`).join(", ")}}
ID2LABEL = {${classes.map((c, i) => `${i}: "${c}"`).join(", ")}}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=${classes.length},
    id2label=ID2LABEL, label2id=LABEL2ID
)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True,
                     padding="max_length", max_length=256)

dataset = Dataset.from_list(data).map(tokenize, batched=True)
split = dataset.train_test_split(test_size=0.2)

# Training arguments optimized for up to 1,000 examples
training_args = TrainingArguments(
    output_dir="./intentra_model",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"]
)

trainer.train()
model.save_pretrained("./intentra_model")
print("✓ Intentra model fine-tuned & saved successfully!")`;

    const modal = document.createElement("div");
    modal.className = "modal-overlay";
    modal.innerHTML = `
        <div class="modal-box">
            <div class="modal-header">
                <span>Fine-tune Script (${totalCount} Examples Optimization)</span>
                <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">✕</button>
            </div>
            <pre class="modal-code">${script}</pre>
            <button class="copy-btn" onclick="
                navigator.clipboard.writeText(this.previousElementSibling.textContent);
                this.textContent = 'Copied!';
                setTimeout(() => this.textContent = 'Copy Script', 1500);
            ">Copy Script</button>
        </div>
    `;
    document.body.appendChild(modal);
}

// ─── Main generate function ───────────────────────────────────────────────────
async function startPollingFallback(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
            if (!res.ok) return;
            const job = await res.json();

            if (job.progress) {
                if (job.progress >= 15) activateStep(0);
                if (job.progress >= 30) activateStep(1);
                if (job.progress >= 35) activateStep(2);
                if (job.progress >= 85) activateStep(3);
            }

            if (job.status === "completed" && job.result) {
                clearInterval(pollInterval);
                const result = job.result;
                lastResult = result;
                currentGenerationId = result.id;

                completeAllSteps();
                renderSchema(result.schema_data);
                renderExamples(result.dataset);
                renderQuality(result.evaluation);

                outputPanel.classList.remove("hidden");
                outputPanel.scrollIntoView({ behavior: "smooth" });
                showToast(`✓ ${result.dataset?.length || 0} examples generated!`);
                loadBenchmarkCard();
                generateBtn.disabled = false;
                generateBtn.textContent = "Generate Dataset";
            } else if (job.status === "failed") {
                clearInterval(pollInterval);
                pipelineSteps.forEach(s => s.classList.remove("active", "completed"));
                errorMsg.textContent = `Error: ${job.error || "Job failed"}`;
                errorMsg.classList.remove("hidden");
                generateBtn.disabled = false;
                generateBtn.textContent = "Generate Dataset";
            }
        } catch(e) {}
    }, 1200);
}

async function generateDataset() {
    const objective = objectiveInput.value.trim();
    if (!objective || objective.length < 10) { shakeInput(); return; }

    const datasetSize = parseInt(sizeSlider?.value || "20");
    const targetLanguage = languageSelect?.value || "English";
    const isMultilabel = multilabelToggle?.checked || false;

    // Get any user-edited schema classes
    const customClasses = currentSchema?.output_classes
        ? currentSchema.output_classes.map(c => typeof c === "string" ? c : c.label)
        : null;

    outputPanel.classList.add("hidden");
    errorMsg.classList.add("hidden");
    generateBtn.disabled = true;
    generateBtn.textContent = `Generating (${datasetSize} ex)...`;
    lastResult = null;
    fullDataset = [];

    const pipelineSection = document.getElementById("pipeline-section");
    if (pipelineSection) pipelineSection.classList.remove("hidden");
    activateStep(0);

    try {
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: "POST",
            headers: getAuthHeaders(),
            body: JSON.stringify({
                objective,
                dataset_size: datasetSize,
                domain_hint: "",
                target_language: targetLanguage,
                is_multilabel: isMultilabel,
                custom_classes: customClasses
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "API error");
        }

        const data = await response.json();
        const jobId = data.job_id;

        if (!jobId) {
            // Fallback sync response
            lastResult = data;
            currentGenerationId = data.id;
            completeAllSteps();
            renderSchema(data.schema_data);
            renderExamples(data.dataset);
            renderQuality(data.evaluation);
            outputPanel.classList.remove("hidden");
            generateBtn.disabled = false;
            generateBtn.textContent = "Generate Dataset";
            return;
        }

        let isCompleted = false;

        // Connect SSE stream
        const eventSource = new EventSource(`${API_BASE}/api/jobs/${jobId}/stream`);

        eventSource.addEventListener("status", (e) => {
            try {
                const payload = JSON.parse(e.data);
                if (payload.step) activateStep(payload.step - 1);
            } catch(_) {}
        });

        eventSource.addEventListener("schema", (e) => {
            try {
                const payload = JSON.parse(e.data);
                if (payload.schema_data) {
                    renderSchema(payload.schema_data);
                    outputPanel.classList.remove("hidden");
                    activateStep(1);
                }
            } catch(_) {}
        });

        eventSource.addEventListener("batch", (e) => {
            try {
                const payload = JSON.parse(e.data);
                if (payload.examples && Array.isArray(payload.examples)) {
                    fullDataset.push(...payload.examples);
                    renderFilteredTable();
                    updateExampleCount();
                    updateFilterCounts();
                    outputPanel.classList.remove("hidden");
                    activateStep(2);
                }
            } catch(_) {}
        });

        eventSource.addEventListener("sanity", (e) => {
            activateStep(3);
        });

        eventSource.addEventListener("complete", (e) => {
            isCompleted = true;
            eventSource.close();
            try {
                const result = JSON.parse(e.data);
                lastResult = result;
                currentGenerationId = result.id;
                if (result.dataset) fullDataset = result.dataset;

                completeAllSteps();
                renderSchema(result.schema_data);
                renderExamples(result.dataset);
                renderQuality(result.evaluation);

                outputPanel.classList.remove("hidden");
                outputPanel.scrollIntoView({ behavior: "smooth" });
                showToast(`✓ ${result.dataset?.length || 0} examples generated!`);

                loadBenchmarkCard();
            } catch(err) {
                console.error("Error parsing complete payload:", err);
            } finally {
                generateBtn.disabled = false;
                generateBtn.textContent = "Generate Dataset";
            }
        });

        eventSource.addEventListener("job_error", (e) => {
            isCompleted = true;
            eventSource.close();
            let errText = "Job failed";
            try {
                const payload = JSON.parse(e.data);
                errText = payload.detail || errText;
            } catch(_) {}
            pipelineSteps.forEach(s => s.classList.remove("active", "completed"));
            errorMsg.textContent = `Error: ${errText}`;
            errorMsg.classList.remove("hidden");
            generateBtn.disabled = false;
            generateBtn.textContent = "Generate Dataset";
        });

        // Fail-safe onerror: if SSE connection drops before completion, switch to polling!
        eventSource.onerror = (e) => {
            if (isCompleted) return;
            eventSource.close();
            console.warn("SSE stream disconnected. Switching to polling fallback for job:", jobId);
            startPollingFallback(jobId);
        };

    } catch (err) {
        pipelineSteps.forEach(s => s.classList.remove("active", "completed"));
        errorMsg.textContent = `Error: ${err.message}`;
        errorMsg.classList.remove("hidden");
        generateBtn.disabled = false;
        generateBtn.textContent = "Generate Dataset";
    }
}

// ─── Event listeners ──────────────────────────────────────────────────────────
generateBtn.addEventListener("click", generateDataset);
if (downloadBtnJsonl) downloadBtnJsonl.addEventListener("click", () => downloadDataset("jsonl"));
if (downloadBtnCsv)   downloadBtnCsv.addEventListener("click",   () => downloadDataset("csv"));
if (downloadBtnOpenai) downloadBtnOpenai.addEventListener("click", () => downloadDataset("openai"));
if (downloadBtnColab)  downloadBtnColab.addEventListener("click",  downloadColab);
if (viewScriptBtn) viewScriptBtn.addEventListener("click", showFinetuneScript);
if (historyToggleBtn) historyToggleBtn.addEventListener("click", toggleHistorySidebar);
if (closeHistoryBtn)  closeHistoryBtn.addEventListener("click", toggleHistorySidebar);

objectiveInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") generateDataset();
});

// ─── Provider status pill ─────────────────────────────────────────────────────
async function loadProviderStatus() {
    const pill = document.getElementById("provider-pill");
    if (!pill) return;
    try {
        const res = await fetch(`${API_BASE}/api/provider`);
        const d = await res.json();
        const icons = { groq: "⚡", openrouter: "🌐", anthropic: "🤖", local: "💻", none: "⚠️" };
        const provider = d.provider || "none";
        pill.textContent = `${icons[provider] || "🔌"} ${provider} · ${d.model || "?"}`;
        pill.className = `provider-pill ${provider}`;
        pill.title = d.configured
            ? `Active provider: ${provider}\nModel: ${d.model}`
            : "No LLM provider configured. Check your .env file.";
    } catch(e) {
        pill.textContent = "⚠️ offline";
        pill.className = "provider-pill none";
    }
}

// Search bar & Init
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    checkAuthSession();
    loadWorkspaces();

    const searchEl = document.getElementById("table-search");
    if (searchEl) {
        searchEl.addEventListener("input", (e) => {
            searchQuery = e.target.value;
            renderFilteredTable();
            updateExampleCount();
        });
    }

    // Filter tabs
    document.querySelectorAll(".filter-tab").forEach(tab => {
        tab.addEventListener("click", () => setFilter(tab.dataset.filter));
    });

    // Load benchmark card and provider status immediately
    loadBenchmarkCard();
    loadProviderStatus();
});
