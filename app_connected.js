/**
 * Intentra Frontend - Connected to Real FastAPI Backend
 * Replace mock data with live API calls to localhost:8000
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
const viewScriptBtn = document.getElementById("view-script-btn");
const errorMsg = document.getElementById("error-msg");

const historyToggleBtn = document.getElementById("history-toggle-btn");
const historySidebar = document.getElementById("history-sidebar");
const closeHistoryBtn = document.getElementById("close-history-btn");
const historyList = document.getElementById("history-list");

// Store last result for download
let lastResult = null;
let currentGenerationId = null;

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
    }, 35);
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
    setTimeout(() => {
        objectiveInput.classList.remove("shake", "error");
    }, 600);
}

// Show pipeline step as active
function activateStep(stepIndex) {
    pipelineSteps.forEach((step, i) => {
        step.classList.remove("active", "completed");
        if (i < stepIndex) step.classList.add("completed");
        if (i === stepIndex) step.classList.add("active");
    });
}

// Complete all steps
function completeAllSteps() {
    pipelineSteps.forEach(step => {
        step.classList.remove("active");
        step.classList.add("completed");
    });
}

// Render intent schema
function renderSchema(schema) {
    const classes = schema.output_classes
        .map(c => `<span class="class-badge">${c.label}</span>`)
        .join(" ");

    const signals = (schema.pragmatic_signals || [])
        .slice(0, 4)
        .map(s => `<li>${s.signal}: <em>${s.description}</em></li>`)
        .join("");

    const mechanisms = (schema.rhetorical_mechanisms || [])
        .slice(0, 3)
        .map(m => `<li>${m.mechanism}</li>`)
        .join("");

    schemaOutput.innerHTML = `
        <div class="schema-section">
            <div class="schema-row">
                <span class="schema-label">Task Type</span>
                <span class="schema-value">${schema.task_type}</span>
            </div>
            <div class="schema-row">
                <span class="schema-label">Deep Task</span>
                <span class="schema-value">${schema.deep_task || schema.intent_description || ""}</span>
            </div>
            <div class="schema-row">
                <span class="schema-label">Output Classes</span>
                <span class="schema-value">${classes}</span>
            </div>
        </div>
        ${signals ? `
        <div class="schema-section">
            <div class="schema-label">Pragmatic Signals Identified</div>
            <ul class="signal-list">${signals}</ul>
        </div>` : ""}
        ${mechanisms ? `
        <div class="schema-section">
            <div class="schema-label">Rhetorical Mechanisms Modeled</div>
            <ul class="signal-list">${mechanisms}</ul>
        </div>` : ""}
        ${schema.why_existing_tools_fail ? `
        <div class="schema-section warning-section">
            <div class="schema-label">Why Existing Tools Fail</div>
            <div class="schema-value warning-text">${schema.why_existing_tools_fail}</div>
        </div>` : ""}
    `;
}

// Render examples table
function renderExamples(dataset) {
    const rows = dataset.slice(0, 12).map(ex => {
        const typeClass = ex.type === "adversarial"
            ? "type-adversarial"
            : ex.type === "boundary"
                ? "type-boundary"
                : "type-canonical";

        const adversarialNote = ex.type === "adversarial" && ex.naive_label
            ? `<br><small class="naive-label">Naive model says: ${ex.naive_label}</small>`
            : "";

        return `
            <tr class="${ex.type === "adversarial" ? "row-adversarial" : ""}">
                <td class="text-cell">${ex.text}</td>
                <td><span class="label-badge">${ex.label}</span>${adversarialNote}</td>
                <td><span class="type-badge ${typeClass}">${ex.type}</span></td>
                <td>${ex.difficulty || "moderate"}</td>
            </tr>
        `;
    }).join("");

    examplesTable.innerHTML = rows;
}

// Render quality scores
function renderQuality(evaluation) {
    const q = evaluation.intent_quality;
    const s = evaluation.structural_metrics;

    const scores = [
        {
            label: "Intent Depth",
            score: q.intent_depth_score,
            tooltip: "How well the dataset teaches intent rather than surface patterns",
            max: 10
        },
        {
            label: "Adversarial Quality",
            score: q.adversarial_quality_score,
            tooltip: "How effectively adversarial examples fool naive pattern matchers",
            max: 10,
            highlight: true
        },
        {
            label: "Domain Authenticity",
            score: q.domain_authenticity_score,
            tooltip: "How realistic the examples sound in their domain",
            max: 10
        },
        {
            label: "Overall Score",
            score: q.overall_score,
            tooltip: "Composite quality score for training readiness",
            max: 10,
            big: true
        }
    ];

    qualityScores.innerHTML = scores.map(s => `
        <div class="score-row ${s.highlight ? "score-highlight" : ""} ${s.big ? "score-big" : ""}">
            <div class="score-label">
                ${s.label}
                <span class="tooltip-icon" data-tip="${s.tooltip}">(?)</span>
            </div>
            <div class="score-bar-wrap">
                <div class="score-bar" style="width: ${(s.score / s.max) * 100}%"></div>
            </div>
            <div class="score-value">${s.score}/10</div>
        </div>
    `).join("") + `
        <div class="metrics-summary">
            <span>Total examples: <strong>${evaluation.structural_metrics.total_examples}</strong></span>
            <span>Class coverage: <strong>${evaluation.structural_metrics.class_coverage}</strong></span>
            <span>Adversarial ratio: <strong>${evaluation.structural_metrics.adversarial_ratio}</strong></span>
            <span class="ready-badge ${evaluation.ready_for_training ? "ready" : "not-ready"}">
                ${evaluation.ready_for_training ? "READY FOR TRAINING" : "NEEDS IMPROVEMENT"}
            </span>
        </div>
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
            if (e.target._tip) {
                e.target._tip.remove();
                e.target._tip = null;
            }
        });
    });
}

// Show toast notification
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

// Download dataset as JSON
function downloadDatasetJsonl() {
    if (!currentGenerationId) return;
    window.location.href = `${API_BASE}/api/export/${currentGenerationId}?format=jsonl`;
    showToast("Downloading dataset as .jsonl");
}

function downloadDatasetCsv() {
    if (!currentGenerationId) return;
    window.location.href = `${API_BASE}/api/export/${currentGenerationId}?format=csv`;
    showToast("Downloading dataset as .csv");
}

// History sidebar toggle
function toggleHistorySidebar() {
    historySidebar.classList.toggle("hidden");
    if (!historySidebar.classList.contains("hidden")) {
        loadHistory();
    }
}

// Load history items
async function loadHistory() {
    try {
        const res = await fetch(`${API_BASE}/api/history`);
        const data = await res.json();
        
        if (data.history.length === 0) {
            historyList.innerHTML = `<div style="text-align:center;color:var(--text-muted);margin-top:2rem;">No generations yet</div>`;
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
        historyList.innerHTML = `<div style="color:#EF4444;">Failed to load history</div>`;
    }
}

// Load a specific generation
async function loadGeneration(id) {
    try {
        showToast("Loading generation...");
        historySidebar.classList.add("hidden");
        
        // Reset UI
        outputPanel.classList.add("hidden");
        errorMsg.classList.add("hidden");
        completeAllSteps();
        
        const res = await fetch(`${API_BASE}/api/generation/${id}`);
        if (!res.ok) throw new Error("Failed to load generation");
        
        const result = await res.json();
        lastResult = result;
        currentGenerationId = result.id;
        objectiveInput.value = result.summary.objective || "Loaded from history";
        
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

// Show fine-tune script modal
function showFinetuneScript() {
    const classes = lastResult
        ? lastResult.summary.classes
        : ["urgent", "moderate", "low"];

    const script = `from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer
)
from datasets import Dataset
import json

# Load Intentra-generated dataset
with open("intentra_dataset.json") as f:
    data = json.load(f)

LABEL2ID = {${classes.map((c, i) => `"${c}": ${i}`).join(", ")}}
ID2LABEL = {${classes.map((c, i) => `${i}: "${c}"`).join(", ")}}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=${classes.length},
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, 
                     padding="max_length", max_length=256)

dataset = Dataset.from_list(data).map(tokenize, batched=True)
split = dataset.train_test_split(test_size=0.2)

training_args = TrainingArguments(
    output_dir="./intentra_model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
)

trainer.train()
model.save_pretrained("./intentra_model")
print("Model ready for deployment!")`;

    const modal = document.createElement("div");
    modal.className = "modal-overlay";
    modal.innerHTML = `
        <div class="modal-box">
            <div class="modal-header">
                <span>Fine-tune Script</span>
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

// Main generate function - calls real API
async function generateDataset() {
    const objective = objectiveInput.value.trim();

    if (!objective || objective.length < 10) {
        shakeInput();
        return;
    }

    // Reset UI
    outputPanel.classList.add("hidden");
    errorMsg.classList.add("hidden");
    generateBtn.disabled = true;
    generateBtn.textContent = "Generating...";
    lastResult = null;

    try {
        // Animate pipeline steps while API call runs
        activateStep(0);
        await new Promise(r => setTimeout(r, 800));
        activateStep(1);
        await new Promise(r => setTimeout(r, 600));
        activateStep(2);

        // Real API call
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                objective: objective,
                dataset_size: 20,
                domain_hint: ""
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "API error");
        }

        const result = await response.json();
        lastResult = result;
        currentGenerationId = result.id;

        activateStep(3);
        await new Promise(r => setTimeout(r, 500));
        activateStep(4);
        await new Promise(r => setTimeout(r, 400));
        completeAllSteps();

        // Render outputs
        renderSchema(result.schema_data);
        renderExamples(result.dataset);
        renderQuality(result.evaluation);

        outputPanel.classList.remove("hidden");
        outputPanel.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
        pipelineSteps.forEach(s => s.classList.remove("active", "completed"));
        errorMsg.textContent = `Error: ${err.message}. Make sure the backend is running: uvicorn main:app --reload`;
        errorMsg.classList.remove("hidden");
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = "Generate Dataset";
    }
}

// Event listeners
generateBtn.addEventListener("click", generateDataset);
downloadBtnJsonl.addEventListener("click", downloadDatasetJsonl);
downloadBtnCsv.addEventListener("click", downloadDatasetCsv);
viewScriptBtn.addEventListener("click", showFinetuneScript);

historyToggleBtn.addEventListener("click", toggleHistorySidebar);
closeHistoryBtn.addEventListener("click", toggleHistorySidebar);

objectiveInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") generateDataset();
});
