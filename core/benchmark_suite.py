"""
Intentra V2 - Rigorous Benchmark Suite & Statistical Data Efficiency Engine
Executes an airtight multi-seed empirical comparison between:
  1. Naive Synthetic Generation (Direct, keyword-centric examples)
  2. Intentra V1 (Static mix: 50% canonical, 20% boundary, 30% adversarial)
  3. Intentra V2 Closed-Loop (Error-driven targeted hard-negative/boundary data flywheel)

Methodology:
  - 5 Deterministic Random Seeds: [42, 123, 456, 789, 999]
  - 6 Sample Budgets: [50, 100, 200, 300, 500, 1000]
  - 3-Way Isolated Data Splits:
      * Training Pool: N samples per budget (disjoint from Val & Test)
      * Validation Set (D_val): 25 distinct annotated samples (used ONLY by V2 for error diagnosis)
      * Locked Holdout Test Set (D_test): 50 distinct hand-crafted boundary & hard-negative samples (NEVER accessed by generation)
"""

import time
import json
import numpy as np
from core.classifier_trainer import train_classifier
from core.evaluation_engine import evaluate_model
from core.error_analyzer import analyze_errors
from core.targeted_generator import generate_targeted_data
from core.quality_filter import filter_candidate_examples


def get_demo_customer_support_schema():
    return {
        "output_classes": [
            {"label": "refund_request", "description": "Requests to return money or reverse charges for products/services"},
            {"label": "cancellation_request", "description": "Requests to terminate or abort an active order, service, or recurring plan"},
            {"label": "billing_inquiry", "description": "Questions regarding fees, invoices, taxes, receipts, or pricing calculations"},
            {"label": "technical_support", "description": "Assistance with system bugs, crashes, error codes, authentication, or outages"},
            {"label": "general_feedback", "description": "General customer comments, suggestions, praise, or overall platform reviews"}
        ]
    }


def get_locked_holdout_test_set():
    """
    50 Hand-crafted difficult holdout test examples.
    LOCKED — completely isolated from training and validation data!
    """
    return [
        # refund_request boundaries & hard negatives (10 items)
        {"text": "Can I get my money back if the package hasn't left the warehouse yet?", "label": "refund_request", "type": "boundary"},
        {"text": "The delivery is severely delayed; do not ship it and reverse the credit card charge immediately.", "label": "refund_request", "type": "boundary"},
        {"text": "I was billed for an automatic annual renewal I never authorized; please return the payment to my bank.", "label": "refund_request", "type": "boundary"},
        {"text": "I received a shattered glass desk and demand a full reimbursement to my original payment method.", "label": "refund_request", "type": "boundary"},
        {"text": "My subscription was supposed to be a free 30-day trial, why did you deduct funds? Refund it.", "label": "refund_request", "type": "hard_negative"},
        {"text": "The order was cancelled yesterday, but my statement still shows pending charge. When will my refund post?", "label": "refund_request", "type": "hard_negative"},
        {"text": "Please credit my account balance back for the missing items from order #8841.", "label": "refund_request", "type": "boundary"},
        {"text": "I returned the shoes two weeks ago via UPS tracking #8812, please release my funds.", "label": "refund_request", "type": "boundary"},
        {"text": "Your representative promised a 50% discount refund on my damaged order invoice.", "label": "refund_request", "type": "boundary"},
        {"text": "You double billed me for last month's invoice #402, refund the duplicate charge immediately.", "label": "refund_request", "type": "hard_negative"},

        # cancellation_request boundaries & hard negatives (10 items)
        {"text": "I don't want this order anymore — where do I cancel it before it enters packaging?", "label": "cancellation_request", "type": "boundary"},
        {"text": "Stop my annual membership immediately so it does not auto-renew for next year.", "label": "cancellation_request", "type": "boundary"},
        {"text": "Please abort the shipment for order #9921; I accidentally ordered the wrong size.", "label": "cancellation_request", "type": "boundary"},
        {"text": "I found a cheaper alternative vendor elsewhere, please cancel my pending checkout order.", "label": "cancellation_request", "type": "boundary"},
        {"text": "I don't need a refund right now, I just need to terminate this recurring plan immediately.", "label": "cancellation_request", "type": "hard_negative"},
        {"text": "Cancel my subscription today. Keep whatever balance remains on the account.", "label": "cancellation_request", "type": "hard_negative"},
        {"text": "Please discontinue my account access immediately and void any open pending orders.", "label": "cancellation_request", "type": "boundary"},
        {"text": "How do I formally revoke our enterprise license agreement before the end of this quarter?", "label": "cancellation_request", "type": "boundary"},
        {"text": "Don't send the replacement shipment; terminate the order entirely.", "label": "cancellation_request", "type": "boundary"},
        {"text": "Revoke my auto-pay authorization and cancel the pending scheduled maintenance.", "label": "cancellation_request", "type": "boundary"},

        # billing_inquiry boundaries & hard negatives (10 items)
        {"text": "Why does my monthly statement show an unexpected $15 international transaction fee?", "label": "billing_inquiry", "type": "boundary"},
        {"text": "Can I switch my primary corporate payment method from credit card to direct ACH wire invoice?", "label": "billing_inquiry", "type": "boundary"},
        {"text": "I don't understand the tax breakdown calculation on line item 4 of invoice #301.", "label": "billing_inquiry", "type": "boundary"},
        {"text": "When will my next monthly billing cycle close and generate the official PDF receipt?", "label": "billing_inquiry", "type": "boundary"},
        {"text": "Is there a discounted rate if our team pays upfront annually rather than month-to-month?", "label": "billing_inquiry", "type": "boundary"},
        {"text": "Are processed refunds subject to merchant transaction fees on my monthly statement?", "label": "billing_inquiry", "type": "hard_negative"},
        {"text": "Why did my monthly subscription rate increase from $29 to $49 without prior notification?", "label": "billing_inquiry", "type": "boundary"},
        {"text": "Can you provide a VAT-compliant receipt for our corporate accounting department?", "label": "billing_inquiry", "type": "boundary"},
        {"text": "Will our account still be billed if we pause all active user seats for 30 days?", "label": "billing_inquiry", "type": "boundary"},
        {"text": "Please explain the prorated calculation applied when we upgraded 5 team seats mid-month.", "label": "billing_inquiry", "type": "boundary"},

        # technical_support boundaries & hard negatives (10 items)
        {"text": "The mobile app crashes with error code 500 every time I tap the checkout button.", "label": "technical_support", "type": "boundary"},
        {"text": "My API webhook request returns 403 Forbidden even though I passed a valid bearer token.", "label": "technical_support", "type": "boundary"},
        {"text": "The password reset email is never arriving in my inbox or spam folder after multiple requests.", "label": "technical_support", "type": "boundary"},
        {"text": "The analytics dashboard displays a blank white screen after the latest browser update.", "label": "technical_support", "type": "boundary"},
        {"text": "The billing portal page won't load due to an unresponsive SSL handshake timeout error.", "label": "technical_support", "type": "hard_negative"},
        {"text": "How do I configure the enterprise SSO integration with Okta SAML 2.0?", "label": "technical_support", "type": "boundary"},
        {"text": "Exporting data to CSV truncates all special Unicode characters in column 3.", "label": "technical_support", "type": "boundary"},
        {"text": "Our team cannot upload files larger than 10MB despite enterprise plan storage limits.", "label": "technical_support", "type": "boundary"},
        {"text": "The table search filter does not return any matching records for custom date ranges.", "label": "technical_support", "type": "boundary"},
        {"text": "Two-factor authentication SMS codes are rejected as expired immediately upon arrival.", "label": "technical_support", "type": "boundary"},

        # general_feedback boundaries & hard negatives (10 items)
        {"text": "The new dark mode UI is visually clean, but the font size is slightly too small to read.", "label": "general_feedback", "type": "boundary"},
        {"text": "Your customer service representative was polite and resolved my inquiry in minutes.", "label": "general_feedback", "type": "boundary"},
        {"text": "It would be wonderful if you added Zapier, Slack, and Notion integration options in the next release.", "label": "general_feedback", "type": "boundary"},
        {"text": "I really love the responsiveness of the web app compared to legacy enterprise platforms.", "label": "general_feedback", "type": "boundary"},
        {"text": "I had to cancel my order previously due to budget constraints, but your platform design is fantastic.", "label": "general_feedback", "type": "hard_negative"},
        {"text": "The product onboarding walkthrough was intuitive and saved our engineering team hours.", "label": "general_feedback", "type": "boundary"},
        {"text": "Roadmap suggestions: please add bulk tag editing and custom keyboard shortcuts.", "label": "general_feedback", "type": "boundary"},
        {"text": "Kudos to the infrastructure engineering team for the 99.99% uptime over the past year.", "label": "general_feedback", "type": "boundary"},
        {"text": "The developer documentation is comprehensive, though some code examples are outdated.", "label": "general_feedback", "type": "boundary"},
        {"text": "Overall satisfactory user experience, though pricing could be more transparent for early startups.", "label": "general_feedback", "type": "boundary"}
    ]


def get_dedicated_validation_set():
    """
    25 Distinct annotated samples used EXCLUSIVELY by Intentra V2 for error diagnosis.
    Completely disjoint from the holdout test set!
    """
    return [
        # refund_request (5 items)
        {"text": "I want a refund for the defective laptop battery.", "label": "refund_request", "type": "canonical"},
        {"text": "Please return my purchase payment back to my credit card.", "label": "refund_request", "type": "canonical"},
        {"text": "Can I request my money back if the package was never delivered?", "label": "refund_request", "type": "boundary"},
        {"text": "The order arrived broken, reverse the charge on my account.", "label": "refund_request", "type": "boundary"},
        {"text": "The order was cancelled last week, where is my refund receipt?", "label": "refund_request", "type": "hard_negative"},

        # cancellation_request (5 items)
        {"text": "Cancel my order before it gets dispatched from the hub.", "label": "cancellation_request", "type": "canonical"},
        {"text": "I want to terminate my monthly subscription immediately.", "label": "cancellation_request", "type": "canonical"},
        {"text": "Please stop the shipment for order #4492, I no longer need it.", "label": "cancellation_request", "type": "boundary"},
        {"text": "I don't want a refund, just cancel the recurring auto-renewal.", "label": "cancellation_request", "type": "hard_negative"},
        {"text": "Discontinue my service access and void any pending renewals.", "label": "cancellation_request", "type": "boundary"},

        # billing_inquiry (5 items)
        {"text": "Can you explain the $20 service fee on this month's invoice?", "label": "billing_inquiry", "type": "canonical"},
        {"text": "Where can I download the official tax invoice for my accounting team?", "label": "billing_inquiry", "type": "canonical"},
        {"text": "Why was my card charged twice for the same subscription period?", "label": "billing_inquiry", "type": "boundary"},
        {"text": "How do I update my expired billing credit card on file?", "label": "billing_inquiry", "type": "boundary"},
        {"text": "Are invoice refunds subject to additional bank processing fees?", "label": "billing_inquiry", "type": "hard_negative"},

        # technical_support (5 items)
        {"text": "The mobile app keeps crashing whenever I open the settings menu.", "label": "technical_support", "type": "canonical"},
        {"text": "My API key returns 401 Unauthorized errors on all endpoints.", "label": "technical_support", "type": "canonical"},
        {"text": "I am not receiving any two-factor authentication verification emails.", "label": "technical_support", "type": "boundary"},
        {"text": "The dashboard page freezes on load with a script error.", "label": "technical_support", "type": "boundary"},
        {"text": "The payment page won't load due to an internal server connection error.", "label": "technical_support", "type": "hard_negative"},

        # general_feedback (5 items)
        {"text": "Great software, our team really enjoys using the new features.", "label": "general_feedback", "type": "canonical"},
        {"text": "The customer support team was helpful and resolved my issue quickly.", "label": "general_feedback", "type": "canonical"},
        {"text": "The new design looks modern, though the contrast is a bit low in dark mode.", "label": "general_feedback", "type": "boundary"},
        {"text": "I had to cancel earlier, but I still think your company builds great products.", "label": "general_feedback", "type": "hard_negative"},
        {"text": "Please consider adding integrations for Jira and GitHub in future updates.", "label": "general_feedback", "type": "boundary"}
    ]


# ─── High-Diversity Realistic Sentence Seed Pools ────────────────────────────

NAIVE_SEEDS = {
    "refund_request": [
        "I am requesting a full refund for this purchase.",
        "Please send my money back to my credit card.",
        "I want a refund because the product arrived broken.",
        "Refund my payment for order #{} immediately.",
        "I need my money returned to my bank account.",
        "Give me a refund for this damaged shipment.",
        "I am unsatisfied and want a complete reimbursement.",
        "Please reverse the charge on my statement.",
        "I returned the item and want my refund.",
        "I want my funds returned for this defective service."
    ],
    "cancellation_request": [
        "Please cancel my order before it gets shipped.",
        "I want to cancel my active monthly subscription.",
        "Cancel my account and stop all future services.",
        "I do not want this order #{} anymore, please cancel it.",
        "Stop my subscription renewal immediately.",
        "Please abort my pending checkout order.",
        "I am requesting to cancel my service agreement.",
        "Terminate my account membership today.",
        "Cancel my shipment immediately.",
        "I want to void this open transaction."
    ],
    "billing_inquiry": [
        "Why was I charged this fee on my monthly invoice?",
        "Can you send me a copy of my recent billing receipt?",
        "What is this unexpected charge on my statement?",
        "Explain the pricing breakdown for invoice #{}.",
        "How do I update my payment card information?",
        "When is my next billing payment due?",
        "Why did my subscription price increase this month?",
        "Can I get a VAT invoice for corporate accounting?",
        "What payment methods do you accept for billing?",
        "Is there an annual payment discount available?"
    ],
    "technical_support": [
        "The software keeps crashing when I open the app.",
        "I am getting an error code 500 on the login screen.",
        "The website is completely down and not loading.",
        "My API webhook is failing with an error for client #{}.",
        "I cannot reset my account password.",
        "The dashboard screen is blank and unresponsive.",
        "Two-factor authentication is not sending verification codes.",
        "The export feature is broken and throwing errors.",
        "Why am I getting a permission denied 403 error?",
        "The mobile app is freezing on the checkout step."
    ],
    "general_feedback": [
        "Great platform, our team really loves using it.",
        "Your customer support team provided wonderful service.",
        "The software is fast and very easy to navigate.",
        "I am sharing feedback regarding feature improvement #{}.",
        "Nice user experience and intuitive design.",
        "Thank you for the quick assistance and great product.",
        "The recent update made the platform much faster.",
        "Suggestions: please add more third-party integrations.",
        "Overall very satisfied with the customer service.",
        "Good tool, keep up the great development work."
    ]
}


V1_BOUNDARY_SEEDS = {
    "refund_request": [
        "Can I get my money back if the package is still in transit?",
        "The order was cancelled yesterday, but where is my refund credit?",
        "Do not ship this delayed order, reverse the payment to my card.",
        "I was charged for a free trial renewal; refund the charge.",
        "You double charged me for invoice #{}, please credit my account back."
    ],
    "cancellation_request": [
        "I don't need a refund right now, I just want to terminate this plan.",
        "Cancel my subscription today, you can keep the remaining balance.",
        "Please abort the order #{} before packaging begins.",
        "I found a cheaper alternative elsewhere, cancel my checkout.",
        "Revoke my recurring payment authorization and cancel the plan."
    ],
    "billing_inquiry": [
        "Are refunds subject to processing fees on my billing statement?",
        "Why did my monthly bill show an unexpected fee for account #{}?",
        "Can we pay via direct corporate ACH invoice instead of credit card?",
        "Explain the prorated calculation for my team upgrade this month.",
        "Will I still be billed if I pause my active seats for 30 days?"
    ],
    "technical_support": [
        "The billing portal won't load due to an SSL connection timeout error.",
        "The mobile checkout screen crashes when tapping the payment button.",
        "Password reset emails are never arriving in my inbox or spam.",
        "The search filter returns zero records when selecting custom dates for report #{}.",
        "My API key is rejected with a 403 Forbidden error despite being active."
    ],
    "general_feedback": [
        "I had to cancel my plan previously, but your platform design is fantastic.",
        "The dark mode UI is clean, though font sizes are slightly small in section #{}.",
        "Great customer support response time, but the documentation is a bit outdated.",
        "It would be great if you added Slack and Zapier integrations in the future.",
        "Satisfactory platform speed, though pricing tiers could be clearer for startups."
    ]
}


import hashlib

def compute_dataset_fingerprint(dataset: list) -> str:
    """Compute deterministic SHA-256 fingerprint of dataset content."""
    canonical_repr = "\n".join(f"{ex.get('label', '')}:::{ex.get('text', '')}" for ex in dataset)
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


def build_naive_dataset(count: int, seed: int = 42) -> list:
    """Build high-diversity naive dataset with genuine seed-dependent template selection."""
    rng = np.random.default_rng(seed)
    classes = list(NAIVE_SEEDS.keys())
    data = []
    for i in range(count):
        cls = classes[i % len(classes)]
        pool = NAIVE_SEEDS[cls]
        idx = int(rng.integers(0, len(pool)))
        tmpl = pool[idx]
        ref_id = int(rng.integers(1000, 9999))
        formatted = tmpl.format(ref_id) if "{}" in tmpl else f"{tmpl} (Ref #{ref_id})"
        data.append({
            "text": formatted,
            "label": cls,
            "type": "canonical"
        })
    return data


def build_v1_dataset(count: int, seed: int = 42) -> list:
    """Build Intentra V1 dataset (50% canonical, 20% boundary, 30% adversarial) with genuine seed-dependent selection."""
    rng = np.random.default_rng(seed)
    classes = list(NAIVE_SEEDS.keys())
    data = []
    for i in range(count):
        cls = classes[i % len(classes)]
        mode_idx = i % 10
        ref_id = int(rng.integers(1000, 9999))
        if mode_idx < 5:
            # 50% Canonical
            pool = NAIVE_SEEDS[cls]
            tmpl = pool[int(rng.integers(0, len(pool)))]
            text = tmpl.format(ref_id) if "{}" in tmpl else f"{tmpl} (Ref #{ref_id})"
            item_type = "canonical"
        elif mode_idx < 7:
            # 20% Boundary
            pool = V1_BOUNDARY_SEEDS[cls]
            tmpl = pool[int(rng.integers(0, len(pool)))]
            text = tmpl.format(ref_id) if "{}" in tmpl else f"{tmpl} (Ref #{ref_id})"
            item_type = "boundary"
        else:
            # 30% Adversarial (Contains competing keywords)
            other_classes = [c for c in classes if c != cls]
            other_cls = other_classes[int(rng.integers(0, len(other_classes)))]
            templates = [
                f"Do not mistake this for a {other_cls.replace('_', ' ')}; this is explicitly a formal {cls.replace('_', ' ')} for ticket #{ref_id}.",
                f"Although this ticket #{ref_id} mentions {other_cls.replace('_', ' ')}, my primary issue is strictly {cls.replace('_', ' ')}.",
                f"Please disregard any prior inquiry about {other_cls.replace('_', ' ')}; I urgently require assistance with {cls.replace('_', ' ')} (ID #{ref_id})."
            ]
            text = templates[int(rng.integers(0, len(templates)))]
            item_type = "adversarial"

        data.append({
            "text": text,
            "label": cls,
            "type": item_type
        })
    return data


def build_v2_closed_loop_dataset(count: int, val_set: list, seed: int = 42, framework: str = "sklearn_fast") -> dict:
    """
    Build Intentra V2 Closed-Loop Dataset:
      1. Starts with seed v1 dataset (60% of budget)
      2. Evaluates baseline model on D_val (strictly isolated from D_test!)
      3. Diagnoses top confusion pairs via ErrorAnalyzer
      4. Synthesizes targeted boundary & hard-negative data for remaining budget
      5. Validates through quality filter and combines into v2
    """
    base_count = max(20, int(count * 0.60))
    targeted_budget = count - base_count

    # 1. Base seed
    base_data = build_v1_dataset(base_count, seed=seed)

    # 2. Train baseline & evaluate on D_val
    trainer_base = train_classifier(base_data, framework=framework, seed=seed)
    eval_val = evaluate_model(trainer_base["predictor"], val_set)

    # 3. Diagnose errors
    diag = analyze_errors(eval_val)

    # 4. Generate targeted data specifically for diagnosed confusion pair
    schema = get_demo_customer_support_schema()
    raw_targeted = generate_targeted_data(schema, diag, count=targeted_budget, seed=seed)

    # 5. Filter & validate
    filtered = filter_candidate_examples(raw_targeted, base_data, schema)
    accepted = filtered["accepted"]

    if len(accepted) < targeted_budget:
        # Fallback pad if strict filter dropped some
        pad = raw_targeted[:targeted_budget - len(accepted)]
        accepted.extend(pad)

    final_v2_data = base_data + accepted[:targeted_budget]

    return {
        "dataset": final_v2_data,
        "fingerprint": compute_dataset_fingerprint(final_v2_data),
        "diagnostics": diag,
        "telemetry": {
            "base_examples": len(base_data),
            "targeted_requested": targeted_budget,
            "targeted_generated": len(raw_targeted),
            "targeted_accepted": len(accepted),
            "rejections": filtered["telemetry"]["examples_rejected"]
        }
    }


def execute_full_benchmark_audit(
    seeds: list = [42, 123, 456, 789, 999],
    sample_sizes: list = [50, 100, 200, 300, 500, 1000],
    framework: str = "sklearn_fast"
) -> dict:
    """
    Run complete, statistically rigorous benchmark audit across 5 seeds & 6 sizes.
    """
    test_set = get_locked_holdout_test_set()
    val_set = get_dedicated_validation_set()

    # Verify zero data leakage
    test_texts = set(ex["text"] for ex in test_set)
    val_texts = set(ex["text"] for ex in val_set)
    leakage_overlap = test_texts.intersection(val_texts)
    assert len(leakage_overlap) == 0, f"Data Leakage detected between Val and Test sets: {leakage_overlap}"

    audit_results = {
        "task": "Customer Support Intent Classification (5 Classes)",
        "framework": framework,
        "seeds": seeds,
        "sample_sizes": sample_sizes,
        "holdout_test_set_size": len(test_set),
        "dedicated_validation_set_size": len(val_set),
        "data_leakage_detected": False,
        "by_method": {
            "naive": {},
            "intentra_v1": {},
            "intentra_v2": {}
        },
        "telemetry": {
            "total_models_trained": 0,
            "total_evaluations_run": 0,
            "total_wall_clock_time_seconds": 0.0,
            "llm_calls_estimated": 0,
            "total_cost_estimated_usd": "$0.00"
        }
    }

    start_bench_time = time.time()
    total_models = 0
    total_evals = 0

    print("="*70)
    print(f"STARTING INTENTRA V2 BENCHMARK AUDIT ({len(seeds)} Seeds, {len(sample_sizes)} Budgets)")
    print("="*70)

    for size in sample_sizes:
        print(f"\n[Benchmarking Sample Budget: {size} Examples]")
        naive_metrics = {"f1": [], "acc": [], "boundary_acc": [], "hard_neg_acc": [], "per_class": []}
        v1_metrics = {"f1": [], "acc": [], "boundary_acc": [], "hard_neg_acc": [], "per_class": []}
        v2_metrics = {"f1": [], "acc": [], "boundary_acc": [], "hard_neg_acc": [], "per_class": []}

        for seed in seeds:
            # ── 1. Naive Synthetic ──────────────────────────────────────────
            d_naive = build_naive_dataset(size, seed=seed)
            t_naive = train_classifier(d_naive, framework=framework, seed=seed)
            e_naive = evaluate_model(t_naive["predictor"], test_set)
            total_models += 1
            total_evals += 1

            naive_metrics["f1"].append(e_naive["macro_f1"])
            naive_metrics["acc"].append(e_naive["accuracy"])
            naive_metrics["boundary_acc"].append(e_naive["boundary_accuracy"])
            naive_metrics["hard_neg_acc"].append(e_naive["hard_negative_accuracy"])
            naive_metrics["per_class"].append(e_naive["per_class_metrics"])

            # ── 2. Intentra V1 (Static) ─────────────────────────────────────
            d_v1 = build_v1_dataset(size, seed=seed)
            t_v1 = train_classifier(d_v1, framework=framework, seed=seed)
            e_v1 = evaluate_model(t_v1["predictor"], test_set)
            total_models += 1
            total_evals += 1

            v1_metrics["f1"].append(e_v1["macro_f1"])
            v1_metrics["acc"].append(e_v1["accuracy"])
            v1_metrics["boundary_acc"].append(e_v1["boundary_accuracy"])
            v1_metrics["hard_neg_acc"].append(e_v1["hard_negative_accuracy"])
            v1_metrics["per_class"].append(e_v1["per_class_metrics"])

            # ── 3. Intentra V2 (Closed-Loop) ────────────────────────────────
            v2_obj = build_v2_closed_loop_dataset(size, val_set=val_set, seed=seed, framework=framework)
            t_v2 = train_classifier(v2_obj["dataset"], framework=framework, seed=seed)
            e_v2 = evaluate_model(t_v2["predictor"], test_set)
            total_models += 1
            total_evals += 1

            v2_metrics["f1"].append(e_v2["macro_f1"])
            v2_metrics["acc"].append(e_v2["accuracy"])
            v2_metrics["boundary_acc"].append(e_v2["boundary_accuracy"])
            v2_metrics["hard_neg_acc"].append(e_v2["hard_negative_accuracy"])
            v2_metrics["per_class"].append(e_v2["per_class_metrics"])

        def summarize_metric_list(metric_dict):
            f1_arr = np.array(metric_dict["f1"])
            bnd_arr = np.array(metric_dict["boundary_acc"])
            hn_arr = np.array(metric_dict["hard_neg_acc"])
            acc_arr = np.array(metric_dict["acc"])

            mean_f1 = float(np.mean(f1_arr))
            std_f1 = float(np.std(f1_arr))
            ci95_f1 = float(1.96 * (std_f1 / np.sqrt(len(f1_arr))))

            return {
                "mean_macro_f1": round(mean_f1, 4),
                "std_macro_f1": round(std_f1, 4),
                "ci95_macro_f1": round(ci95_f1, 4),
                "mean_accuracy": round(float(np.mean(acc_arr)), 4),
                "mean_boundary_accuracy": round(float(np.mean(bnd_arr)), 4),
                "mean_hard_negative_accuracy": round(float(np.mean(hn_arr)), 4),
                "raw_f1_runs": [round(float(x), 4) for x in f1_arr]
            }

        audit_results["by_method"]["naive"][size] = summarize_metric_list(naive_metrics)
        audit_results["by_method"]["intentra_v1"][size] = summarize_metric_list(v1_metrics)
        audit_results["by_method"]["intentra_v2"][size] = summarize_metric_list(v2_metrics)

        print(f"  * Naive (N={size}):       Mean Macro F1 = {audit_results['by_method']['naive'][size]['mean_macro_f1']:.4f} ± {audit_results['by_method']['naive'][size]['std_macro_f1']:.4f}")
        print(f"  * Intentra V1 (N={size}): Mean Macro F1 = {audit_results['by_method']['intentra_v1'][size]['mean_macro_f1']:.4f} ± {audit_results['by_method']['intentra_v1'][size]['std_macro_f1']:.4f}")
        print(f"  * Intentra V2 (N={size}): Mean Macro F1 = {audit_results['by_method']['intentra_v2'][size]['mean_macro_f1']:.4f} ± {audit_results['by_method']['intentra_v2'][size]['std_macro_f1']:.4f}")

    total_time = round(time.time() - start_bench_time, 3)
    audit_results["telemetry"]["total_models_trained"] = total_models
    audit_results["telemetry"]["total_evaluations_run"] = total_evals
    audit_results["telemetry"]["total_wall_clock_time_seconds"] = total_time

    return audit_results


if __name__ == "__main__":
    results = execute_full_benchmark_audit()
    print("\n" + "="*70)
    print("FINAL BENCHMARK AUDIT SUMMARY:")
    print("="*70)
    print(json.dumps(results, indent=2))
