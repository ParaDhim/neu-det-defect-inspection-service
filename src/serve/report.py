"""
Automated Defect Inspection & Metallurgical Engineering Report Generator.
Synthesizes bounding box detections into plain-language plant QA summaries with
severity assessment, root-cause diagnosis, and actionable coil disposition recommendations.
Supports optional external LLM integration (Anthropic Claude / OpenAI) with deterministic fallback.
"""

import os
import json
import requests
from datetime import datetime

METALLURGICAL_KNOWLEDGE = {
    "crazing": {
        "severity": "CRITICAL",
        "description": "Network of fine surface micro-cracks on the strip surface.",
        "root_cause": "Thermal fatigue cycling, excessive roll heating in finishing mill stands, or poor lubrication.",
        "consequence": "Risk of severe strip rupture during downstream cold reduction or deep drawing operations.",
        "recommended_action": "QUARANTINE COIL. Perform dye-penetrant test. Inspect work roll cooling headers.",
    },
    "rolled-in_scale": {
        "severity": "CRITICAL",
        "description": "Iron oxide scale pressed firmly into the steel substrate during high-pressure rolling passes.",
        "root_cause": "Descaling header nozzle clogging, insufficient primary descaling water pressure (<150 bar), or secondary oxidation delay.",
        "consequence": "Surface blisters, severe plating/galvanizing delamination, and premature localized pitting corrosion.",
        "recommended_action": "REJECT FOR SURFACE-CRITICAL ORDERS. Reroute to abrasive blasting or heavy acid pickling line.",
    },
    "patches": {
        "severity": "MODERATE",
        "description": "Irregular localized surface indentations or partial thickness tears.",
        "root_cause": "Foreign particulate matter on intermediate transfer tables or localized cold-welding adhesion on guide plates.",
        "consequence": "Dimensional gauge inconsistency and poor visual aesthetic finish.",
        "recommended_action": "DOWNGRADE to Grade B/Structural framing. Clean conveyor apron plates.",
    },
    "inclusion": {
        "severity": "MODERATE",
        "description": "Non-metallic metallurgical inclusions (alumina, silicates, slag entrainment) exposed at the surface.",
        "root_cause": "Ladle shroud misalignment during continuous casting, insufficient tundish residence time, or refractory erosion.",
        "consequence": "Stress concentration notch causing brittle fracture during bend testing.",
        "recommended_action": "FLAG CASTER HEAT. Divert coil to non-critical automotive underbody or general construction applications.",
    },
    "pitted_surface": {
        "severity": "LOW",
        "description": "Periodic crater-like surface depressions caused by roll pitting or chemical etching unevenness.",
        "root_cause": "Rough or work-hardened work roll surface, roll spalling, or localized acid over-pickling.",
        "consequence": "Minor visual texture variation, acceptable under standard primer coats.",
        "recommended_action": "ACCEPT WITH CAUTION. Schedule work roll re-grinding at next scheduled shift maintenance.",
    },
    "scratches": {
        "severity": "LOW",
        "description": "Linear longitudinal abrasions along strip transit direction.",
        "root_cause": "Contact friction against misaligned edge guides, dead rollers, or side trimmers on run-out table.",
        "consequence": "Superficial visual blemish, negligible impact on mechanical yield strength.",
        "recommended_action": "PASS. Flag tension leveler entry guides for maintenance inspection.",
    },
}


def generate_deterministic_report(detections: list, image_meta: dict = None) -> dict:
    """
    Generates a structured metallurgical inspection report using domain engineering rules.
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    total_defects = len(detections)

    if total_defects == 0:
        return {
            "timestamp": timestamp,
            "overall_status": "PASS",
            "risk_level": "NEGLIGIBLE",
            "defect_count": 0,
            "summary": "Coil surface is defect-free. No surface anomalies detected above confidence threshold.",
            "disposition": "RELEASE TO PRODUCTION. Surface meets Class A metallurgical quality standards.",
            "defects_breakdown": [],
            "action_items": ["Proceed with scheduled downstream processing."],
        }

    # Count by class
    counts = {}
    severities = []
    for d in detections:
        c_name = d["class_name"]
        counts[c_name] = counts.get(c_name, 0) + 1
        meta = METALLURGICAL_KNOWLEDGE.get(c_name, {"severity": "LOW"})
        severities.append(meta["severity"])

    # Determine overall status
    if "CRITICAL" in severities or total_defects >= 4:
        overall_status = "REJECT"
        risk_level = "HIGH"
    elif "MODERATE" in severities or total_defects >= 2:
        overall_status = "HOLD / DOWNGRADE"
        risk_level = "ELEVATED"
    else:
        overall_status = "PASS WITH ADVISORY"
        risk_level = "LOW"

    # Compile defect breakdown
    breakdown = []
    action_items = set()
    root_causes = []

    for c_name, count in counts.items():
        info = METALLURGICAL_KNOWLEDGE.get(c_name, {})
        breakdown.append({
            "class": c_name,
            "count": count,
            "severity": info.get("severity", "UNKNOWN"),
            "description": info.get("description", ""),
            "consequence": info.get("consequence", ""),
        })
        action_items.add(info.get("recommended_action", "Inspect component."))
        root_causes.append(f"{c_name.capitalize()}: {info.get('root_cause', '')}")

    summary_text = (
        f"Inspection completed with {total_defects} defect(s) detected across {len(counts)} defect category(ies). "
        f"Overall Coil Disposition: [{overall_status}]. "
        f"Dominant defects identified: {', '.join([f'{k} (x{v})' for k, v in counts.items()])}."
    )

    return {
        "timestamp": timestamp,
        "overall_status": overall_status,
        "risk_level": risk_level,
        "defect_count": total_defects,
        "summary": summary_text,
        "disposition": "QUARANTINE COIL" if overall_status == "REJECT" else "CONDITIONAL RELEASE / RE-INSPECT",
        "defects_breakdown": breakdown,
        "probable_root_causes": root_causes,
        "action_items": list(action_items),
        "source": "deterministic_metallurgical_rules",
    }


def generate_llm_inspection_report(detections: list, api_key: str = None) -> dict:
    """
    Generates a natural language inspection report.
    If ANTHROPIC_API_KEY is available, calls Claude for advanced synthesis;
    otherwise seamlessly uses the deterministic metallurgical engine.
    """
    anthropic_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    # If no key, use domain deterministic engine
    if not anthropic_key:
        return generate_deterministic_report(detections)

    # Optional Claude API synthesis
    try:
        headers = {
            "x-api-key": anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        prompt = f"""You are a Lead Metallurgical Quality Engineer at a flat-rolled steel production plant.
Analyze the following automated visual defect detection results for a steel coil segment:
Detections: {json.dumps(detections, indent=2)}

Generate a concise, professional JSON inspection report with the following keys:
- overall_status: "PASS", "HOLD / DOWNGRADE", or "REJECT"
- risk_level: "NEGLIGIBLE", "LOW", "ELEVATED", or "HIGH"
- summary: 2-3 sentences plain-language factory summary
- root_cause_hypothesis: metallurgical cause in rolling/casting
- disposition_recommendation: immediate actionable instruction for line operators
Return only valid JSON."""

        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }
        res = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=8)
        if res.status_code == 200:
            content = res.json()["content"][0]["text"]
            # Clean json
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != -1:
                parsed = json.loads(content[start:end])
                parsed["source"] = "claude-3-haiku"
                parsed["defect_count"] = len(detections)
                return parsed
    except Exception as e:
        print(f"LLM call fallback to deterministic rules: {e}")

    return generate_deterministic_report(detections)
