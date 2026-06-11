"""
Simple HDFS prediction translator.

Purpose:
Convert Transformer prediction rows into short operator-facing explanations.
Designed for the prediction database with columns such as:
dataset, seq_id, input_seq, target_seq, target_event, pred_events, pred_probs,
target_event_is_unseen, target_in_topk_pred, target_in_pred_pos,
target_in_topk_pred_prob, results_matrix, sequence_results_matrix.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

PAD_ID = 0
ANOMALY_MARKER = -1


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def parse_list(value: Any) -> List[Any]:
    """Parse list-like cells such as [1, 2], '1 2', or 'tensor([1, 2])'."""
    if is_missing(value):
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return list(value)

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return []

    tensor_match = re.fullmatch(r"tensor\((.*)\)", text)
    if tensor_match:
        text = tensor_match.group(1).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
        return [parsed]
    except Exception:
        return re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, flags=re.IGNORECASE)


def flatten(values: Any) -> List[Any]:
    if not isinstance(values, (list, tuple, set)):
        return [values]

    output = []
    for item in values:
        output.extend(flatten(item))
    return output


def clean_sequence(
    value: Any,
    pad_id: int = PAD_ID,
    anomaly_marker: int = ANOMALY_MARKER,
    bos_id: Optional[int] = None,
) -> List[int]:
    cleaned = []

    for item in flatten(parse_list(value)):
        if is_missing(item):
            continue
        event_id = int(float(item))
        if event_id in {pad_id, anomaly_marker}:
            continue
        if bos_id is not None and event_id == bos_id:
            continue
        cleaned.append(event_id)

    return cleaned


def to_bool(value: Any) -> Optional[bool]:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(value: Any) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def percent(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.1%}"


def first_available(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and not is_missing(row[name]):
            return row[name]
    return None


def load_event_templates(csv_path: str) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    templates = {}

    for _, row in df.iterrows():
        raw_key = first_available(row, ["Log Key", "EventId", "EventID", "event_id"])
        if is_missing(raw_key):
            continue

        try:
            key = str(int(float(raw_key)))
        except Exception:
            key = str(raw_key).strip()

        templates[key] = {
            "message": str(first_available(row, ["Message", "EventTemplate", "Template"]) or ""),
            "occurrences": first_available(row, ["Occurrences", "Count", "count"]),
        }

    return templates


def event_label(event_id: Optional[int], templates: Optional[Mapping[str, Mapping[str, Any]]] = None,
                max_chars: int = 90) -> str:
    if event_id is None:
        return "N/A"

    label = f"Event {int(event_id)}"
    if templates:
        message = str(templates.get(str(int(event_id)), {}).get("message", "")).strip()
        if message and message.lower() != "nan":
            label = f"{label}: {message}"

    return label if len(label) <= max_chars else label[:max_chars - 3] + "..."


@dataclass
class TrainingReference:
    transition_counts: Dict[int, Counter] = field(default_factory=dict)
    ending_event_counts: Counter = field(default_factory=Counter)
    event_counts: Counter = field(default_factory=Counter)

    @property
    def known_event_ids(self) -> set:
        return set(self.event_counts.keys())

    @property
    def ending_event_ids(self) -> set:
        return set(self.ending_event_counts.keys())

    def transition_count(self, prev_event: Optional[int], next_event: Optional[int]) -> int:
        if prev_event is None or next_event is None:
            return 0
        return int(self.transition_counts.get(int(prev_event), Counter()).get(int(next_event), 0))

    def transition_total(self, prev_event: Optional[int]) -> int:
        if prev_event is None:
            return 0
        return int(sum(self.transition_counts.get(int(prev_event), Counter()).values()))

    def transition_probability(self, prev_event: Optional[int], next_event: Optional[int]) -> Optional[float]:
        total = self.transition_total(prev_event)
        if total == 0:
            return None
        return self.transition_count(prev_event, next_event) / total


def build_training_reference(training_sequences: Iterable[Any], bos_id: Optional[int] = None) -> TrainingReference:
    transition_counts = defaultdict(Counter)
    ending_event_counts = Counter()
    event_counts = Counter()

    for raw_seq in training_sequences:
        seq = clean_sequence(raw_seq, bos_id=bos_id)
        if not seq:
            continue

        event_counts.update(seq)
        ending_event_counts[seq[-1]] += 1

        for prev_event, next_event in zip(seq[:-1], seq[1:]):
            transition_counts[int(prev_event)][int(next_event)] += 1

    return TrainingReference(
        transition_counts=dict(transition_counts),
        ending_event_counts=ending_event_counts,
        event_counts=event_counts,
    )


def load_training_sequences_from_txt(path: str, bos_id: Optional[int] = None) -> List[List[int]]:
    sequences = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            seq = clean_sequence(line, bos_id=bos_id)
            if seq:
                sequences.append(seq)

    return sequences


def previous_event_from_row(row: Mapping[str, Any], actual_event: Optional[int], bos_id: Optional[int] = None) -> Optional[int]:
    input_seq = clean_sequence(first_available(row, ["input_seq", "input_keys", "src", "source_sequence"]), bos_id=bos_id)
    target_seq = clean_sequence(first_available(row, ["target_seq", "actual_next_seq", "actual_seq", "tgt"]), bos_id=bos_id)

    if actual_event is not None and len(target_seq) >= 2 and target_seq[-1] == actual_event:
        return target_seq[-2]

    if input_seq:
        return input_seq[-1]

    return None


def extract_actual_event(row: Mapping[str, Any], bos_id: Optional[int] = None) -> Optional[int]:
    value = first_available(row, ["target_event", "actual_event", "actual_event_id", "true_next_event_id"])
    if not is_missing(value):
        return int(float(value))

    target_seq = clean_sequence(first_available(row, ["target_seq", "actual_next_seq", "actual_seq", "tgt"]), bos_id=bos_id)
    return target_seq[-1] if target_seq else None


def extract_topk(row: Mapping[str, Any]) -> List[int]:
    values = first_available(row, ["pred_events", "top_k_event_ids", "topk_event_ids", "topk_keys", "top_5_pred"])
    return [int(float(x)) for x in flatten(parse_list(values)) if not is_missing(x) and int(float(x)) != PAD_ID]


def extract_probs(row: Mapping[str, Any]) -> List[float]:
    values = first_available(row, ["pred_probs", "top_k_probs", "topk_probs", "top_5_prob"])
    return [float(x) for x in flatten(parse_list(values)) if not is_missing(x)]


def infer_status(row: Mapping[str, Any], hit_topk: Optional[bool]) -> str:
    result = str(first_available(row, ["results_matrix", "sequence_results_matrix", "result_type"]) or "").lower()

    if "true positive" in result or "false positive" in result:
        return "Anomaly"
    if "true negative" in result or "false negative" in result:
        return "Normal"

    predicted_status = str(first_available(row, ["predicted_status", "sequence_predicted_status"]) or "").lower()
    if predicted_status in {"normal", "anomaly", "abnormal"}:
        return "Anomaly" if predicted_status in {"anomaly", "abnormal"} else "Normal"

    return "Anomaly" if hit_topk is False else "Normal"


def estimate_severity(
    status: str,
    hit_topk: Optional[bool],
    unseen_event: Optional[bool],
    transition_status: str,
    target_probability: Optional[float],
) -> str:
    if status == "Normal":
        return "Normal"

    score = 0
    if hit_topk is False:
        score += 2
    if unseen_event is True:
        score += 2
    if transition_status == "unseen":
        score += 1
    if target_probability is not None and target_probability < 0.05:
        score += 1

    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def translate_prediction_row(
    row: Mapping[str, Any] | pd.Series,
    transition_reference: Optional[TrainingReference] = None,
    templates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    top_k: int = 9,
    rare_transition_threshold: float = 0.02,
    bos_id: Optional[int] = None,
) -> Dict[str, Any]:
    row = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    sequence_id = first_available(row, ["seq_id", "sequence_id", "id"])
    dataset = first_available(row, ["dataset"])
    result_type = first_available(row, ["results_matrix", "sequence_results_matrix"])

    actual_event = extract_actual_event(row, bos_id=bos_id)
    previous_event = previous_event_from_row(row, actual_event, bos_id=bos_id)

    topk_events = extract_topk(row)[:top_k]
    topk_probs = extract_probs(row)[:top_k]

    hit_topk = to_bool(first_available(row, ["target_in_topk_pred", "hit_top_k"]))
    if hit_topk is None and actual_event is not None and topk_events:
        hit_topk = actual_event in topk_events

    rank = safe_float(first_available(row, ["target_in_pred_pos", "actual_rank_in_topk"]))
    if rank is None and actual_event is not None and actual_event in topk_events:
        rank = topk_events.index(actual_event) + 1

    target_probability = safe_float(first_available(row, ["target_in_topk_pred_prob", "actual_probability", "true_event_probability"]))
    if target_probability is None and rank is not None and topk_probs:
        prob_index = int(rank) - 1
        if 0 <= prob_index < len(topk_probs):
            target_probability = topk_probs[prob_index]

    unseen_event = to_bool(first_available(row, ["target_event_is_unseen", "unseen_event"]))
    if unseen_event is None and transition_reference is not None and actual_event is not None:
        unseen_event = actual_event not in transition_reference.known_event_ids

    transition_probability = None
    transition_status = "not checked"

    if transition_reference is not None and previous_event is not None and actual_event is not None:
        transition_probability = transition_reference.transition_probability(previous_event, actual_event)
        transition_count = transition_reference.transition_count(previous_event, actual_event)
        transition_total = transition_reference.transition_total(previous_event)

        if transition_total == 0 or transition_count == 0:
            transition_status = "unseen"
        elif transition_probability is not None and transition_probability < rare_transition_threshold:
            transition_status = "rare"
        else:
            transition_status = "seen"

    status = infer_status(row, hit_topk)
    severity = estimate_severity(status, hit_topk, unseen_event, transition_status, target_probability)

    evidence = []
    if hit_topk is False:
        evidence.append(f"Actual event was not found in the model top-{top_k} candidates.")
    elif hit_topk is True:
        evidence.append(f"Actual event was inside top-{top_k} at rank {int(rank) if rank else 'N/A'}.")

    if target_probability is not None:
        evidence.append(f"Model probability for the actual event was {percent(target_probability)}.")

    if unseen_event is True:
        evidence.append("Actual EventID was unseen in the normal training reference.")
    elif unseen_event is False:
        evidence.append("Actual EventID was seen in the normal training reference.")

    if transition_status == "unseen":
        evidence.append(f"Transition {previous_event} -> {actual_event} was not seen in normal training.")
    elif transition_status == "rare":
        evidence.append(f"Transition {previous_event} -> {actual_event} was rare in normal training ({percent(transition_probability)}).")
    elif transition_status == "seen":
        evidence.append(f"Transition {previous_event} -> {actual_event} was seen in normal training ({percent(transition_probability)}).")

    if not evidence:
        evidence.append("Translator used available model output only; semantic reference was unavailable.")

    if status == "Anomaly":
        summary = (
            f"Sequence {sequence_id}: ANOMALY ({severity}). "
            f"{event_label(actual_event, templates)} occurred after {event_label(previous_event, templates)}."
        )
        action = (
            f"Review the log context around transition {previous_event} -> {actual_event}; "
            "check whether this event order is expected for the current HDFS block operation."
        )
    else:
        summary = (
            f"Sequence {sequence_id}: NORMAL. "
            f"The actual event is consistent with the model candidate set."
        )
        action = "Continue normal monitoring."

    return {
        "sequence_id": sequence_id,
        "dataset": dataset,
        "result_type": result_type,
        "status": status,
        "severity": severity,
        "previous_event_id": previous_event,
        "actual_event_id": actual_event,
        "previous_event": event_label(previous_event, templates),
        "actual_event": event_label(actual_event, templates),
        "hit_topk": hit_topk,
        "rank_in_topk": int(rank) if rank is not None and not math.isnan(rank) else None,
        "target_probability": target_probability,
        "target_probability_pct": percent(target_probability),
        "unseen_event": unseen_event,
        "transition_status": transition_status,
        "transition_probability": transition_probability,
        "transition_probability_pct": percent(transition_probability),
        "topk_event_ids": topk_events,
        "operator_summary": summary,
        "evidence_1": evidence[0] if len(evidence) > 0 else "",
        "evidence_2": evidence[1] if len(evidence) > 1 else "",
        "evidence_3": evidence[2] if len(evidence) > 2 else "",
        "evidence_4": evidence[3] if len(evidence) > 3 else "",
        "operator_action": action,
    }


def translate_dataframe(
    prediction_df: pd.DataFrame,
    transition_reference: Optional[TrainingReference] = None,
    templates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    top_k: int = 9,
    rare_transition_threshold: float = 0.02,
    bos_id: Optional[int] = None,
) -> pd.DataFrame:
    records = []

    for _, row in prediction_df.iterrows():
        records.append(
            translate_prediction_row(
                row=row,
                transition_reference=transition_reference,
                templates=templates,
                top_k=top_k,
                rare_transition_threshold=rare_transition_threshold,
                bos_id=bos_id,
            )
        )

    return pd.DataFrame(records)


def format_operator_message(explanation: Mapping[str, Any]) -> str:
    evidence = [
        explanation.get("evidence_1", ""),
        explanation.get("evidence_2", ""),
        explanation.get("evidence_3", ""),
        explanation.get("evidence_4", ""),
    ]
    evidence = [x for x in evidence if x]

    lines = [str(explanation.get("operator_summary", "")), "", "Evidence:"]
    for i, item in enumerate(evidence, start=1):
        lines.append(f"{i}. {item}")

    action = explanation.get("operator_action")
    if action:
        lines.extend(["", f"Suggested check: {action}"])

    return "\n".join(lines)


def save_operator_report(report_df: pd.DataFrame, output_path: str) -> None:
    report_df.to_csv(output_path, index=False)