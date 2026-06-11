"""
Baseline translator for Transformer anomaly predictions.

Purpose
-------
Convert raw sequence prediction results into short, operator-friendly
explanations such as:

    Sequence 1: ANOMALY
    Observed event 20 after event 19.
    Why:
    1. Event 20 was not in the model top-k candidates.
    2. Transition 19 -> 20 was not seen in the training reference.
    3. Event 20 is not a normal ending event.

This is intentionally simple for the first baseline. Later, SHAP or other XAI
results can be added as extra reason columns without changing the operator view.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd


PAD_ID = 0
ANOMALY_MARKER = -1


# ---------------------------------------------------------------------------
# Basic parsers
# ---------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    """Return True for None / NaN-like values."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _strip_tensor_wrapper(text: str) -> str:
    """Convert strings like 'tensor([1, 2])' to '[1, 2]'."""
    text = text.strip()
    match = re.fullmatch(r"tensor\((.*)\)", text)
    if match:
        return match.group(1).strip()
    return text


def _convert_nested(value: Any, target_type: str = "int") -> Any:
    """Convert nested lists/tuples into int/float values."""
    if isinstance(value, (list, tuple, set)):
        return [_convert_nested(v, target_type=target_type) for v in value]

    if _is_missing(value):
        return None

    if target_type == "float":
        return float(value)

    # int parser: convert floats such as 1.0 into 1
    return int(float(value))


def _numbers_from_text(text: str, target_type: str = "int") -> List[Union[int, float]]:
    """Fallback parser for strings containing numbers."""
    pattern = r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    numbers = re.findall(pattern, text, flags=re.IGNORECASE)
    if target_type == "float":
        return [float(x) for x in numbers]
    return [int(float(x)) for x in numbers]


def parse_list(value: Any) -> List[Any]:
    """
    Parse a sequence-like value into a Python list.

    Handles:
    - [1, 2, 3]
    - "1 2 3"
    - "tensor([1, 2, 3])"
    - nested lists like "[[1, 2, 3], [4, 5, 6]]"
    """
    if _is_missing(value):
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        converted = _convert_nested(value, target_type="int")
        return [] if converted is None else list(converted)

    text = _strip_tensor_wrapper(str(value).strip())
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        converted = _convert_nested(parsed, target_type="int")
        if converted is None:
            return []
        if isinstance(converted, list):
            return converted
        return [converted]
    except Exception:
        return _numbers_from_text(text, target_type="int")


def parse_float_list(value: Any) -> List[Any]:
    """Same as parse_list, but keeps numbers as floats."""
    if _is_missing(value):
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        converted = _convert_nested(value, target_type="float")
        return [] if converted is None else list(converted)

    text = _strip_tensor_wrapper(str(value).strip())
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        converted = _convert_nested(parsed, target_type="float")
        if converted is None:
            return []
        if isinstance(converted, list):
            return converted
        return [converted]
    except Exception:
        return _numbers_from_text(text, target_type="float")


def flatten(values: Any) -> List[Any]:
    """Flatten nested lists while preserving order."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        return [values]

    out: List[Any] = []
    for item in values:
        out.extend(flatten(item))
    return out


def clean_sequence(
    sequence: Any,
    pad_id: Optional[int] = PAD_ID,
    anomaly_marker: Optional[int] = ANOMALY_MARKER,
) -> List[int]:
    """Parse and remove padding/anomaly marker from a sequence."""
    parsed = parse_list(sequence)
    flat = flatten(parsed)

    cleaned: List[int] = []
    for item in flat:
        if item is None:
            continue
        event_id = int(item)
        if pad_id is not None and event_id == pad_id:
            continue
        if anomaly_marker is not None and event_id == anomaly_marker:
            continue
        cleaned.append(event_id)

    return cleaned


# ---------------------------------------------------------------------------
# Event template helpers
# ---------------------------------------------------------------------------

def load_event_templates(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load event templates from a CSV with columns:
    - Log Key
    - Message
    - Occurrences

    Returns a dictionary keyed by event ID as string.
    """
    df = pd.read_csv(csv_path)
    templates: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        raw_key = row.get("Log Key", row.get("EventId", row.get("event_id")))
        if _is_missing(raw_key):
            continue

        # Handles keys stored as 1 or 1.0
        key = str(int(float(raw_key))) if str(raw_key).replace(".", "", 1).isdigit() else str(raw_key).strip()
        templates[key] = {
            "raw_message": row.get("Message", ""),
            "occurrences": row.get("Occurrences", ""),
        }

    return templates


def event_name(event_id: Optional[int], templates: Optional[Mapping[str, Mapping[str, Any]]] = None) -> str:
    """Return a readable event label."""
    if event_id is None:
        return "N/A"

    event_id_int = int(event_id)
    if not templates:
        return f"Event {event_id_int}"

    template = templates.get(str(event_id_int), {})
    message = str(template.get("raw_message", "")).strip()

    if not message or message.lower() == "nan":
        return f"Event {event_id_int}"

    return f"Event {event_id_int}: {message}"


def short_event_name(
    event_id: Optional[int],
    templates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    max_chars: int = 95,
) -> str:
    """Return a compact event label for table/report display."""
    name = event_name(event_id, templates=templates)
    if len(name) <= max_chars:
        return name
    return name[: max_chars - 3] + "..."


# ---------------------------------------------------------------------------
# Training reference: transition and ending-event statistics
# ---------------------------------------------------------------------------

@dataclass
class TransitionReference:
    """
    Simple reference built from normal training sequences.

    transition_counts:
        previous_event -> Counter(next_event)

    ending_event_counts:
        event -> number of sequences ending with this event
    """

    transition_counts: Dict[int, Counter] = field(default_factory=dict)
    ending_event_counts: Counter = field(default_factory=Counter)
    event_counts: Counter = field(default_factory=Counter)

    @property
    def ending_event_ids(self) -> set:
        return set(self.ending_event_counts.keys())

    @property
    def known_event_ids(self) -> set:
        return set(self.event_counts.keys())

    def transition_total(self, previous_event: Optional[int]) -> int:
        if previous_event is None:
            return 0
        return int(sum(self.transition_counts.get(int(previous_event), Counter()).values()))

    def transition_count(self, previous_event: Optional[int], next_event: Optional[int]) -> int:
        if previous_event is None or next_event is None:
            return 0
        return int(self.transition_counts.get(int(previous_event), Counter()).get(int(next_event), 0))

    def transition_probability(self, previous_event: Optional[int], next_event: Optional[int]) -> Optional[float]:
        total = self.transition_total(previous_event)
        if total <= 0:
            return None
        return self.transition_count(previous_event, next_event) / total

    def top_next_events(self, previous_event: Optional[int], n: int = 5) -> List[Tuple[int, int, float]]:
        """Return [(event_id, count, probability), ...] for the most common next events."""
        if previous_event is None:
            return []

        counter = self.transition_counts.get(int(previous_event), Counter())
        total = sum(counter.values())
        if total <= 0:
            return []

        return [(event_id, count, count / total) for event_id, count in counter.most_common(n)]


def build_transition_reference(
    training_sequences: Iterable[Any],
    pad_id: int = PAD_ID,
    anomaly_marker: int = ANOMALY_MARKER,
) -> TransitionReference:
    """Build transition and ending-event statistics from normal training sequences."""
    transition_counts: Dict[int, Counter] = defaultdict(Counter)
    ending_event_counts: Counter = Counter()
    event_counts: Counter = Counter()

    for raw_sequence in training_sequences:
        sequence = clean_sequence(raw_sequence, pad_id=pad_id, anomaly_marker=anomaly_marker)
        if not sequence:
            continue

        event_counts.update(sequence)
        ending_event_counts[sequence[-1]] += 1

        for previous_event, next_event in zip(sequence[:-1], sequence[1:]):
            transition_counts[int(previous_event)][int(next_event)] += 1

    return TransitionReference(
        transition_counts=dict(transition_counts),
        ending_event_counts=ending_event_counts,
        event_counts=event_counts,
    )


def load_training_sequences_from_txt(path: str, max_rows: Optional[int] = None) -> List[List[int]]:
    """
    Load a simple text dataset where each line contains event IDs separated by spaces.
    This matches the style used by the Transformer.py data generator.
    """
    sequences: List[List[int]] = []

    with open(path, "r", encoding="utf-8") as file:
        for row_number, line in enumerate(file):
            if max_rows is not None and row_number >= max_rows:
                break
            sequence = clean_sequence(line)
            if sequence:
                sequences.append(sequence)

    return sequences


# ---------------------------------------------------------------------------
# Prediction explanation
# ---------------------------------------------------------------------------

def _is_nested(values: Any) -> bool:
    return isinstance(values, list) and any(isinstance(item, (list, tuple, set)) for item in values)


def _extract_step_items(
    values: Any,
    step_index: int,
    actual_len: int,
    k: Optional[int] = None,
    value_type: str = "int",
) -> List[Any]:
    """
    Extract values for one step from either:
    - nested list: [[step0_topk], [step1_topk]]
    - flat list with actual_len * k values
    - flat list with actual_len values
    - flat top-k list for a single step
    """
    parsed = parse_float_list(values) if value_type == "float" else parse_list(values)

    if not parsed:
        return []

    if _is_nested(parsed):
        if step_index < len(parsed):
            return flatten(parsed[step_index])
        return []

    flat = flatten(parsed)

    if actual_len <= 1:
        return flat

    if k and len(flat) >= actual_len * k:
        start = step_index * k
        stop = start + k
        return flat[start:stop]

    if len(flat) == actual_len:
        return [flat[step_index]] if step_index < len(flat) else []

    # Last fallback: use all values for first step only.
    return flat if step_index == 0 else []


def _first_available(row: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    for column in candidates:
        if column in row and not _is_missing(row[column]):
            return row[column]
    return None


def _as_percent(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.1%}"


def _safe_probability_from_logprob(value: Optional[float]) -> Optional[float]:
    """
    Convert a value to probability.

    If value looks like log-probability (negative), exp(value) is used.
    If value already looks like probability [0, 1], it is returned.
    """
    if value is None:
        return None
    try:
        value_float = float(value)
    except Exception:
        return None

    if value_float < 0:
        return math.exp(value_float)

    if 0 <= value_float <= 1:
        return value_float

    return value_float


def explain_prediction(
    input_seq: Any,
    actual_next_seq: Any,
    predicted_seq: Any = None,
    topk_seq: Any = None,
    topk_prob: Any = None,
    transition_reference: Optional[TransitionReference] = None,
    ending_event_ids: Optional[Iterable[int]] = None,
    templates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    sequence_id: Optional[Any] = None,
    top_k: int = 5,
    rare_transition_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Explain one prediction result.

    Baseline anomaly logic:
    1. If actual event is not in top-k candidates -> anomaly.
    2. If top-k is unavailable, top-1 mismatch is treated as a warning/anomaly.
    3. If model output is unavailable, unseen transition / non-ending event rules
       can still flag the sequence as a simple rule-based anomaly.
    """
    input_events = clean_sequence(input_seq)
    actual_events = clean_sequence(actual_next_seq)
    predicted_events = clean_sequence(predicted_seq)

    flat_pred_raw = flatten(parse_list(predicted_seq))
    model_inserted_anomaly_marker = ANOMALY_MARKER in flat_pred_raw

    if ending_event_ids is None and transition_reference is not None:
        ending_event_ids_set = transition_reference.ending_event_ids
    elif ending_event_ids is not None:
        ending_event_ids_set = {int(x) for x in ending_event_ids}
    else:
        ending_event_ids_set = set()

    context = list(input_events)
    step_details: List[Dict[str, Any]] = []
    first_issue_step: Optional[int] = None

    for step_index, actual_event in enumerate(actual_events):
        previous_event = context[-1] if context else None

        topk_events = [
            int(x)
            for x in _extract_step_items(topk_seq, step_index, len(actual_events), k=top_k, value_type="int")
            if x is not None and int(x) != PAD_ID
        ]

        topk_prob_values = [
            _safe_probability_from_logprob(x)
            for x in _extract_step_items(topk_prob, step_index, len(actual_events), k=top_k, value_type="float")
        ]

        top1_event = None
        if step_index < len(predicted_events):
            top1_event = int(predicted_events[step_index])
        elif topk_events:
            top1_event = int(topk_events[0])

        hit_top1 = (top1_event == actual_event) if top1_event is not None else None
        hit_topk = (actual_event in topk_events) if topk_events else None

        actual_rank = None
        actual_probability = None
        if topk_events and actual_event in topk_events:
            actual_rank = topk_events.index(actual_event) + 1
            probability_index = actual_rank - 1
            if probability_index < len(topk_prob_values):
                actual_probability = topk_prob_values[probability_index]

        transition_count = None
        transition_total = None
        transition_probability = None
        unseen_transition = None
        rare_transition = None

        if transition_reference is not None:
            transition_count = transition_reference.transition_count(previous_event, actual_event)
            transition_total = transition_reference.transition_total(previous_event)
            transition_probability = transition_reference.transition_probability(previous_event, actual_event)

            if transition_total > 0:
                unseen_transition = transition_count == 0
                rare_transition = (
                    transition_probability is not None
                    and transition_count > 0
                    and transition_probability < rare_transition_threshold
                )
            else:
                unseen_transition = True
                rare_transition = False

        is_final_actual_event = step_index == len(actual_events) - 1
        is_allowed_ending_event = None
        if ending_event_ids_set and is_final_actual_event:
            is_allowed_ending_event = actual_event in ending_event_ids_set

        has_model_signal = hit_topk is not None or hit_top1 is not None or model_inserted_anomaly_marker
        issue_by_model = (hit_topk is False) or (hit_topk is None and hit_top1 is False)
        issue_by_reference = (unseen_transition is True) or (is_allowed_ending_event is False)

        is_issue = bool(issue_by_model or (not has_model_signal and issue_by_reference))
        if is_issue and first_issue_step is None:
            first_issue_step = step_index + 1

        step_details.append(
            {
                "step": step_index + 1,
                "previous_event_id": previous_event,
                "actual_event_id": actual_event,
                "actual_event": short_event_name(actual_event, templates),
                "top1_event_id": top1_event,
                "top1_event": short_event_name(top1_event, templates) if top1_event is not None else "N/A",
                "topk_event_ids": topk_events,
                "topk_events": [short_event_name(event_id, templates) for event_id in topk_events],
                "hit_top1": hit_top1,
                "hit_topk": hit_topk,
                "actual_rank_in_topk": actual_rank,
                "actual_probability": actual_probability,
                "transition_count": transition_count,
                "transition_total": transition_total,
                "transition_probability": transition_probability,
                "unseen_transition": unseen_transition,
                "rare_transition": rare_transition,
                "is_allowed_ending_event": is_allowed_ending_event,
                "is_issue": is_issue,
            }
        )

        context.append(actual_event)

    # Baseline final status
    anomaly = bool(
        model_inserted_anomaly_marker
        or any(step.get("hit_topk") is False for step in step_details)
        or (
            not any(step.get("hit_topk") is not None or step.get("hit_top1") is not None for step in step_details)
            and any(step.get("is_issue") for step in step_details)
        )
    )

    status = "Anomaly" if anomaly else "Normal"

    if first_issue_step is None and model_inserted_anomaly_marker:
        first_issue_step = 1

    issue_detail = None
    if first_issue_step is not None and 1 <= first_issue_step <= len(step_details):
        issue_detail = step_details[first_issue_step - 1]
    elif step_details:
        issue_detail = step_details[-1]

    reasons = _build_reasons(
        status=status,
        issue_detail=issue_detail,
        model_inserted_anomaly_marker=model_inserted_anomaly_marker,
        templates=templates,
        top_k=top_k,
    )

    severity = _estimate_severity(status, issue_detail, model_inserted_anomaly_marker)

    if issue_detail:
        previous_event_id = issue_detail["previous_event_id"]
        actual_event_id = issue_detail["actual_event_id"]
        top1_event_id = issue_detail["top1_event_id"]
        actual_probability = issue_detail["actual_probability"]
        transition_probability = issue_detail["transition_probability"]
        hit_top1 = issue_detail["hit_top1"]
        hit_topk = issue_detail["hit_topk"]
        topk_event_ids = issue_detail["topk_event_ids"]
    else:
        previous_event_id = None
        actual_event_id = None
        top1_event_id = None
        actual_probability = None
        transition_probability = None
        hit_top1 = None
        hit_topk = None
        topk_event_ids = []

    operator_summary = _build_operator_summary(
        status=status,
        sequence_id=sequence_id,
        severity=severity,
        previous_event_id=previous_event_id,
        actual_event_id=actual_event_id,
        templates=templates,
    )

    operator_action = _build_operator_action(status, previous_event_id, actual_event_id, templates)

    explanation: Dict[str, Any] = {
        "sequence_id": sequence_id,
        "status": status,
        "severity": severity,
        "first_issue_step": first_issue_step,
        "previous_event_id": previous_event_id,
        "previous_event": short_event_name(previous_event_id, templates) if previous_event_id is not None else "N/A",
        "unexpected_event_id": actual_event_id,
        "unexpected_event": short_event_name(actual_event_id, templates) if actual_event_id is not None else "N/A",
        "top1_event_id": top1_event_id,
        "top1_event": short_event_name(top1_event_id, templates) if top1_event_id is not None else "N/A",
        "hit_top1": hit_top1,
        "hit_topk": hit_topk,
        "actual_probability": actual_probability,
        "transition_probability": transition_probability,
        "topk_event_ids": topk_event_ids,
        "operator_summary": operator_summary,
        "operator_action": operator_action,
        "reasons": reasons,
        "reason_1": reasons[0] if len(reasons) > 0 else "",
        "reason_2": reasons[1] if len(reasons) > 1 else "",
        "reason_3": reasons[2] if len(reasons) > 2 else "",
        "step_details": step_details,
    }

    return explanation


def _build_reasons(
    status: str,
    issue_detail: Optional[Mapping[str, Any]],
    model_inserted_anomaly_marker: bool,
    templates: Optional[Mapping[str, Mapping[str, Any]]],
    top_k: int,
) -> List[str]:
    if not issue_detail:
        return ["No issue detail was available."]

    if status == "Normal":
        actual_event_id = issue_detail.get("actual_event_id")
        top1_event_id = issue_detail.get("top1_event_id")
        probability = issue_detail.get("actual_probability")

        reasons = [f"Actual event {actual_event_id} is accepted by the baseline sequence model."]
        if issue_detail.get("hit_top1") is True:
            reasons.append(f"Actual event matched the top-1 prediction: {short_event_name(top1_event_id, templates)}.")
        elif issue_detail.get("hit_topk") is True:
            rank = issue_detail.get("actual_rank_in_topk")
            reasons.append(f"Actual event was still inside top-{top_k} candidates, rank {rank}.")

        if probability is not None:
            reasons.append(f"Model probability for the actual event is {_as_percent(probability)}.")

        return reasons

    actual_event_id = issue_detail.get("actual_event_id")
    previous_event_id = issue_detail.get("previous_event_id")

    reasons: List[str] = []

    if model_inserted_anomaly_marker:
        reasons.append("Model output contains anomaly marker -1.")

    if issue_detail.get("hit_topk") is False:
        reasons.append(
            f"Actual event {actual_event_id} was not inside the model top-{top_k} candidate events."
        )
    elif issue_detail.get("hit_top1") is False:
        reasons.append(
            f"Actual event {actual_event_id} did not match the top-1 predicted event "
            f"{issue_detail.get('top1_event_id')}."
        )

    if issue_detail.get("unseen_transition") is True:
        reasons.append(
            f"Transition {previous_event_id} -> {actual_event_id} was not seen in the normal training reference."
        )
    elif issue_detail.get("rare_transition") is True:
        probability = issue_detail.get("transition_probability")
        reasons.append(
            f"Transition {previous_event_id} -> {actual_event_id} is rare in the training reference "
            f"({_as_percent(probability)} of transitions after event {previous_event_id})."
        )

    if issue_detail.get("is_allowed_ending_event") is False:
        reasons.append(f"Event {actual_event_id} is not registered as a normal ending event.")

    if issue_detail.get("actual_probability") is not None:
        reasons.append(
            f"Model probability for the actual event is {_as_percent(issue_detail.get('actual_probability'))}."
        )
    elif issue_detail.get("hit_topk") is False:
        reasons.append("Actual event probability is below the displayed top-k candidate range.")

    if not reasons:
        reasons.append("Baseline rule marked this sequence for operator review.")

    return reasons[:5]


def _estimate_severity(
    status: str,
    issue_detail: Optional[Mapping[str, Any]],
    model_inserted_anomaly_marker: bool,
) -> str:
    if status == "Normal":
        return "Normal"

    score = 0
    if model_inserted_anomaly_marker:
        score += 1
    if issue_detail:
        if issue_detail.get("hit_topk") is False:
            score += 2
        if issue_detail.get("unseen_transition") is True:
            score += 2
        if issue_detail.get("is_allowed_ending_event") is False:
            score += 1
        probability = issue_detail.get("actual_probability")
        if probability is not None and probability < 0.05:
            score += 1

    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def _build_operator_summary(
    status: str,
    sequence_id: Optional[Any],
    severity: str,
    previous_event_id: Optional[int],
    actual_event_id: Optional[int],
    templates: Optional[Mapping[str, Mapping[str, Any]]],
) -> str:
    seq_text = f"Sequence {sequence_id}" if sequence_id is not None else "This sequence"

    if status == "Normal":
        return f"{seq_text}: NORMAL. Event order is still consistent with the baseline model."

    return (
        f"{seq_text}: ANOMALY ({severity}). "
        f"Unexpected event {short_event_name(actual_event_id, templates)} occurred after "
        f"{short_event_name(previous_event_id, templates)}."
    )


def _build_operator_action(
    status: str,
    previous_event_id: Optional[int],
    actual_event_id: Optional[int],
    templates: Optional[Mapping[str, Mapping[str, Any]]],
) -> str:
    if status == "Normal":
        return "No immediate action from this baseline translator. Continue normal monitoring."

    return (
        "Check the process/log context around "
        f"{short_event_name(previous_event_id, templates)} -> {short_event_name(actual_event_id, templates)}. "
        "Confirm whether this event order is expected for the current operation; if not, escalate according to the alarm response procedure."
    )


def explain_observed_sequence(
    sequence: Any,
    predicted_next: Any = None,
    topk_next: Any = None,
    topk_prob: Any = None,
    transition_reference: Optional[TransitionReference] = None,
    ending_event_ids: Optional[Iterable[int]] = None,
    templates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    sequence_id: Optional[Any] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Convenience function for a full observed sequence.

    The last event is treated as the actual next event to explain.
    Example:
        [1, 2, 3, ..., 19, 20]
        input/history = [1, 2, ..., 19]
        actual_next = [20]
    """
    sequence_clean = clean_sequence(sequence)

    if len(sequence_clean) < 2:
        raise ValueError("Sequence must contain at least two events.")

    return explain_prediction(
        input_seq=sequence_clean[:-1],
        actual_next_seq=[sequence_clean[-1]],
        predicted_seq=predicted_next,
        topk_seq=topk_next,
        topk_prob=topk_prob,
        transition_reference=transition_reference,
        ending_event_ids=ending_event_ids,
        templates=templates,
        sequence_id=sequence_id,
        top_k=top_k,
    )


def explain_prediction_row(
    row: Union[pd.Series, Mapping[str, Any]],
    transition_reference: Optional[TransitionReference] = None,
    templates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Explain one row from a prediction CSV/DataFrame.

    Supported common column names:
    - input_seq / input_keys
    - actual_next_seq / actual_keys
    - next_seq_pred / pred_keys
    - top_5_pred / topk_keys / top_k_pred
    - top_5_prob / topk_probs / top_k_prob
    - seq_id / sequence_id
    """
    row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    input_seq = _first_available(row_dict, ["input_keys", "input_seq", "input", "src", "source_sequence"])
    actual_next_seq = _first_available(row_dict, ["actual_keys", "actual_next_seq", "actual_seq", "target", "tgt"])
    predicted_seq = _first_available(row_dict, ["pred_keys", "next_seq_pred", "pred_seq", "prediction_seq"])
    topk_seq = _first_available(row_dict, ["topk_keys", "top_5_pred", "top_k_pred", "topk_pred", "candidate_logs"])
    topk_prob = _first_available(row_dict, ["topk_probs", "top_5_prob", "top_k_prob", "topk_prob", "candidate_probs"])
    sequence_id = _first_available(row_dict, ["seq_id", "sequence_id", "id"])

    # Fallback: if only one full sequence is available, explain its last event.
    if actual_next_seq is None and input_seq is not None:
        full_sequence = clean_sequence(input_seq)
        if len(full_sequence) >= 2:
            return explain_observed_sequence(
                sequence=full_sequence,
                predicted_next=predicted_seq,
                topk_next=topk_seq,
                topk_prob=topk_prob,
                transition_reference=transition_reference,
                templates=templates,
                sequence_id=sequence_id,
                top_k=top_k,
            )

    return explain_prediction(
        input_seq=input_seq,
        actual_next_seq=actual_next_seq,
        predicted_seq=predicted_seq,
        topk_seq=topk_seq,
        topk_prob=topk_prob,
        transition_reference=transition_reference,
        templates=templates,
        sequence_id=sequence_id,
        top_k=top_k,
    )


def explain_dataframe(
    predictions_df: pd.DataFrame,
    transition_reference: Optional[TransitionReference] = None,
    templates: Optional[Mapping[str, Mapping[str, Any]]] = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """Apply the translator to all prediction rows and return a flat operator report."""
    records: List[Dict[str, Any]] = []

    for _, row in predictions_df.iterrows():
        explanation = explain_prediction_row(
            row,
            transition_reference=transition_reference,
            templates=templates,
            top_k=top_k,
        )

        # Remove nested step details from the flat report.
        flat_record = {k: v for k, v in explanation.items() if k != "step_details" and k != "reasons"}
        flat_record["topk_event_ids"] = str(flat_record.get("topk_event_ids", []))
        records.append(flat_record)

    return pd.DataFrame(records)


def format_operator_message(explanation: Mapping[str, Any]) -> str:
    """Create a readable text block for control room/operator display."""
    lines = [
        str(explanation.get("operator_summary", "")),
        "",
        "Reasons :",
    ]

    reasons = explanation.get("reasons")
    if not reasons:
        reasons = [
            explanation.get("reason_1", ""),
            explanation.get("reason_2", ""),
            explanation.get("reason_3", ""),
        ]

    reasons = [reason for reason in reasons if reason]
    for index, reason in enumerate(reasons, start=1):
        lines.append(f"{index}. {reason}")

    action = explanation.get("operator_action")
    if action:
        lines.extend(["", f"Suggested operator check: {action}"])

    return "\n".join(lines)


def save_operator_report(report_df: pd.DataFrame, output_path: str) -> None:
    """Save the flat operator report to CSV."""
    report_df.to_csv(output_path, index=False)
