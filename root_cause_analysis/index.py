from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── metryki i etykiety ────────────────────────────────────────────────
METRIC_LABELS: Dict[str, str] = {
	"LatencyMs": "latencja",
	"CpuUsage": "CPU",
	"MemoryUsageMb": "RAM",
	"DiskQueueLength": "dysk I/O",
	"NetworkErrors": "sieć",
	"LocalQps": "QPS",
	"Retries": "retry",
	"RequestSizeBytes": "rozmiar żądania",
	"ResponseSizeBytes": "rozmiar odpowiedzi",
}

FLAG_LABELS: Dict[str, str] = {
	"IsSpikeByLatency": "spike latencji",
	"IsSpikeByCpu": "spike CPU",
	"IsSpikeByQps": "spike QPS",
	"SpikeWindow": "okno spike'a",
	"FailureWindow": "okno awarii",
	"TrendWindow": "okno trendu",
}

RECOMMENDATIONS: Dict[str, List[str]] = {
	"latencja": [
		"Sprawdź p95/p99 opóźnienia per endpoint i zależności downstream",
		"Zweryfikuj timeouty, connection-pool i kolejki requestów",
		"Monitoruj: LatencyMs, DiskQueueLength, NetworkErrors",
	],
	"CPU": [
		"Sprawdź hot-path'y, GC-pressure i wykorzystanie CPU per endpoint",
		"Rozważ skalowanie poziome lub optymalizację compute-intensive ścieżek",
		"Monitoruj: CpuUsage, LocalQps, MemoryUsageMb",
	],
	"RAM": [
		"Sprawdź wycieki pamięci, presję GC i alokacje dużych obiektów",
		"Zweryfikuj limity pamięci kontenerów / JVM heap",
		"Monitoruj: MemoryUsageMb, CpuUsage (GC), ResponseSizeBytes",
	],
	"dysk I/O": [
		"Sprawdź IO-wait, opóźnienia storage/DB i głębokość kolejki",
		"Ogranicz synchroniczne zapisy, logowanie i checkpointy",
		"Monitoruj: DiskQueueLength, LatencyMs, IOPS storage'u",
	],
	"sieć": [
		"Sprawdź timeouty, retry-storm i błędy połączeń",
		"Zweryfikuj obciążenie downstream, limity połączeń i DNS",
		"Monitoruj: NetworkErrors, Retries, LatencyMs",
	],
	"QPS": [
		"Zweryfikuj rate-limiting i auto-scaling pod kątem nagłych skoków ruchu",
		"Sprawdź czy wzrost QPS nie powoduje kaskadowego przeciążenia",
		"Monitoruj: LocalQps, CpuUsage, LatencyMs, EventCode (429/503)",
	],
	"retry": [
		"Wykryto retry-storm — sprawdź exponential backoff i circuit-breaker",
		"Zidentyfikuj serwis źródłowy błędów powodujących retry",
		"Monitoruj: Retries, NetworkErrors, EventCode (5xx), LatencyMs",
	],
	"rozmiar żądania": [
		"Nietypowy rozmiar payloadu — sprawdź walidację wejścia i limity",
		"Rozważ kompresję / paginację dużych requestów",
		"Monitoruj: RequestSizeBytes, LatencyMs, MemoryUsageMb",
	],
	"rozmiar odpowiedzi": [
		"Nietypowy rozmiar odpowiedzi — sprawdź serializację i limity",
		"Rozważ lazy-loading / paginację dużych odpowiedzi",
		"Monitoruj: ResponseSizeBytes, LatencyMs, MemoryUsageMb",
	],
}


# ── helpers ───────────────────────────────────────────────────────────

def _compute_global_stats(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
	"""Mediana, Q25, Q75 dla dostępnych metryk (liczone raz na całym df)."""
	cols = [c for c in METRIC_LABELS if c in df.columns]
	numeric = df[cols].apply(pd.to_numeric, errors="coerce")
	med = numeric.median()
	q25 = numeric.quantile(0.25)
	q75 = numeric.quantile(0.75)
	return med, q25, q75


def _iqr_deviations(row: pd.Series, med: pd.Series, iqr: pd.Series) -> List[Tuple[str, float]]:
	"""Zwraca listę (etykieta, z-score IQR) posortowaną malejąco."""
	results = []
	for col in iqr.index:
		if col not in row.index or pd.isna(row[col]) or pd.isna(iqr[col]) or iqr[col] == 0:
			continue
		z = abs((row[col] - med[col]) / iqr[col])
		if z > 1.0:
			results.append((METRIC_LABELS.get(col, col), float(z)))
	results.sort(key=lambda x: -x[1])
	return results


def _active_flags(row: pd.Series) -> List[str]:
	"""Zwraca aktywne flagi boolean z wiersza."""
	active = []
	for flag, label in FLAG_LABELS.items():
		if flag in row.index and row[flag] is True:
			active.append(label)
	return active


def _diagnose_events(
	events: pd.DataFrame,
	med: pd.Series,
	iqr: pd.Series,
) -> Dict[str, Dict]:
	"""Diagnoza per system: dominujące odchylenia i aktywne flagi."""
	diagnosis: Dict[str, Dict] = {}
	for system, grp in events.groupby("SourceSystem"):
		all_devs: Dict[str, List[float]] = {}
		all_flags: Dict[str, int] = {}

		for _, row in grp.iterrows():
			for label, z in _iqr_deviations(row, med, iqr):
				all_devs.setdefault(label, []).append(z)
			for flag in _active_flags(row):
				all_flags[flag] = all_flags.get(flag, 0) + 1

		top_reasons = sorted(
			((label, np.mean(zs), len(zs)) for label, zs in all_devs.items()),
			key=lambda x: -x[1],
		)[:3]

		diagnosis[system] = {
			"events": len(grp),
			"top_reasons": [(r[0], round(r[1], 2), r[2]) for r in top_reasons],
			"flags": all_flags,
		}
	return diagnosis


def _assess_pre_failure(
	pre_events: pd.DataFrame,
	med: pd.Series,
	iqr: pd.Series,
) -> List[str]:
	"""Ocena eventów sprzed awarii — co mogło wpłynąć."""
	observations: List[str] = []
	if pre_events.empty:
		return ["Brak eventów poprzedzających awarię"]

	for col, label in METRIC_LABELS.items():
		if col not in pre_events.columns or col not in med.index:
			continue
		vals = pd.to_numeric(pre_events[col], errors="coerce").dropna()
		if vals.empty:
			continue
		avg = vals.mean()
		iqr_val = iqr.get(col, 0)
		if pd.notna(iqr_val) and iqr_val > 0 and abs(avg - med[col]) / iqr_val > 1.5:
			direction = "powyżej" if avg > med[col] else "poniżej"
			observations.append(
				f"{label} ({col}) = avg {avg:.1f} — znacznie {direction} mediany ({med[col]:.1f})"
			)

	if len(pre_events) >= 2:
		for col in ["LatencyMs", "CpuUsage", "DiskQueueLength", "NetworkErrors", "Retries"]:
			if col not in pre_events.columns:
				continue
			vals = pd.to_numeric(pre_events[col], errors="coerce").dropna()
			if len(vals) >= 2:
				trend = vals.iloc[-1] - vals.iloc[0]
				med_val = med.get(col, 0)
				if trend > 0 and med_val > 0 and abs(trend) > med_val * 0.3:
					observations.append(
						f"Narastający trend {METRIC_LABELS.get(col, col)}: "
						f"{vals.iloc[0]:.0f} → {vals.iloc[-1]:.0f} (+{trend:.0f})"
					)

	for flag, label in FLAG_LABELS.items():
		if flag in pre_events.columns:
			cnt = (pre_events[flag] == True).sum()
			if cnt > 0:
				observations.append(
					f"Flaga '{label}' aktywna w {cnt}/{len(pre_events)} eventów przed awarią"
				)

	if "Retries" in pre_events.columns:
		retries = pd.to_numeric(pre_events["Retries"], errors="coerce").sum()
		if retries > 0:
			observations.append(f"Retry'e przed awarią: łącznie {int(retries)}")

	if not observations:
		observations.append("Metryki przed awarią w normie — problem mógł wystąpić nagle")

	return observations


def _build_recommendations(diagnosis: Dict[str, Dict]) -> List[str]:
	"""Rekomendacje na podstawie diagnozy wszystkich systemów."""
	seen_reasons: Dict[str, float] = {}
	for sys_diag in diagnosis.values():
		for label, z, _ in sys_diag["top_reasons"]:
			if label not in seen_reasons or z > seen_reasons[label]:
				seen_reasons[label] = z

	top_reasons = sorted(seen_reasons.items(), key=lambda x: -x[1])[:3]
	recs: List[str] = []
	for label, _ in top_reasons:
		if label in RECOMMENDATIONS:
			for r in RECOMMENDATIONS[label]:
				if r not in recs:
					recs.append(r)

	if not recs:
		recs.append("Przeanalizuj kombinacje metryk i korelacje z ruchem w danym oknie czasowym")

	return recs


def _build_watch_list(diagnosis: Dict[str, Dict]) -> List[str]:
	"""Co dodatkowo obserwować — metryki i flagi które odstawały."""
	watch = set()
	for sys_diag in diagnosis.values():
		for label, z, _ in sys_diag["top_reasons"]:
			if z > 2.0:
				for col, lbl in METRIC_LABELS.items():
					if lbl == label:
						watch.add(col)
		for flag in sys_diag["flags"]:
			for f_col, f_lbl in FLAG_LABELS.items():
				if f_lbl == flag:
					watch.add(f_col)
	return sorted(watch) if watch else ["LatencyMs", "CpuUsage", "EventCode"]


# ── główna funkcja ────────────────────────────────────────────────────

def root_cause_analysis(
	df: pd.DataFrame,
	correlation_id: Optional[str] = None,
	pre_failure_window: int = 5,
) -> Dict[str, object]:
	"""
	Rekonstrukcja historii i RCA dla wskazanego CorrelationId.

	Jeśli correlation_id jest None, wybiera CorrelationId z IsAnomaly == 1.

	Zwraca słownik z pełną diagnozą:
	  events, path, scenario, first_problem_system, diagnosis,
	  victims, pre_failure_events, pre_failure_assessment,
	  recommendations, watch_list
	"""
	work_df = df

	# ── 1. wybór CorrelationId ─────────────────────────────────────
	if correlation_id is None:
		if "IsAnomaly" in work_df.columns:
			candidate = work_df[work_df["IsAnomaly"] == 1]
		else:
			candidate = work_df
		if candidate.empty:
			candidate = work_df
		if "CorrelationId" in candidate.columns:
			correlation_id = (
				candidate["CorrelationId"].iloc[7]
				if len(candidate) > 7
				else candidate["CorrelationId"].iloc[0]
			)

	# ── 2. wyciągnięcie eventów ────────────────────────────────────
	subset = work_df[work_df["CorrelationId"] == correlation_id].copy()
	if subset.empty:
		print(f"Brak eventów dla CorrelationId={correlation_id}")
		return {}

	# ── 3. sortowanie ──────────────────────────────────────────────
	if "Timestamp" in subset.columns:
		subset["Timestamp"] = pd.to_datetime(subset["Timestamp"], errors="coerce")
	sort_cols = [c for c in ["Timestamp", "ScenarioStepIndex"] if c in subset.columns]
	if sort_cols:
		subset = subset.sort_values(sort_cols, ascending=True)

	# ── 4. kontekst: scenariusz, rola, region ──────────────────────
	scenario = subset["Scenario"].iloc[0] if "Scenario" in subset.columns else "nieznany"
	regions = subset["Region"].unique().tolist() if "Region" in subset.columns else []
	nodes = subset["ClusterNode"].unique().tolist() if "ClusterNode" in subset.columns else []

	# ── 5. ścieżka przejścia ──────────────────────────────────────
	path: List[str] = []
	last_sys = None
	for sys in subset["SourceSystem"].tolist():
		if sys != last_sys:
			path.append(sys)
			last_sys = sys

	# ── 6. globalne statystyki (mediana / IQR) ─────────────────────
	med, q25, q75 = _compute_global_stats(work_df)
	iqr = (q75 - q25).replace(0, np.nan)

	# ── 7. identyfikacja problemów (wektorowo) ────────────────────
	problem_mask = pd.Series(False, index=subset.index)

	if "IsAnomaly" in subset.columns:
		problem_mask |= subset["IsAnomaly"] == 1
	if "IF_Anomaly" in subset.columns:
		problem_mask |= subset["IF_Anomaly"] == 1
	if "EventCode" in subset.columns:
		codes = pd.to_numeric(subset["EventCode"], errors="coerce")
		problem_mask |= codes >= 500
	for flag in ["FailureWindow", "SpikeWindow"]:
		if flag in subset.columns:
			problem_mask |= subset[flag] == True

	for col in ["LatencyMs", "CpuUsage", "LocalQps"]:
		if col in subset.columns and col in q75.index:
			p95 = float(pd.to_numeric(work_df[col], errors="coerce").quantile(0.95))
			problem_mask |= pd.to_numeric(subset[col], errors="coerce") >= p95

	# ── 8. pierwszy system z problemem ─────────────────────────────
	first_problem_idx = None
	first_problem_system = None
	first_problem_step = None
	if problem_mask.any():
		first_problem_idx = problem_mask.idxmax()
		first_problem_system = subset.loc[first_problem_idx, "SourceSystem"]
		if "ScenarioStepIndex" in subset.columns:
			first_problem_step = int(subset.loc[first_problem_idx, "ScenarioStepIndex"])

	# ── 9. ofiary ──────────────────────────────────────────────────
	victims: List[str] = []
	propagation_evidence = False
	if first_problem_system is not None:
		if first_problem_system in path:
			root_pos = path.index(first_problem_system)
			victims = path[root_pos + 1 :]
			propagation_evidence = len(victims) > 0
		else:
			victims = []

		# słabszy sygnał propagacji: odchylenia metryk w eventach po awarii
		if not propagation_evidence and first_problem_idx is not None:
			after = subset.loc[subset.index > first_problem_idx]
			if not after.empty:
				weak = _diagnose_events(after, med, iqr)
				propagation_evidence = any(d["top_reasons"] for d in weak.values())

	# ── 10. eventy przed awarią ────────────────────────────────────
	if first_problem_idx is not None:
		pre_events = subset.loc[subset.index < first_problem_idx].tail(pre_failure_window)
	else:
		pre_events = subset.tail(pre_failure_window)

	# ── 11. diagnoza per system ────────────────────────────────────
	problem_events = subset.loc[problem_mask]
	diagnosis = _diagnose_events(problem_events, med, iqr)

	# ── 12. ocena pre-failure ──────────────────────────────────────
	pre_failure_assessment = _assess_pre_failure(pre_events, med, iqr)

	# ── 13. rekomendacje ───────────────────────────────────────────
	recommendations = _build_recommendations(diagnosis)
	watch_list = _build_watch_list(diagnosis)

	# ── 14. wydruk podsumowania ────────────────────────────────────
	_print_summary(
		correlation_id, scenario, regions, nodes, path,
		first_problem_system, first_problem_step,
		diagnosis, victims, propagation_evidence, pre_events, pre_failure_assessment,
		recommendations, watch_list, subset,
	)

	return {
		"events": subset,
		"path": path,
		"scenario": scenario,
		"first_problem_system": first_problem_system,
		"diagnosis": diagnosis,
		"victims": victims,
		"pre_failure_events": pre_events,
		"propagation_evidence": propagation_evidence,
		"pre_failure_assessment": pre_failure_assessment,
		"recommendations": recommendations,
		"watch_list": watch_list,
	}


# ── wydruk ────────────────────────────────────────────────────────────

def _print_summary(
	correlation_id, scenario, regions, nodes, path,
	first_problem_system, first_problem_step,
	diagnosis, victims, propagation_evidence, pre_events, pre_failure_assessment,
	recommendations, watch_list, subset,
):
	sep = "=" * 65

	print(f"\n{sep}")
	print("  ROOT CAUSE ANALYSIS")
	print(sep)

	print(f"\nCorrelationId : {correlation_id}")
	print(f"Scenariusz    : {scenario}")
	print(f"Regiony       : {', '.join(str(r) for r in regions) if regions else '—'}")
	print(f"Węzły         : {', '.join(str(n) for n in nodes) if nodes else '—'}")
	print(f"Liczba eventów: {len(subset)}")

	# ścieżka
	print(f"\n── Ścieżka przejścia ──")
	print("  " + " → ".join(path))

	# root cause
	print(f"\n── Pierwsze źródło problemu ──")
	if first_problem_system:
		step_info = f" (krok {first_problem_step})" if first_problem_step is not None else ""
		print(f"  System: {first_problem_system}{step_info}")
		if first_problem_system in diagnosis:
			d = diagnosis[first_problem_system]
			if d["top_reasons"]:
				reasons_str = ", ".join(
					f"{r[0]} (z={r[1]:.1f}, n={r[2]})" for r in d["top_reasons"]
				)
				print(f"  Przyczyny: {reasons_str}")
			if d["flags"]:
				flags_str = ", ".join(f"{f} ({c}x)" for f, c in d["flags"].items())
				print(f"  Flagi: {flags_str}")
	else:
		print("  Nie wykryto problemu w tej korelacji")

	# diagnoza per system
	if diagnosis:
		print(f"\n── Diagnoza per system ──")
		for sys_name, d in diagnosis.items():
			marker = " ← ROOT CAUSE" if sys_name == first_problem_system else ""
			marker = marker if marker else (" ← OFIARA" if sys_name in victims else "")
			print(f"  [{sys_name}]{marker}  ({d['events']} eventów z problemem)")
			if d["top_reasons"]:
				for label, z, cnt in d["top_reasons"]:
					bar = "█" * min(int(z), 20)
					print(f"    • {label:20s} z={z:5.1f} ({cnt}x) {bar}")
			if d["flags"]:
				for flag, cnt in d["flags"].items():
					print(f"    ⚑ {flag}: {cnt}x")

	# ofiary
	print(f"\n── Ofiary przeciążenia ──")
	if victims:
		for v in victims:
			if v in diagnosis and diagnosis[v]["top_reasons"]:
				symptom = diagnosis[v]["top_reasons"][0][0]
				print(f"  • {v} — główny symptom: {symptom}")
			else:
				print(f"  • {v}")
	else:
		msg = "Brak dowodów propagacji" if not propagation_evidence else "Możliwa propagacja — brak twardych ofiar"
		print(f"  {msg}")

	# pre-failure
	print(f"\n── Co działo się przed awarią ──")
	if not pre_events.empty:
		show_cols = ["SourceSystem", "EventCode", "LatencyMs", "CpuUsage",
					 "DiskQueueLength", "NetworkErrors", "Retries", "Description"]
		show_cols = [c for c in show_cols if c in pre_events.columns]
		print(pre_events[show_cols].to_string(index=False))

	print(f"\n── Ocena pre-failure ──")
	for obs in pre_failure_assessment:
		print(f"  • {obs}")

	# rekomendacje
	print(f"\n── Rekomendacje ──")
	for i, rec in enumerate(recommendations, 1):
		print(f"  {i}. {rec}")

	# watch list
	print(f"\n── Co monitorować ──")
	print(f"  Kolumny: {', '.join(watch_list)}")

	print(f"\n{sep}\n")

