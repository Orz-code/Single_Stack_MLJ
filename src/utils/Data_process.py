"""Standardized experiment data processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gzip
import math
import pickle
import numpy as np
import pandas as pd

from utils.keys import Cols


SUPPORTED_SUFFIXES = {".csv", ".txt", ".xlsx", ".xls"}

DEFAULT_COLUMN_MAPPING = {
	"时间": Cols.date_time,
	"1#碱液流量": Cols.lye_flow,
	"1#碱液进口温度": Cols.lye_temp,
	"1#氧侧出口温度": Cols.temp_O,
	"1#氢侧出口温度": Cols.temp_H,
	"1#电流显示": Cols.current,
	"1#电压显示": Cols.voltage,
	"氧中氢": Cols.HTO,
	"氢中氧": Cols.OTH,
	"氧气累积量": Cols.O_production_accumulated,
}


@dataclass
class PipelineConfig:
	input_paths: Sequence[str]
	column_mapping: Dict[str, str]
	default_freq: str = "10S"
	num_cells: Optional[float] = None
	electrode_diameter_m: Optional[float] = None
	pressure_columns: Tuple[str, str] = ("氧分离器压力", "氢分离器压力")
	output_name: str = "processed gzip"


def discover_input_files(paths: Sequence[str]) -> List[Path]:
	files: List[Path] = []
	for raw in paths:
		path = Path(raw)
		if path.is_dir():
			for candidate in path.rglob("*"):
				if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
					files.append(candidate)
		else:
			expanded = list(Path().glob(raw)) if any(ch in raw for ch in "*?[]") else [path]
			for candidate in expanded:
				if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
					files.append(candidate)
	return sorted(set(files))


def load_file(file_path: Path) -> pd.DataFrame:
	suffix = file_path.suffix.lower()
	if suffix in {".xlsx", ".xls"}:
		return pd.read_excel(file_path)
	return pd.read_csv(file_path, compression="infer")


def standardize_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
	if not mapping:
		return df
	lowered = {key.lower(): value for key, value in mapping.items()}
	rename_map = {}
	for col in df.columns:
		target = lowered.get(str(col).lower())
		if target:
			rename_map[col] = target
	return df.rename(columns=rename_map)


def drop_invalid_datetime_rows(df: pd.DataFrame) -> pd.DataFrame:
	if Cols.date_time not in df.columns:
		raise ValueError(f"Missing required column: {Cols.date_time}")
	date_series = pd.to_datetime(df[Cols.date_time], errors="coerce")
	df = df.assign(**{Cols.date_time: date_series})
	return df.dropna(subset=[Cols.date_time])


def compute_derived_columns(
	df: pd.DataFrame,
	num_cells: Optional[float],
	electrode_diameter_m: Optional[float],
	pressure_columns: Tuple[str, str],
) -> pd.DataFrame:
	df = df.copy()

	if num_cells and Cols.voltage in df.columns:
		df[Cols.cell_voltage] = pd.to_numeric(df[Cols.voltage], errors="coerce") / num_cells

	if electrode_diameter_m and Cols.current in df.columns:
		area = math.pi * (electrode_diameter_m / 2) ** 2
		df[Cols.current_density] = pd.to_numeric(df[Cols.current], errors="coerce") / area

	if Cols.temp_H in df.columns and Cols.temp_O in df.columns:
		df[Cols.temp_out] = (
			pd.to_numeric(df[Cols.temp_H], errors="coerce")
			+ pd.to_numeric(df[Cols.temp_O], errors="coerce")
		) / 2

	pressure_oxygen, pressure_hydrogen = pressure_columns
	if pressure_oxygen in df.columns and pressure_hydrogen in df.columns:
		df[Cols.pressure] = (
			pd.to_numeric(df[pressure_oxygen], errors="coerce")
			+ pd.to_numeric(df[pressure_hydrogen], errors="coerce")
		)

	return df


def fill_negative_9999(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	for col in df.columns:
		if col == Cols.date_time:
			continue
		numeric = pd.to_numeric(df[col], errors="coerce")
		mask = numeric == -9999
		if not mask.any():
			continue
		prev_vals = numeric.shift(1)
		next_vals = numeric.shift(-1)
		avg_vals = pd.concat([prev_vals, next_vals], axis=1).mean(axis=1, skipna=True)
		numeric = numeric.mask(mask, avg_vals)
		df[col] = numeric
	return df


def infer_frequency(df: pd.DataFrame, default_freq: str) -> str:
	if Cols.date_time not in df.columns or len(df) < 2:
		return default_freq
	sorted_times = pd.to_datetime(df[Cols.date_time]).sort_values()
	diffs = sorted_times.diff().dropna()
	if diffs.empty:
		return default_freq
	median_delta = diffs.median()
	return pd.tseries.frequencies.to_offset(median_delta).freqstr


def merge_and_fill(dfs: Sequence[pd.DataFrame], freq: str) -> pd.DataFrame:
	combined = pd.concat(dfs, ignore_index=True)
	combined = combined.sort_values(Cols.date_time)
	combined = combined.drop_duplicates(subset=[Cols.date_time])
	combined = combined.set_index(Cols.date_time).sort_index()
	full_index = pd.date_range(start=combined.index.min(), end=combined.index.max(), freq=freq)
	combined = combined.reindex(full_index)
	combined.index.name = Cols.date_time
	combined = combined.fillna(0).reset_index()
	return combined


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with gzip.open(output_path, "wb") as handle:
		pickle.dump(df, handle)


def build_output_path(file_path: Path, output_name: str) -> Path:
	parts = list(file_path.parts)
	if "raw" in parts:
		parts[parts.index("raw")] = "processed"
		target_dir = Path(*parts[:-1])
	else:
		target_dir = file_path.parent / "processed"
	return target_dir / output_name


def process_files(config: PipelineConfig) -> None:
	input_files = discover_input_files(config.input_paths)
	if not input_files:
		raise FileNotFoundError("No input files found for the given paths.")

	expected_freq: Optional[str] = None
	all_dfs: List[pd.DataFrame] = []

	for file_path in input_files:
		df = load_file(file_path)
		df = standardize_columns(df, config.column_mapping)
		df = drop_invalid_datetime_rows(df)
		df = compute_derived_columns(
			df,
			num_cells=config.num_cells,
			electrode_diameter_m=config.electrode_diameter_m,
			pressure_columns=config.pressure_columns,
		)
		df = fill_negative_9999(df)
		df = df.sort_values(Cols.date_time)

		freq = infer_frequency(df, config.default_freq)
		if expected_freq is None:
			expected_freq = freq
		elif freq != expected_freq:
			raise ValueError(
				f"Inconsistent time intervals: {file_path} has {freq}, expected {expected_freq}."
			)

		all_dfs.append(df)

	if not all_dfs:
		raise FileNotFoundError("No valid data frames to merge.")

	merged = merge_and_fill(all_dfs, expected_freq)
	output_path = build_output_path(input_files[0], config.output_name)
	save_dataframe(merged, output_path)


def run_pipeline(
	input_paths: Sequence[str],
	column_mapping: Optional[Dict[str, str]] = None,
	default_freq: str = "10S",
	num_cells: Optional[float] = None,
	electrode_diameter_m: Optional[float] = None,
	pressure_columns: Tuple[str, str] = ("氧分离器压力", "氢分离器压力"),
	output_name: str = "processed gzip",
) -> None:
	config = PipelineConfig(
		input_paths=input_paths,
		column_mapping=column_mapping or DEFAULT_COLUMN_MAPPING,
		default_freq=default_freq,
		num_cells=num_cells,
		electrode_diameter_m=electrode_diameter_m,
		pressure_columns=pressure_columns,
		output_name=output_name,
	)
	process_files(config)

