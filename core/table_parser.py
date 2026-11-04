"""
表格结构解析模块

目标：把“OCR 后的一串文字”或“PDF 原生表格”还原成可入库、可检索的
“字段-值”结构，并保留页码、表格位置等证据信息。
"""

from typing import Any, Dict, List, Optional

from core.logger import get_logger

logger = get_logger(__name__)


def _box_bounds(bbox: Any) -> Optional[List[float]]:
    """把 RapidOCR 的四点多边形或普通矩形转换为 [x0, y0, x1, y1]。"""
    if not bbox:
        return None

    try:
        if isinstance(bbox[0], (list, tuple)):
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    except (TypeError, ValueError, IndexError):
        return None


def _cell_center(bbox: Any) -> Optional[List[float]]:
    bounds = _box_bounds(bbox)
    if not bounds:
        return None
    x0, y0, x1, y1 = bounds
    return [(x0 + x1) / 2, (y0 + y1) / 2]


def _merge_table_bbox(items: List[Dict]) -> Optional[List[float]]:
    bounds_list = [_box_bounds(item.get("bbox")) for item in items]
    bounds_list = [bounds for bounds in bounds_list if bounds]
    if not bounds_list:
        return None
    return [
        min(bounds[0] for bounds in bounds_list),
        min(bounds[1] for bounds in bounds_list),
        max(bounds[2] for bounds in bounds_list),
        max(bounds[3] for bounds in bounds_list),
    ]


def _group_ocr_lines(items: List[Dict]) -> List[List[Dict]]:
    """按垂直中心把 OCR 文本框聚成视觉行。"""
    valid_items = []
    for item in items:
        center = _cell_center(item.get("bbox"))
        if center:
            valid_items.append({**item, "_center": center})

    valid_items.sort(key=lambda item: item["_center"][1])
    lines: List[List[Dict]] = []

    for item in valid_items:
        center_y = item["_center"][1]
        bounds = _box_bounds(item.get("bbox"))
        height = max(bounds[3] - bounds[1], 1.0) if bounds else 10.0

        if not lines:
            lines.append([item])
            continue

        last_line = lines[-1]
        last_center_y = sum(cell["_center"][1] for cell in last_line) / len(last_line)
        tolerance = max(8.0, height * 0.7)
        if abs(center_y - last_center_y) <= tolerance:
            last_line.append(item)
        else:
            lines.append([item])

    for line in lines:
        line.sort(key=lambda item: item["_center"][0])

    return lines


def _find_table_line_groups(lines: List[List[Dict]]) -> List[List[List[Dict]]]:
    """
    从视觉行中找出连续的“多格行”。

    这是一个启发式版面分析：表格区域的连续行通常会被 OCR 拆成多个
    文本框，而普通段落一行常常只有一个长文本框。
    """
    groups: List[List[List[Dict]]] = []
    current: List[List[Dict]] = []

    for line in lines:
        if len(line) >= 2:
            current.append(line)
            continue
        if len(current) >= 2:
            groups.append(current)
        current = []

    if len(current) >= 2:
        groups.append(current)

    return groups


def _cluster_columns(cells: List[Dict]) -> List[Dict]:
    """把横向位置接近的单元格归为同一列。"""
    centers = sorted(cell["_center"][0] for cell in cells)
    if not centers:
        return []

    widths = []
    for cell in cells:
        bounds = _box_bounds(cell.get("bbox"))
        if bounds:
            widths.append(max(bounds[2] - bounds[0], 1.0))
    tolerance = max(22.0, (sorted(widths)[len(widths) // 2] if widths else 40.0) * 0.55)

    clusters: List[Dict] = []
    for center in centers:
        if not clusters:
            clusters.append({"sum": center, "count": 1})
            continue
        current_center = clusters[-1]["sum"] / clusters[-1]["count"]
        if center - current_center <= tolerance:
            clusters[-1]["sum"] += center
            clusters[-1]["count"] += 1
        else:
            clusters.append({"sum": center, "count": 1})

    columns = []
    for cluster in clusters:
        columns.append({
            "center": cluster["sum"] / cluster["count"],
            "support": cluster["count"],
        })
    return columns


def _cell_text(item: Dict) -> str:
    return str(item.get("text", "")).strip()


def _parse_ocr_table(
    source: str,
    page_num: int,
    table_index: int,
    line_group: List[List[Dict]]
) -> Optional[Dict]:
    """把连续多格 OCR 行解析成结构化表格。"""
    all_cells = [cell for line in line_group for cell in line]
    columns = _cluster_columns(all_cells)

    # 至少要两列，并且至少两列在多行中重复出现，才认为像表格。
    repeated_columns = [column for column in columns if column["support"] >= 2]
    if len(columns) < 2 or len(repeated_columns) < 2:
        return None

    def assign_row(line: List[Dict]) -> List[Optional[Dict]]:
        assigned: List[Optional[Dict]] = [None] * len(columns)
        for cell in sorted(line, key=lambda item: item["_center"][0]):
            column_index = min(
                range(len(columns)),
                key=lambda index: abs(columns[index]["center"] - cell["_center"][0])
            )
            if assigned[column_index] is None:
                assigned[column_index] = cell
            else:
                assigned[column_index] = {
                    **assigned[column_index],
                    "text": _cell_text(assigned[column_index]) + " " + _cell_text(cell),
                }
        return assigned

    assigned_lines = [assign_row(line) for line in line_group]
    header_assigned = assigned_lines[0]
    headers = [
        _cell_text(cell) if cell else f"列{index + 1}"
        for index, cell in enumerate(header_assigned)
    ]

    # 过滤空表头，避免 JSON 里出现多余空键。
    selected_column_indexes = [index for index, header in enumerate(headers) if header]
    if len(selected_column_indexes) < 2:
        return None

    rows = []
    for row_index, assigned in enumerate(assigned_lines[1:], start=1):
        data: Dict[str, str] = {}
        cells = []
        has_value = False
        for column_index in selected_column_indexes:
            cell = assigned[column_index]
            text = _cell_text(cell) if cell else ""
            if text:
                has_value = True
            data[headers[column_index]] = text
            cells.append({
                "column": headers[column_index],
                "text": text,
                "bbox": cell.get("bbox") if cell else None,
                "confidence": cell.get("confidence") if cell else None,
            })
        if has_value:
            rows.append({
                "row_index": row_index,
                "data": data,
                "cells": cells,
            })

    if not rows:
        return None

    named_columns = [index for index in selected_column_indexes if not headers[index].startswith("列")]
    auto_columns = [index for index in selected_column_indexes if headers[index].startswith("列")]
    merge_targets = {}
    for auto_index in auto_columns:
        if not named_columns:
            break
        target_index = min(
            named_columns,
            key=lambda index: abs(columns[index]["center"] - columns[auto_index]["center"]),
        )
        merge_targets[auto_index] = target_index

    if merge_targets:
        for row in rows:
            for auto_index, target_index in merge_targets.items():
                auto_name, target_name = headers[auto_index], headers[target_index]
                auto_text = row["data"].get(auto_name, "")
                target_text = row["data"].get(target_name, "")
                if auto_text:
                    row["data"][target_name] = f"{target_text} {auto_text}".strip()
                row["data"].pop(auto_name, None)

                auto_cell = next(
                    (cell for cell in row["cells"] if cell.get("column") == auto_name),
                    None,
                )
                target_cell = next(
                    (cell for cell in row["cells"] if cell.get("column") == target_name),
                    None,
                )
                if auto_cell and auto_cell.get("text"):
                    if target_cell:
                        target_cell["text"] = f"{target_cell.get('text', '')} {auto_cell['text']}".strip()
                        target_cell["bbox"] = target_cell.get("bbox") or auto_cell.get("bbox")
                        target_cell["confidence"] = target_cell.get("confidence") or auto_cell.get("confidence")
                    else:
                        row["cells"].append(auto_cell)
                if target_cell:
                    target_cell["column"] = target_name
                row["cells"] = [cell for cell in row["cells"] if cell.get("column") != auto_name]

        selected_column_indexes = [index for index in selected_column_indexes if index not in merge_targets]

    # 去掉 OCR 对齐产生的空列；只有自动命名且整列无值时才移除。
    removable_indexes = [
        index
        for index, header in enumerate(headers)
        if header.startswith("列")
        and all(not row["data"].get(header, "") for row in rows)
    ]
    if removable_indexes:
        removed_names = {headers[index] for index in removable_indexes}
        for row in rows:
            row["cells"] = [
                cell for cell in row["cells"]
                if cell.get("column") not in removed_names
            ]
            row["data"] = {
                key: value for key, value in row["data"].items()
                if key not in removed_names
            }
        selected_column_indexes = [
            index for index in selected_column_indexes
            if index not in removable_indexes
        ]

    return {
        "source": source,
        "page": page_num,
        "table_index": table_index,
        "parser": "ocr_layout",
        "headers": [headers[index] for index in selected_column_indexes],
        "rows": rows,
        "bbox": _merge_table_bbox(all_cells),
        "row_count": len(rows),
    }


def _parse_pdf_table(
    source: str,
    page_num: int,
    table_index: int,
    raw_rows: List[List[Any]]
) -> Optional[Dict]:
    """解析 pdfplumber 抽出的原生表格。"""
    rows = []
    for raw_row in raw_rows:
        row = ["" if value is None else str(value).strip() for value in raw_row]
        if any(row):
            rows.append(row)

    if len(rows) < 2:
        return None

    width = max(len(row) for row in rows)
    headers = rows[0] + [""] * (width - len(rows[0]))
    for index, header in enumerate(headers):
        if not header:
            headers[index] = f"列{index + 1}"

    parsed_rows = []
    for row_index, raw_row in enumerate(rows[1:], start=1):
        values = raw_row + [""] * (width - len(raw_row))
        data = {}
        cells = []
        for header, value in zip(headers, values):
            data[header] = value
            cells.append({
                "column": header,
                "text": value,
                "bbox": None,
                "confidence": None,
            })
        if any(data.values()):
            parsed_rows.append({
                "row_index": row_index,
                "data": data,
                "cells": cells,
            })

    if not parsed_rows:
        return None

    return {
        "source": source,
        "page": page_num,
        "table_index": table_index,
        "parser": "pdfplumber",
        "headers": headers,
        "rows": parsed_rows,
        "bbox": None,
        "row_count": len(parsed_rows),
    }


def parse_tables(pages: List[Dict]) -> List[Dict]:
    """
    从解析后的页面中抽取结构化表格。

    Args:
        pages: DocumentParser 返回的页面列表。扫描页包含 ocr_items，
               原生 PDF 页可能包含 pdf_tables。

    Returns:
        统一结构的表格列表，每个表格包含 headers、rows、page、bbox 等证据。
    """
    tables: List[Dict] = []

    try:
        for page in pages:
            source = page.get("source", "unknown")
            page_num = int(page.get("page", 1))

            for raw_table in page.get("pdf_tables", []) or []:
                table = _parse_pdf_table(
                    source=source,
                    page_num=page_num,
                    table_index=len(tables),
                    raw_rows=raw_table,
                )
                if table:
                    tables.append(table)

            ocr_items = page.get("ocr_items", []) or []
            if ocr_items:
                lines = _group_ocr_lines(ocr_items)
                line_groups = _find_table_line_groups(lines)
                for line_group in line_groups:
                    table = _parse_ocr_table(
                        source=source,
                        page_num=page_num,
                        table_index=len(tables),
                        line_group=line_group,
                    )
                    if table:
                        tables.append(table)

        for table_index, table in enumerate(tables):
            table["table_index"] = table_index

        logger.info(
            "表格结构解析完成 | 页数=%d | 表格数=%d",
            len(pages), len(tables)
        )
        return tables
    except Exception:
        logger.exception("表格结构解析失败")
        return []
