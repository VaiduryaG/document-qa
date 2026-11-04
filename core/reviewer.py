"""结构化表格的字段抽取和基础审查规则。"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


MONEY_KEYWORDS = ("金额", "费用", "价格", "单价", "总价", "授权费", "预算")
DATE_KEYWORDS = ("日期", "时间")
ORG_KEYWORDS = ("部门", "单位", "负责")
NAME_KEYWORDS = ("产品名称", "项目名称", "设备名称", "文件名称", "名称")
IDENTIFIER_KEYWORDS = ("文号", "字号", "编号", "单号")


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _classify_header(header: str) -> str:
    if any(keyword in header for keyword in MONEY_KEYWORDS):
        return "money"
    if any(keyword in header for keyword in DATE_KEYWORDS):
        return "date"
    if any(keyword in header for keyword in ORG_KEYWORDS):
        return "organization"
    if any(keyword in header for keyword in NAME_KEYWORDS):
        return "name"
    if any(keyword in header for keyword in IDENTIFIER_KEYWORDS):
        return "identifier"
    return ""


def _parse_amount(value: str) -> float | None:
    text = _clean_text(value).replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(亿元|万元|千元|元)?", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2) or ""
    if unit == "亿元":
        number *= 100_000_000
    elif unit == "万元":
        number *= 10_000
    elif unit == "千元":
        number *= 1_000
    return number


def _parse_quantity(value: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(台|个|套|件|项|人)?", _clean_text(value))
    if not match:
        return None
    return float(match.group(1))


def _parse_date(value: str) -> str | None:
    text = _clean_text(value)
    match = re.search(r"(\d{4})[-/.年]\s*(\d{1,2})[-/.月]\s*(\d{1,2})日?", text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _is_document_number_header(header: str) -> bool:
    return any(keyword in header for keyword in IDENTIFIER_KEYWORDS)


def _field_evidence(table: Dict[str, Any], row: Dict[str, Any], column: str) -> Dict[str, Any]:
    cell = next(
        (item for item in row.get("cells", []) if item.get("column") == column),
        {},
    )
    return {
        "source": table.get("source", "unknown"),
        "column": column,
        "bbox": cell.get("bbox"),
        "confidence": cell.get("confidence"),
    }


def extract_fields(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从结构化表格行中抽取具有审查价值的字段。"""
    fields = []
    for table in tables:
        page = int(table.get("page", 1))
        table_index = int(table.get("table_index", 0))

        for row in table.get("rows", []):
            row_index = int(row.get("row_index", 0))
            for column, value in row.get("data", {}).items():
                text = _clean_text(value)
                field_type = _classify_header(column)
                if not field_type or not text:
                    continue

                fields.append({
                    "source": table.get("source", "unknown"),
                    "page": page,
                    "table_index": table_index,
                    "row_index": row_index,
                    "column": column,
                    "field_type": field_type,
                    "field_name": column,
                    "value": text,
                    "evidence": _field_evidence(table, row, column),
                })

    return fields


def _issue(
    table: Dict[str, Any],
    row: Dict[str, Any],
    rule_id: str,
    severity: str,
    message: str,
    column: str = "",
    evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "source": table.get("source", "unknown"),
        "page": int(table.get("page", 1)),
        "table_index": int(table.get("table_index", 0)),
        "row_index": int(row.get("row_index", -1)) if row else -1,
        "column": column,
        "rule_id": rule_id,
        "severity": severity,
        "message": message,
        "evidence": evidence or {},
    }


def review_tables(
    tables: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """抽取关键字段并执行基础审查规则。"""
    fields = extract_fields(tables)
    issues: List[Dict[str, Any]] = []

    for table in tables:
        rows = table.get("rows", [])
        source = table.get("source", "unknown")
        page = int(table.get("page", 1))
        table_index = int(table.get("table_index", 0))

        for row in rows:
            data = row.get("data", {})

            for column, value in data.items():
                if _classify_header(column) and not _clean_text(value):
                    issues.append(_issue(
                        table,
                        row,
                        "missing_field",
                        "error",
                        f"关键字段「{column}」为空",
                        column,
                    ))

            start_columns = [
                column for column in data
                if any(keyword in column for keyword in ("生效日期", "开始日期", "起始日期", "生效时间", "开始时间"))
            ]
            end_columns = [
                column for column in data
                if any(keyword in column for keyword in ("废止日期", "结束日期", "截止日期", "失效日期", "结束时间"))
            ]
            if start_columns and end_columns:
                start_date = _parse_date(data.get(start_columns[0], ""))
                end_date = _parse_date(data.get(end_columns[0], ""))
                if start_date and end_date and start_date > end_date:
                    issues.append(_issue(
                        table,
                        row,
                        "invalid_date_range",
                        "error",
                        f"开始/生效日期 {start_date} 晚于结束/废止日期 {end_date}",
                        end_columns[0],
                        {"start_date": start_date, "end_date": end_date},
                    ))

            unit_columns = [column for column in data if "单价" in column]
            quantity_columns = [column for column in data if "数量" in column]
            total_columns = [column for column in data if "总价" in column]
            if unit_columns and quantity_columns and total_columns:
                unit_price = _parse_amount(data.get(unit_columns[0], ""))
                quantity = _parse_quantity(data.get(quantity_columns[0], ""))
                total_price = _parse_amount(data.get(total_columns[0], ""))
                if unit_price is not None and quantity is not None and total_price is not None:
                    expected_total = unit_price * quantity
                    if abs(expected_total - total_price) > max(0.01, expected_total * 0.001):
                        issues.append(_issue(
                            table,
                            row,
                            "amount_mismatch",
                            "error",
                            f"单价×数量={expected_total:.2f}，与总价 {total_price:.2f} 不一致",
                            total_columns[0],
                            {
                                "unit_price": unit_price,
                                "quantity": quantity,
                                "total_price": total_price,
                                "expected_total": expected_total,
                            },
                        ))

        number_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for column, value in row.get("data", {}).items():
                if _is_document_number_header(column) and _clean_text(value):
                    number_groups[f"{column}:{_clean_text(value)}"].append({
                        "row": row,
                        "column": column,
                        "value": _clean_text(value),
                    })

        for group in number_groups.values():
            if len(group) <= 1:
                continue
            first = group[0]
            issues.append(_issue(
                table,
                first["row"],
                "duplicate_identifier",
                "warning",
                f"编号「{first['value']}」在同一表格中重复出现 {len(group)} 次",
                first["column"],
                {
                    "value": first["value"],
                    "count": len(group),
                    "row_indexes": [int(item["row"].get("row_index", -1)) for item in group],
                    "source": source,
                    "page": page,
                    "table_index": table_index,
                },
            ))

    return fields, issues
