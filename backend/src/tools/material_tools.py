from src.config.request_limits import statement_options

import os
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from langchain_core.tools import tool
from src.tools.material_scope import normalize_material_category

from datetime import date, datetime, time, timedelta

# 加载项目环境变量
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@tool
def query_material_inventory(material_code: str) -> dict:
    """
    查询指定半导体材料的当前库存、仓库及批次信息。
    :param material_code: 物料编码，例如MAT-WAF-001
    :return: 包含库存总量、单位和各仓库批次明细的字典
    """

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("未找到DATABASE_URL环境变量")

    # 通过参数化SQL查询，避免SQL注入
    sql = """
        SELECT
            material_code,
            material_name,
            specification,
            unit,
            batch_code,
            quality_status,
            expiry_date,
            warehouse_code,
            warehouse_name,
            current_stock
        FROM material_mgmt.v_inventory_balance
        WHERE material_code = %s
          AND current_stock > 0
        ORDER BY warehouse_code, batch_code
    """

    # 连接PostgreSQL数据库
    with psycopg.connect(
        db_url,
        row_factory=dict_row,
        connect_timeout=10,
        options=statement_options("TOOL_QUERY_TIMEOUT_MS", "30000")
    ) as conn:

        with conn.cursor() as cursor:
            cursor.execute(sql, (material_code,))
            rows = cursor.fetchall()

    # 没有查询到正库存记录
    if not rows:
        return {
            "material_code": material_code,
            "total_stock": "0",
            "details": [],
            "message": "未找到该物料的正库存记录"
        }

    # 计算账面库存总量
    total_stock = sum(
        (row["current_stock"] for row in rows),
        Decimal("0")
    )

    # 整理各仓库、各批次库存
    details = []

    for row in rows:
        details.append({
            "warehouse_code": row["warehouse_code"],
            "warehouse_name": row["warehouse_name"],
            "batch_code": row["batch_code"],
            "quality_status": row["quality_status"],
            "expiry_date": (
                row["expiry_date"].isoformat()
                if row["expiry_date"]
                else None
            ),
            "quantity": str(row["current_stock"])
        })

    return {
        "material_code": material_code,
        "material_name": rows[0]["material_name"],
        "specification": rows[0]["specification"],
        "unit": rows[0]["unit"],
        "total_stock": str(total_stock),
        "batch_count": len({
            row["batch_code"] for row in rows
        }),
        "details": details,
        "note": "当前为账面库存，未扣除冻结、预留或不合格批次"
    }


@tool
def query_material_transactions(
    material_code: str,
    start_date: str,
    end_date: str
) -> dict:
    """
    查询指定材料在某一时间范围内的出入库流水及数量统计。
    :param material_code: 物料编码，例如MAT-WAF-001
    :param start_date: 开始日期，格式YYYY-MM-DD
    :param end_date: 结束日期，格式YYYY-MM-DD，包含当天
    :return: 出入库总数、入库量、出库量、净变化及流水明细
    """

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("未找到DATABASE_URL环境变量")

    # 1. 日期参数检查
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("开始日期不能晚于结束日期")

    start_time = datetime.combine(start, time.min)
    end_time = datetime.combine(
        end + timedelta(days=1),
        time.min
    )

    # 2. 查询指定材料的出入库流水
    sql = """
        SELECT
            m.material_code,
            m.material_name,
            m.unit,
            b.batch_code,
            w.warehouse_code,
            w.warehouse_name,
            t.transaction_code,
            t.transaction_type,
            t.quantity,
            t.transaction_time,
            t.reference_no,
            t.production_lot

        FROM material_mgmt.inventory_transactions t

        JOIN material_mgmt.material_batches b
            ON t.batch_id = b.batch_id

        JOIN material_mgmt.materials m
            ON b.material_id = m.material_id

        JOIN material_mgmt.warehouses w
            ON t.warehouse_id = w.warehouse_id

        WHERE m.material_code = %s

          AND t.transaction_time >= (
              %s::timestamp AT TIME ZONE 'Asia/Shanghai'
          )

          AND t.transaction_time < (
              %s::timestamp AT TIME ZONE 'Asia/Shanghai'
          )

        ORDER BY
            t.transaction_time,
            t.transaction_id
    """

    # 3. 连接数据库并执行查询
    with psycopg.connect(
        db_url,
        row_factory=dict_row,
        connect_timeout=10,
        options=statement_options("TOOL_QUERY_TIMEOUT_MS", "30000")
    ) as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                sql,
                (material_code, start_time, end_time)
            )

            rows = cursor.fetchall()

    # 4. 没有出入库记录
    if not rows:
        return {
            "material_code": material_code,
            "start_date": start_date,
            "end_date": end_date,
            "total_records": 0,
            "message": "指定时间范围内没有出入库记录",
            "records": []
        }

    # 5. 定义增加库存的业务类型
    inbound_types = {
        "opening",
        "inbound",
        "return_in",
        "transfer_in",
        "adjust_in"
    }

    # 6. 统计入库与出库数量
    total_in = Decimal("0")
    total_out = Decimal("0")

    for row in rows:

        quantity = row["quantity"]

        if row["transaction_type"] in inbound_types:
            total_in += quantity
        else:
            total_out += quantity

    net_change = total_in - total_out

    # 7. 组织流水明细
    records = []

    # 最多返回前50条，避免工具输出过大
    for row in rows[:50]:

        records.append({
            "transaction_code": row["transaction_code"],
            "transaction_type": row["transaction_type"],
            "quantity": str(row["quantity"]),
            "transaction_time": (
                row["transaction_time"].isoformat()
            ),
            "warehouse_code": row["warehouse_code"],
            "warehouse_name": row["warehouse_name"],
            "batch_code": row["batch_code"],
            "reference_no": row["reference_no"],
            "production_lot": row["production_lot"]
        })

    # 8. 返回完整统计结果
    return {
        "material_code": material_code,
        "material_name": rows[0]["material_name"],
        "unit": rows[0]["unit"],
        "start_date": start_date,
        "end_date": end_date,
        "total_records": len(rows),
        "total_inbound": str(total_in),
        "total_outbound": str(total_out),
        "net_change": str(net_change),
        "returned_records": len(records),
        "has_more": len(rows) > 50,
        "records": records
    }


@tool
def generate_material_statistics(
    start_date: str,
    end_date: str,
    category: str | None = None,
    material_code: str | None = None
) -> dict:
    """
    统计指定周期内材料的库存变化，可限定材料分类或物料编码。
    :param category: 可选分类代码，例如specialty_gas表示半导体工艺气体
    :param material_code: 可选物料编码，指定时仅统计该物料
    :param start_date: 统计开始日期，格式YYYY-MM-DD
    :param end_date: 统计结束日期，格式YYYY-MM-DD
    :return: 各物料期初库存、入库量、出库量和期末库存
    """

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("未找到DATABASE_URL环境变量")

    category = normalize_material_category(category)
    material_code = material_code.strip() if material_code else None

    # 1. 日期校验
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("开始日期不能晚于结束日期")

    # 结束日期包含当天
    end_exclusive = end + timedelta(days=1)

    # 2. 统计库存
    sql = """
        WITH ledger AS (
            SELECT
                b.material_id,
                t.transaction_time,
                t.transaction_type,
                t.quantity,

                CASE
                    WHEN t.transaction_type IN (
                        'opening',
                        'inbound',
                        'return_in',
                        'transfer_in',
                        'adjust_in'
                    )
                    THEN t.quantity
                    ELSE -t.quantity
                END AS signed_quantity

            FROM material_mgmt.inventory_transactions t

            JOIN material_mgmt.material_batches b
                ON t.batch_id = b.batch_id

            WHERE t.transaction_time < (
                %s::date::timestamp
                AT TIME ZONE 'Asia/Shanghai'
            )
        )

        SELECT
            m.material_code,
            m.material_name,
            m.category,
            m.unit,

            COALESCE(
                SUM(l.signed_quantity) FILTER (
                    WHERE l.transaction_time < (
                        %s::date::timestamp
                        AT TIME ZONE 'Asia/Shanghai'
                    )
                ), 0
            ) AS opening_stock,

            COALESCE(
                SUM(l.quantity) FILTER (
                    WHERE l.transaction_time >= (
                        %s::date::timestamp
                        AT TIME ZONE 'Asia/Shanghai'
                    )
                    AND l.transaction_type IN (
                        'opening',
                        'inbound',
                        'return_in',
                        'transfer_in',
                        'adjust_in'
                    )
                ), 0
            ) AS total_inbound,

            COALESCE(
                SUM(l.quantity) FILTER (
                    WHERE l.transaction_time >= (
                        %s::date::timestamp
                        AT TIME ZONE 'Asia/Shanghai'
                    )
                    AND l.transaction_type IN (
                        'outbound',
                        'return_out',
                        'transfer_out',
                        'adjust_out'
                    )
                ), 0
            ) AS total_outbound,

            COALESCE(
                SUM(l.signed_quantity), 0
            ) AS closing_stock

        FROM material_mgmt.materials m

        LEFT JOIN ledger l
            ON m.material_id = l.material_id

        WHERE (%s::text IS NULL OR m.category = %s)
          AND (%s::text IS NULL OR m.material_code = %s)

        GROUP BY
            m.material_id,
            m.material_code,
            m.material_name,
            m.category,
            m.unit

        ORDER BY m.material_code
    """

    # 3. 执行SQL
    with psycopg.connect(
        db_url,
        row_factory=dict_row,
        connect_timeout=10,
        options=statement_options("TOOL_QUERY_TIMEOUT_MS", "30000")
    ) as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                sql,
                (
                    end_exclusive.isoformat(),
                    start.isoformat(),
                    start.isoformat(),
                    start.isoformat(),
                    category, category,
                    material_code, material_code
                )
            )

            rows = cursor.fetchall()

    # 4. 整理统计结果
    materials = []

    for row in rows:

        opening = row["opening_stock"]
        inbound = row["total_inbound"]
        outbound = row["total_outbound"]
        closing = row["closing_stock"]

        # 排除整个统计周期内没有库存的物料
        if (
            opening == 0
            and inbound == 0
            and outbound == 0
            and closing == 0
        ):
            continue

        materials.append({
            "material_code": row["material_code"],
            "material_name": row["material_name"],
            "category": row["category"],
            "unit": row["unit"],
            "opening_stock": str(opening),
            "total_inbound": str(inbound),
            "total_outbound": str(outbound),
            "net_change": str(inbound - outbound),
            "closing_stock": str(closing)
        })

    # 5. 返回统计结果
    return {
        "start_date": start_date,
        "end_date": end_date,
        "category": category,
        "material_code": material_code,
        "material_count": len(materials),
        "materials": materials,
        "note": (
            "库存按物料独立统计，不同计量单位不可直接相加。"
            "入库与出库统计包含调拨和库存调整。"
            "期末库存为账面数量，不代表全部可用于生产。"
        )
    }
