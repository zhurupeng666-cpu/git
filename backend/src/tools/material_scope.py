"""数据库材料分类名称和报告范围校验。"""
MATERIAL_CATEGORY_LABELS = {
    "wafer": "晶圆材料", "lithography": "光刻材料", "cmp": "CMP抛光材料",
    "specialty_gas": "半导体工艺气体", "precursor": "沉积前驱体",
    "target": "靶材", "wet_chemical": "湿法化学品", "packaging": "封装材料",
    "spare_part": "备品备件", "cleanroom": "洁净室耗材",
}


def normalize_material_category(category):
    if category is None:
        return None
    value = category.strip()
    if value in MATERIAL_CATEGORY_LABELS:
        return value
    aliases = {label: code for code, label in MATERIAL_CATEGORY_LABELS.items()}
    aliases.update({"工艺气体": "specialty_gas", "半导体气体": "specialty_gas"})
    if value in aliases:
        return aliases[value]
    raise ValueError("无法识别材料分类，请选择：" + "、".join(MATERIAL_CATEGORY_LABELS.values()))


def validate_report_scope(scope, category=None, material_code=None):
    category = normalize_material_category(category)
    material_code = material_code.strip() if material_code else None
    if scope == "all":
        if category or material_code:
            raise ValueError("全部材料报告不能同时指定分类或物料，请确认报告范围。")
    elif scope == "category":
        if not category or material_code:
            raise ValueError("分类报告需要指定一个材料分类；如要查询单一物料，请选择物料报告。")
    elif scope == "material":
        if not material_code or category:
            raise ValueError("物料报告需要指定物料编码；如要查询整个分类，请选择分类报告。")
    else:
        raise ValueError("请说明报告范围：全部材料、某类材料，还是指定物料。")
    return category, material_code
