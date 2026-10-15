"""统一原生PDF、OCR文字和表格的数据结构；坐标使用页面左上角归一化坐标。"""

from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
import re


@dataclass
class DocumentBlock:
    page_number: int
    block_index: int
    kind: str
    text: str
    bbox: list[float]
    method: str
    confidence: float | None = None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.page_number < 1 or self.block_index < 0:
            raise ValueError("页码从1开始，块序号不能为负")
        if len(self.bbox) != 4 or not all(0 <= value <= 1 for value in self.bbox):
            raise ValueError("区域必须使用0到1的归一化坐标")
        if self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]:
            raise ValueError("区域坐标顺序无效")

    def to_dict(self):
        return asdict(self)


class TableReader(HTMLParser):
    """只读取表格内容，不执行HTML；合并单元格展开后仍保留行列关联。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if self.hidden:
            return
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            values = dict(attrs)
            self.cell = {
                "text": "",
                "rowspan": min(100, max(1, int(values.get("rowspan", 1)))),
                "colspan": min(100, max(1, int(values.get("colspan", 1)))),
            }
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += " "

    def handle_data(self, data):
        if self.cell is not None and not self.hidden:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
            return
        if self.hidden:
            return
        if tag in ("td", "th") and self.cell is not None:
            if self.row is not None:
                self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def grid(self):
        occupied = {}
        result = []
        for row_number, cells in enumerate(self.rows):
            column = 0
            for cell in cells:
                while (row_number, column) in occupied:
                    column += 1
                value = re.sub(r"\s+", " ", cell["text"]).strip().replace("|", "／")
                for dr in range(cell["rowspan"]):
                    for dc in range(cell["colspan"]):
                        occupied[(row_number + dr, column + dc)] = value
                column += cell["colspan"]
        if not occupied:
            return result
        width = max(column for _, column in occupied) + 1
        height = max(row for row, _ in occupied) + 1
        if width > 100 or height > 5000:
            raise ValueError("表格结构异常，需人工复核")
        for row in range(height):
            result.append([occupied.get((row, column), "") for column in range(width)])
        return result


def parse_table_html(value):
    reader = TableReader()
    reader.feed(value)
    return reader.grid()


def table_markdown(rows):
    if not rows:
        return ""
    lines = [" | ".join(rows[0]), " | ".join(["---"] * len(rows[0]))]
    lines.extend(" | ".join(row) for row in rows[1:])
    return "\n".join(lines)


def restore_rotated_bbox(box, angle):
    """把逆时针方向校正后的归一化区域映回原页，避免横向扫描引用错位。"""
    angle = int(angle) % 360
    if angle not in (0, 90, 180, 270):
        raise ValueError("只支持直角方向校正的坐标回映")
    transforms = {
        0: lambda x, y: (x, y),
        90: lambda x, y: (1 - y, x),
        180: lambda x, y: (1 - x, 1 - y),
        270: lambda x, y: (y, 1 - x),
    }
    corners = [transforms[angle](x, y) for x in (box[0], box[2]) for y in (box[1], box[3])]
    return [
        min(x for x, y in corners),
        min(y for x, y in corners),
        max(x for x, y in corners),
        max(y for x, y in corners),
    ]
