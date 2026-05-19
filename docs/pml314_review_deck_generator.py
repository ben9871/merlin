"""Generate the PML-314 review slide deck from local review artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pml314_review_slides.pptx"

W = 13.333
H = 7.5

NAVY = RGBColor(18, 34, 52)
TEAL = RGBColor(0, 122, 130)
GREEN = RGBColor(55, 128, 88)
ORANGE = RGBColor(205, 110, 45)
GRAY = RGBColor(94, 105, 116)
LIGHT = RGBColor(245, 247, 249)
WHITE = RGBColor(255, 255, 255)
PINK = RGBColor(172, 74, 117)


def inch(value: float):
    return Inches(value)


def set_text(frame, text: str, *, size=20, bold=False, color=NAVY, align=None):
    frame.clear()
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    paragraph.alignment = align or PP_ALIGN.LEFT
    frame.margin_left = Inches(0.08)
    frame.margin_right = Inches(0.08)
    frame.margin_top = Inches(0.04)
    frame.margin_bottom = Inches(0.04)


def add_title(slide, title: str, subtitle: str | None = None):
    box = slide.shapes.add_textbox(inch(0.45), inch(0.25), inch(12.4), inch(0.55))
    set_text(box.text_frame, title, size=25, bold=True, color=NAVY)
    accent = slide.shapes.add_shape(1, inch(0.45), inch(0.9), inch(1.55), inch(0.05))
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()
    if subtitle:
        sub = slide.shapes.add_textbox(inch(0.45), inch(0.98), inch(12.0), inch(0.35))
        set_text(sub.text_frame, subtitle, size=11, color=GRAY)


def add_footer(slide, index: int):
    box = slide.shapes.add_textbox(inch(11.9), inch(7.05), inch(0.95), inch(0.25))
    set_text(box.text_frame, f"PML-314 / {index}", size=8, color=GRAY, align=PP_ALIGN.RIGHT)


def add_bullets(slide, bullets, x, y, w, h, *, size=18, color=NAVY):
    box = slide.shapes.add_textbox(inch(x), inch(y), inch(w), inch(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = 0
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(7)
    return box


def add_label_box(slide, x, y, w, h, title, body=None, *, fill=LIGHT, line=TEAL):
    shape = slide.shapes.add_shape(1, inch(x), inch(y), inch(w), inch(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1.2)
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = NAVY
    if body:
        p2 = tf.add_paragraph()
        p2.text = body
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = GRAY
    return shape


def add_code(slide, code: str, x, y, w, h, *, size=13):
    shape = slide.shapes.add_shape(1, inch(x), inch(y), inch(w), inch(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(39, 48, 59)
    shape.line.color.rgb = RGBColor(39, 48, 59)
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = False
    for i, line in enumerate(code.splitlines()):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.name = "Consolas"
        p.font.color.rgb = WHITE
    return shape


def parse_dot(path: Path):
    nodes: dict[str, dict[str, str]] = {}
    edges: list[tuple[str, str]] = []
    node_re = re.compile(r'^(node_[a-f0-9]+) \[label="([^"]+)" name="([^"]+)"')
    edge_re = re.compile(r"^(node_[a-f0-9]+) -> (node_[a-f0-9]+)")
    for raw in path.read_text().splitlines():
        line = raw.strip()
        node_match = node_re.match(line)
        if node_match:
            node_id, label, name = node_match.groups()
            nodes[node_id] = {"label": label, "name": name}
            continue
        edge_match = edge_re.match(line)
        if edge_match:
            edges.append(edge_match.groups())
    by_name = {data["name"]: node_id for node_id, data in nodes.items()}
    return nodes, by_name, edges


def draw_edge(slide, start, end, *, color=GRAY):
    x1, y1, w1, h1 = start
    x2, y2, w2, h2 = end
    conn = slide.shapes.add_connector(
        1,
        inch(x1 + w1),
        inch(y1 + h1 / 2),
        inch(x2),
        inch(y2 + h2 / 2),
    )
    conn.line.color.rgb = color
    conn.line.width = Pt(1.2)


def draw_code2flow(slide, dot_file: str, specs, *, title=None):
    nodes, by_name, edges = parse_dot(ROOT / dot_file)
    id_to_spec = {}
    for spec in specs:
        name = spec["name"]
        node_id = by_name.get(name)
        if node_id is None:
            continue
        id_to_spec[node_id] = spec

    if title:
        label = slide.shapes.add_textbox(inch(0.55), inch(1.25), inch(12.0), inch(0.3))
        set_text(label.text_frame, title, size=12, bold=True, color=GRAY)

    drawn = {}
    for node_id, spec in id_to_spec.items():
        label = spec.get("label") or nodes[node_id]["name"].split("::")[-1]
        shape = add_label_box(
            slide,
            spec["x"],
            spec["y"],
            spec["w"],
            spec["h"],
            label,
            spec.get("body"),
            fill=spec.get("fill", LIGHT),
            line=spec.get("line", TEAL),
        )
        drawn[node_id] = (spec["x"], spec["y"], spec["w"], spec["h"], shape)

    for src, dst in edges:
        if src == dst:
            continue
        if src in drawn and dst in drawn:
            draw_edge(slide, drawn[src][:4], drawn[dst][:4])


def add_table(slide, rows, x, y, w, h, *, col_widths=None, font_size=10):
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), inch(x), inch(y), inch(w), inch(h))
    table = table_shape.table
    if col_widths:
        for i, width in enumerate(col_widths):
            table.columns[i].width = inch(width)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.text = text
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if r else TEAL
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = WHITE if r == 0 else NAVY
                    run.font.bold = r == 0
    return table_shape


def new_deck():
    prs = Presentation()
    prs.slide_width = inch(W)
    prs.slide_height = inch(H)
    blank = prs.slide_layouts[6]
    return prs, blank


def slide(prs, blank, title, subtitle=None):
    s = prs.slides.add_slide(blank)
    add_title(s, title, subtitle)
    add_footer(s, len(prs.slides))
    return s


def main():
    prs, blank = new_deck()

    s = slide(prs, blank, "PML-314: ComputationSpace QLOQ Output Selection", "Review deck for the API-first implementation")
    add_bullets(
        s,
        [
            "Goal: keep the old ComputationSpace API usable while adding QLOQ and custom partitioned output spaces.",
            "Public meaning stays output-side: it selects and orders measurement outputs.",
            "Backend mapping now reuses the EncodingSpace basis machinery instead of duplicating ordering rules.",
        ],
        0.65,
        1.75,
        7.2,
        3.0,
        size=18,
    )
    add_label_box(s, 8.4, 1.85, 3.8, 1.0, "Main API", "ComputationSpace.qloq([2, 1])")
    add_label_box(s, 8.4, 3.05, 3.8, 1.0, "Output size", "(4, 2) -> 8 retained states")
    add_label_box(s, 8.4, 4.25, 3.8, 1.0, "Scope", "Correctness and API now; graph optimization later")

    s = slide(prs, blank, "Why This Ticket Exists")
    add_bullets(
        s,
        [
            "EncodingSpace already understood partitioned and QLOQ layouts.",
            "ComputationSpace only exposed fock, unbunched, and dual_rail.",
            "MeasurementStrategy could not ask for QLOQ-shaped output selection directly.",
            "The gap was artificial: input embedding and output filtering use the same basis mapping ideas, but they are different public contracts.",
        ],
        0.65,
        1.55,
        7.2,
        4.3,
    )
    add_label_box(s, 8.25, 1.55, 4.0, 1.15, "Before", "QLOQ output selection required indirect unbunched/filter assumptions", fill=RGBColor(253, 242, 235), line=ORANGE)
    add_label_box(s, 8.25, 3.0, 4.0, 1.15, "After", "QLOQ is a first-class ComputationSpace option", fill=RGBColor(235, 247, 247), line=TEAL)

    s = slide(prs, blank, "Design Boundary")
    add_label_box(s, 0.75, 1.6, 5.5, 2.7, "EncodingSpace", "Input-side contract. It embeds logical amplitudes into full Fock space before simulation.", fill=RGBColor(236, 244, 252), line=TEAL)
    add_label_box(s, 7.05, 1.6, 5.5, 2.7, "ComputationSpace", "Output-side contract. It retains, orders, and exposes output basis states for measurement.", fill=RGBColor(238, 247, 241), line=GREEN)
    add_bullets(
        s,
        [
            "They may share mapping code.",
            "They should not become aliases.",
            "StateVector.from_tensor(...) behavior is not part of this ticket.",
        ],
        1.0,
        4.75,
        11.0,
        1.2,
        size=17,
    )

    s = slide(prs, blank, "User-Facing API")
    add_code(
        s,
        """from merlin.core import ComputationSpace

assert ComputationSpace.FOCK.kind == "fock"
assert ComputationSpace.UNBUNCHED.kind == "unbunched"
assert ComputationSpace.DUAL_RAIL.kind == "dual_rail"

space = ComputationSpace(modes_per_photon=[3, 4, 2])
qloq = ComputationSpace.qloq(qubit_groups=[2, 1])

assert qloq.modes_per_photon == (4, 2)""",
        0.65,
        1.35,
        6.25,
        4.7,
        size=12,
    )
    add_bullets(
        s,
        [
            "Built-ins are still the normal path.",
            "Partitioned spaces define one photon per configured block.",
            "QLOQ expands qubit groups deterministically with 2**k modes per group.",
        ],
        7.25,
        1.55,
        5.2,
        3.2,
        size=17,
    )

    s = slide(prs, blank, "Compatibility Contract")
    add_table(
        s,
        [
            ("Old expectation", "Status"),
            ("ComputationSpace.FOCK / UNBUNCHED / DUAL_RAIL", "Preserved"),
            ("ComputationSpace.coerce('fock') is ComputationSpace.FOCK", "Preserved"),
            ("space.value and space.name", "Preserved"),
            ("list(ComputationSpace)", "Preserved for built-ins"),
            ("isinstance(ComputationSpace.FOCK, str)", "Preserved"),
            ("Custom qloq compares equal to plain 'qloq'", "Not preserved intentionally"),
        ],
        0.75,
        1.35,
        11.8,
        4.7,
        col_widths=(6.3, 5.5),
        font_size=11,
    )
    add_bullets(
        s,
        ["Custom spaces need metadata-aware equality, so two different QLOQ layouts do not collapse to the same plain string."],
        0.95,
        6.15,
        11.5,
        0.7,
        size=13,
        color=GRAY,
    )

    s = slide(prs, blank, "Implementation Shape")
    add_label_box(s, 0.75, 1.45, 3.7, 1.25, "String-backed", "Built-ins still behave like strings and keep legacy values.")
    add_label_box(s, 4.85, 1.45, 3.7, 1.25, "Metadata-carrying", "Custom spaces store family, kind, modes_per_photon, qubit_groups.")
    add_label_box(s, 8.95, 1.45, 3.7, 1.25, "Mapping delegate", "Basis helpers forward to EncodingSpace internally.")
    add_code(
        s,
        """ComputationSpace.qloq([2, 1])
# kind: "qloq"
# family: "partitioned"
# modes_per_photon: (4, 2)
# n_modes: 6
# n_photons: 2""",
        1.1,
        3.25,
        5.2,
        2.4,
    )
    add_bullets(
        s,
        [
            "Built-ins are singleton constants.",
            "Custom spaces are ordinary immutable values.",
            "The public type remains ComputationSpace, not EncodingSpace.",
        ],
        7.05,
        3.3,
        5.2,
        2.0,
        size=16,
    )

    s = slide(prs, blank, "QLOQ Mapping Example")
    add_bullets(
        s,
        [
            "Input: qubit_groups=[2, 1].",
            "Expansion: modes_per_photon=(4, 2).",
            "Photon 0 chooses one of modes 0..3.",
            "Photon 1 chooses one of modes 4..5.",
            "Output basis size: 4 * 2 = 8.",
        ],
        0.8,
        1.55,
        4.5,
        4.2,
        size=17,
    )
    add_code(
        s,
        """qloq.fock_basis_states()

(1,0,0,0,1,0)
(1,0,0,0,0,1)
(0,1,0,0,1,0)
(0,1,0,0,0,1)
...
(0,0,0,1,0,1)""",
        6.0,
        1.45,
        5.7,
        4.4,
        size=13,
    )

    s = slide(prs, blank, "Code2flow: Shared Basis Mapping")
    draw_code2flow(
        s,
        "pml314_code2flow_basis.dot",
        [
            {"name": "computation_space::ComputationSpace.fock_basis_states", "x": 0.8, "y": 2.0, "w": 3.1, "h": 0.9, "label": "ComputationSpace.fock_basis_states()", "body": "public output ordering"},
            {"name": "computation_space::ComputationSpace._as_encoding_space", "x": 5.0, "y": 2.0, "w": 3.0, "h": 0.9, "label": "_as_encoding_space()", "body": "private delegate"},
            {"name": "computation_space::ComputationSpace.qloq", "x": 9.0, "y": 1.35, "w": 2.7, "h": 0.8, "label": "EncodingSpace.qloq()", "body": "QLOQ expansion"},
            {"name": "computation_space::_encoding_space_cls", "x": 9.0, "y": 2.75, "w": 2.7, "h": 0.8, "label": "_encoding_space_cls()", "body": "lazy import"},
        ],
    )
    add_bullets(s, ["The public computation API is distinct, but the ordering and mapping behavior comes from the same backend logic."], 1.0, 5.25, 11.0, 0.75, size=15, color=GRAY)

    s = slide(prs, blank, "Measurement Strategy Boundary")
    add_code(
        s,
        """qloq = ComputationSpace.qloq([2, 1])

strategy = MeasurementStrategy.probs(qloq)
assert strategy.computation_space == qloq

strategy = MeasurementStrategy.mode_expectations(qloq)
strategy = MeasurementStrategy.amplitudes(qloq)
strategy = MeasurementStrategy.partial([0], qloq)""",
        0.8,
        1.45,
        6.2,
        3.9,
        size=12,
    )
    add_bullets(
        s,
        [
            "Factory signatures accept ComputationSpace | str.",
            "Factories still default to UNBUNCHED.",
            "The strategy object just stores the selected output-space contract.",
        ],
        7.45,
        1.65,
        4.8,
        2.4,
        size=17,
    )

    s = slide(prs, blank, "Code2flow: QuantumLayer Output Selection")
    draw_code2flow(
        s,
        "pml314_code2flow_layer_output.dot",
        [
            {"name": "process::ComputationProcessFactory.create", "x": 0.55, "y": 1.7, "w": 2.35, "h": 0.8, "label": "Factory.create()"},
            {"name": "process::ComputationProcess.__init__", "x": 3.3, "y": 1.7, "w": 2.2, "h": 0.8, "label": "ComputationProcess.__init__()"},
            {"name": "process::ComputationProcess._setup_computation_graphs", "x": 6.0, "y": 1.7, "w": 2.65, "h": 0.8, "label": "_setup_computation_graphs()"},
            {"name": "process::ComputationProcess._computation_space_output_map", "x": 6.0, "y": 3.15, "w": 2.65, "h": 0.9, "label": "_computation_space_output_map()", "body": "custom spaces only"},
            {"name": "slos_torchscript::build_slos_distribution_computegraph", "x": 9.15, "y": 1.7, "w": 3.0, "h": 0.8, "label": "build_slos_distribution_computegraph()"},
            {"name": "slos_torchscript::SLOSComputeGraph.__init__", "x": 9.15, "y": 3.15, "w": 3.0, "h": 0.8, "label": "SLOSComputeGraph.__init__()"},
            {"name": "slos_torchscript::SLOSComputeGraph._build_graph_structure", "x": 2.0, "y": 5.05, "w": 2.7, "h": 0.8, "label": "_build_graph_structure()"},
            {"name": "slos_torchscript::SLOSComputeGraph._create_torchscript_modules", "x": 5.25, "y": 5.05, "w": 3.0, "h": 0.8, "label": "_create_torchscript_modules()"},
            {"name": "slos_torchscript::_serialize_computation_space", "x": 8.8, "y": 5.05, "w": 2.9, "h": 0.8, "label": "_serialize_computation_space()"},
        ],
    )

    s = slide(prs, blank, "How Output Selection Works")
    add_bullets(
        s,
        [
            "For built-ins, graph construction remains the existing built-in path.",
            "For custom spaces, ComputationProcess builds an output_map_func from space.fock_basis_states(...).",
            "The SLOS graph keeps only mapped final states and accumulates repeated mapped outputs.",
            "Probabilities and amplitudes are renormalized because non-FOCK spaces are postselected.",
        ],
        0.75,
        1.45,
        6.2,
        4.5,
        size=16,
    )
    add_label_box(s, 7.55, 1.55, 4.1, 0.9, "API first", "No graph-level optimization required for this ticket.", fill=RGBColor(235, 247, 247), line=TEAL)
    add_label_box(s, 7.55, 2.75, 4.1, 0.9, "Correctness", "Output keys match qloq.fock_basis_states().", fill=RGBColor(238, 247, 241), line=GREEN)
    add_label_box(s, 7.55, 3.95, 4.1, 0.9, "Future optimization", "Restrict photon ranges directly in the graph later.", fill=RGBColor(253, 242, 235), line=ORANGE)

    s = slide(prs, blank, "Code2flow: ProbabilityDistribution Filtering")
    draw_code2flow(
        s,
        "pml314_code2flow_probability_filter.dot",
        [
            {"name": "probability_distribution::ProbabilityDistribution.filter", "x": 0.55, "y": 2.4, "w": 2.6, "h": 0.85, "label": "ProbabilityDistribution.filter()"},
            {"name": "computation_space::ComputationSpace.coerce", "x": 3.75, "y": 1.4, "w": 2.55, "h": 0.75, "label": "ComputationSpace.coerce()"},
            {"name": "probability_distribution::_space_predicate", "x": 3.75, "y": 2.4, "w": 2.55, "h": 0.75, "label": "_space_predicate()"},
            {"name": "probability_distribution::_basis_for_space", "x": 3.75, "y": 3.4, "w": 2.55, "h": 0.75, "label": "_basis_for_space()"},
            {"name": "computation_space::ComputationSpace.fock_basis_states", "x": 6.95, "y": 2.4, "w": 2.75, "h": 0.85, "label": "space.fock_basis_states()"},
            {"name": "computation_space::ComputationSpace._as_encoding_space", "x": 10.15, "y": 2.4, "w": 2.5, "h": 0.85, "label": "_as_encoding_space()"},
            {"name": "probability_distribution::FilteredBasis.__init__", "x": 3.75, "y": 4.85, "w": 2.55, "h": 0.75, "label": "FilteredBasis(...)"},
            {"name": "probability_distribution::ProbabilityDistribution.to_dense", "x": 0.55, "y": 4.85, "w": 2.6, "h": 0.75, "label": "to_dense()"},
        ],
    )

    s = slide(prs, blank, "Conversion And Normalization Updates")
    draw_code2flow(
        s,
        "pml314_code2flow_conversion.dot",
        [
            {"name": "utils::pcvl_to_tensor", "x": 0.8, "y": 2.1, "w": 2.8, "h": 0.85, "label": "pcvl_to_tensor()"},
            {"name": "computation_space::ComputationSpace.fock_basis_states", "x": 4.5, "y": 2.1, "w": 3.0, "h": 0.85, "label": "space.fock_basis_states()"},
            {"name": "computation_space::ComputationSpace._as_encoding_space", "x": 8.4, "y": 2.1, "w": 2.9, "h": 0.85, "label": "_as_encoding_space()"},
            {"name": "computation_space::_encoding_space_cls", "x": 8.4, "y": 3.65, "w": 2.9, "h": 0.8, "label": "_encoding_space_cls()"},
        ],
    )
    add_bullets(
        s,
        [
            "pcvl_to_tensor no longer assumes .value is a Combinadics scheme.",
            "embed_tensor_in_fock_basis can remap compact custom-space tensors.",
            "Normalization now treats every non-FOCK computation space as postselected.",
            "Mode expectations use occupancy masks for postselected spaces.",
        ],
        0.9,
        5.1,
        11.7,
        1.3,
        size=14,
    )

    s = slide(prs, blank, "Remote Execution Boundary")
    add_label_box(s, 0.85, 1.55, 5.4, 1.6, "Current remote mapping", "Uses built-in scheme strings and Combinadics(scheme, n_photons, n_modes).", fill=RGBColor(253, 242, 235), line=ORANGE)
    add_label_box(s, 7.0, 1.55, 5.4, 1.6, "Custom-space mapping", "Should use space.fock_basis_states(n_modes=..., n_photons=...) client-side.", fill=RGBColor(235, 247, 247), line=TEAL)
    add_bullets(
        s,
        [
            "QLOQ is not inherently incompatible with remote results if remote returns Fock/BasicState outcomes.",
            "The conservative implementation fails clearly instead of silently using the wrong built-in basis.",
            "Follow-up option: update MerlinProcessor._get_state_mapping(...) to consume the full ComputationSpace object.",
        ],
        1.0,
        4.0,
        11.4,
        1.6,
        size=16,
    )

    s = slide(prs, blank, "Files That Matter")
    add_table(
        s,
        [
            ("Area", "Main files"),
            ("Public API", "merlin/core/computation_space.py"),
            ("Measurement boundary", "merlin/measurement/strategies.py"),
            ("Layer output selection", "merlin/core/process.py, merlin/pcvl_pytorch/slos_torchscript.py"),
            ("Filtering and conversion", "probability_distribution.py, state_vector.py, pcvl_pytorch/utils.py"),
            ("Postselection semantics", "utils/normalization.py, measurement/mappers.py"),
            ("Docs and tests", "computation_space.rst, test_computation_space_core.py, test_strategies.py"),
        ],
        0.65,
        1.35,
        12.0,
        5.0,
        col_widths=(3.2, 8.8),
        font_size=10,
    )

    s = slide(prs, blank, "Tests And Validation")
    add_bullets(
        s,
        [
            "70 focused tests passed: computation-space API, probability filtering, measurement strategies, pcvl_to_tensor.",
            "38 runtime-path tests passed: SLOS, normalization, mappers, unbunched behavior.",
            "108 layer/kernel tests passed, 4 skipped.",
            "Ruff check and Ruff format passed.",
            "Mypy hook-equivalent passed on touched Merlin modules.",
            "Strict Sphinx docs passed with -W --keep-going -n.",
        ],
        0.8,
        1.45,
        11.8,
        4.4,
        size=16,
    )

    s = slide(prs, blank, "What Did Not Change")
    add_label_box(s, 0.9, 1.55, 3.7, 1.3, "No input API work", "StateVector.from_tensor(...) integration is not part of PML-314.", fill=LIGHT, line=GRAY)
    add_label_box(s, 4.9, 1.55, 3.7, 1.3, "No built-in behavior change", "FOCK, UNBUNCHED, DUAL_RAIL ordering and defaults remain the same.", fill=LIGHT, line=GRAY)
    add_label_box(s, 8.9, 1.55, 3.7, 1.3, "No graph optimization", "Custom spaces filter outputs now; direct graph pruning can come later.", fill=LIGHT, line=GRAY)
    add_bullets(
        s,
        [
            "The changes are additive capability extensions.",
            "Existing users should not need to migrate code.",
            "The major review question is whether remote custom-space support belongs in this PR or a follow-up.",
        ],
        1.0,
        4.15,
        11.2,
        1.5,
        size=17,
    )

    s = slide(prs, blank, "Reviewer Checklist")
    add_bullets(
        s,
        [
            "Does the enum-like compatibility contract cover the expectations we care about?",
            "Is it acceptable that custom spaces are string-backed but metadata-distinct?",
            "Is QLOQ output ordering exactly the intended partition product order?",
            "Should remote client-side custom-space mapping be included now?",
            "Is output filtering acceptable for this API ticket, with graph pruning later?",
        ],
        0.85,
        1.55,
        11.6,
        3.5,
        size=18,
    )
    add_code(
        s,
        """Review artifacts:
docs/pml314_code2flow_full.dot
docs/pml314_code2flow_basis.dot
docs/pml314_code2flow_layer_output.dot
docs/pml314_code2flow_probability_filter.dot
docs/pml314_code2flow_conversion.dot""",
        1.0,
        5.25,
        10.9,
        1.1,
        size=11,
    )

    prs.save(OUT)


if __name__ == "__main__":
    main()
