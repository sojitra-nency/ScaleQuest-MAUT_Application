"""Mermaid-based flowchart rendering with 3D-style visuals and phase-based layouts.

Implementation flows: LR direction with 3 subgraph phases (Data Setup / Analysis / Output).
Conceptual flows: TD direction with logical subgraph groupings.
3D effect: SVG linearGradient fills + feDropShadow filter applied via JS post-processing.
"""

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Heights (px) per method and side — generous enough for scrolling=False.
# ---------------------------------------------------------------------------
_HEIGHTS: dict[str, dict[str, int]] = {
    # impl  — LR pipeline, single row of boxes; height = one-node row + subgraph frame
    # concept — TD chain, rendered at natural (unscaled) width; height = natural SVG height
    "MANUAL":  {"impl": 220, "concept": 680},
    "AHP":     {"impl": 220, "concept": 470},
    "LOSS":    {"impl": 220, "concept": 680},
    "TOPSIS":  {"impl": 240, "concept": 1100},
}

# ---------------------------------------------------------------------------
# Mermaid diagram strings.
#
# Shape key:
#   ([TEXT])  stadium oval  → START/END terminals
#   [/TEXT/]  parallelogram → I/O nodes (file read, store CSV, show chart)
#   [TEXT]    rectangle     → process / computation steps
#   (TEXT)    rounded rect  → conceptual-flow nodes
#
# \\n inside a label = Mermaid line-break within that node's text.
# ---------------------------------------------------------------------------
_DIAGRAMS: dict[str, dict[str, str]] = {

    # ------------------------------------------------------------------
    # MAUT / MANUAL
    # ------------------------------------------------------------------
    "MANUAL": {
        "impl": """\
flowchart LR
    subgraph s1 ["Phase 1 — Data Setup"]
        direction TB
        A([START]) --> B[IMPORT REQUIRED\\nLIBRARIES]
        B --> C[/READ DATA FROM\\nEXCEL FILE/]
        C --> D[EXTRACT FEATURES]
    end
    subgraph s2 ["Phase 2 — MAUT Analysis"]
        direction TB
        E[ASSIGN WEIGHTS\\nMANUALLY]
        E --> F[APPLY LINEAR\\nUTILITY FUNCTION]
    end
    subgraph s3 ["Phase 3 — Output"]
        direction TB
        G[/STORE RESULT\\nTO CSV FILE/]
        G --> H[/SHOW THE\\nUTILITY CURVE/]
        H --> I([END])
    end
    D --> E
    F --> G""",

        "concept": """\
flowchart TD
    subgraph g1 ["Decision Matrix"]
        direction TB
        A(DETERMINE THE\\nDECISION MATRIX & CRITERIA)
        A --> B(ASSIGN WEIGHTS TO\\nEACH DM MANUALLY)
    end
    subgraph g2 ["Scoring & Ranking"]
        direction TB
        C(NORMALIZE THE DM)
        C --> D(CALCULATE WEIGHTED\\nNORMALIZED DM)
        D --> E(RANK THE\\nPREFERENCE ORDER)
    end
    B --> C""",
    },

    # ------------------------------------------------------------------
    # AHP
    # ------------------------------------------------------------------
    "AHP": {
        "impl": """\
flowchart LR
    subgraph s1 ["Phase 1 — Data Setup"]
        direction TB
        A([START]) --> B[IMPORT REQUIRED\\nLIBRARIES]
        B --> C[/READ DATA FROM\\nEXCEL FILE/]
        C --> D[EXTRACT FEATURES]
    end
    subgraph s2 ["Phase 2 — AHP Analysis"]
        direction TB
        E[DO PAIRWISE\\nCOMPARISONS]
        E --> F[APPLY EXPONENTIAL\\nUTILITY FUNCTION]
    end
    subgraph s3 ["Phase 3 — Output"]
        direction TB
        G[/STORE RESULT\\nTO CSV FILE/]
        G --> H[/SHOW THE\\nUTILITY CURVE/]
        H --> I([END])
    end
    D --> E
    F --> G""",

        "concept": """\
flowchart TD
    A(DEFINE THE\\nDECISION PROBLEM)
    A --> B(CREATE A HIERARCHY)
    B --> C(DO PAIRWISE\\nCOMPARISONS)
    C --> D(CALCULATE PRIORITY\\nVECTOR & MAX EIGENVALUE)""",
    },

    # ------------------------------------------------------------------
    # LOSS
    # ------------------------------------------------------------------
    "LOSS": {
        "impl": """\
flowchart LR
    subgraph s1 ["Phase 1 — Data Setup"]
        direction TB
        A([START]) --> B[IMPORT REQUIRED\\nLIBRARIES]
        B --> C[/READ DATA FROM\\nEXCEL FILE/]
        C --> D[EXTRACT FEATURES]
    end
    subgraph s2 ["Phase 2 — LOSS Analysis"]
        direction TB
        E[APPLY LOSS\\nFUNCTION]
        E --> F[APPLY EXPONENTIAL\\nUTILITY FUNCTION]
    end
    subgraph s3 ["Phase 3 — Output"]
        direction TB
        G[/STORE RESULT\\nTO CSV FILE/]
        G --> H[/SHOW THE\\nUTILITY CURVE/]
        H --> I([END])
    end
    D --> E
    F --> G""",

        "concept": """\
flowchart TD
    subgraph g1 ["Problem Scoping"]
        direction TB
        A(IDENTIFY MAIN FACTORS\\nFOR MINING METHOD SELECTION)
        A --> B(DESIGN\\nASSESSMENT CRITERIA)
    end
    subgraph g2 ["Loss Calculation"]
        direction TB
        C(DEFINE\\nLOSS FUNCTION)
        C --> D(AGGREGATE LOSSES)
        D --> E(RANK THE\\nPREFERENCE ORDER)
    end
    B --> C""",
    },

    # ------------------------------------------------------------------
    # TOPSIS
    # ------------------------------------------------------------------
    "TOPSIS": {
        "impl": """\
flowchart LR
    subgraph s1 ["Phase 1 — Data Setup"]
        direction TB
        A([START]) --> B[IMPORT REQUIRED\\nLIBRARIES]
        B --> C[/READ DATA FROM\\nEXCEL FILE/]
        C --> D[EXTRACT FEATURES]
    end
    subgraph s2 ["Phase 2 — TOPSIS Analysis"]
        direction TB
        E[DETERMINE POSITIVE &\\nNEGATIVE IDEAL SOLUTIONS]
        E --> F[CALCULATE\\nSEPARATION MEASURES]
        F --> G[CALCULATE RELATIVE\\nCLOSENESS TO IDEAL]
        G --> H[APPLY EXPONENTIAL\\nUTILITY FUNCTION]
    end
    subgraph s3 ["Phase 3 — Output"]
        direction TB
        I[/STORE RESULT\\nTO CSV FILE/]
        I --> J[/SHOW THE\\nUTILITY CURVE/]
        J --> K([END])
    end
    D --> E
    H --> I""",

        "concept": """\
flowchart TD
    subgraph g1 ["Problem Definition"]
        direction TB
        A(IDENTIFY MAIN FACTORS\\nFOR MINING METHOD SELECTION)
        A --> B(DESIGN ASSESSMENT CRITERIA)
        B --> C(ESTABLISH A\\nPERFORMANCE MATRIX)
    end
    subgraph g2 ["Matrix Processing"]
        direction TB
        D(NORMALIZE THE\\nDECISION MATRIX)
        D --> E(CALCULATE WEIGHTED\\nNORMALIZED MATRIX)
        E --> F(DETERMINE POSITIVE &\\nNEGATIVE IDEAL SOLUTIONS)
    end
    subgraph g3 ["Ranking"]
        direction TB
        G(CALCULATE\\nSEPARATION MEASURES)
        G --> H(CALCULATE RELATIVE\\nCLOSENESS TO IDEAL SOLUTION)
        H --> I(RANK THE\\nPREFERENCE ORDER)
    end
    C --> D
    F --> G""",
    },
}

# ---------------------------------------------------------------------------
# HTML template — %%TITLE%% and %%DIAGRAM%% are replaced at call time.
# Using .replace() instead of f-strings avoids escaping the JS curly braces.
# ---------------------------------------------------------------------------
_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 4px 2px; background: transparent; font-family: sans-serif; }
  h4 {
    font-size: 11px; color: #64748b; margin: 0 0 6px;
    text-transform: uppercase; letter-spacing: .08em; font-weight: 700;
  }
  .mermaid svg { max-width: 100%; display: block; overflow: visible; }
</style>
</head>
<body>
<h4>%%TITLE%%</h4>
<div class="mermaid">%%DIAGRAM%%</div>
<script>
mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  flowchart: {
    curve: 'basis',
    useMaxWidth: true,
    htmlLabels: true,
    padding: 24,
    nodeSpacing: 55,
    rankSpacing: 65
  },
  themeVariables: {
    primaryColor: '#1e3a8a',
    primaryBorderColor: '#3b82f6',
    primaryTextColor: '#ffffff',
    lineColor: '#3b82f6',
    secondaryColor: '#3730a3',
    tertiaryColor: '#0f172a',
    clusterBkg: 'rgba(15,23,42,0.05)',
    clusterBorder: '#64748b',
    edgeLabelBackground: '#f0f9ff',
    fontSize: '15px'
  }
});

mermaid.run({ querySelector: '.mermaid' }).then(function () {
  var svgs = document.querySelectorAll('.mermaid svg');
  svgs.forEach(function (svg, i) {

    /* ── 1. Inject gradient defs + drop-shadow filter ────────────────── */
    var defs = svg.querySelector('defs');
    if (!defs) {
      defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
      svg.insertBefore(defs, svg.firstChild);
    }
    defs.innerHTML += [
      /* process  (blue) */
      '<linearGradient id="gP' + i + '" x1="0%" y1="0%" x2="90%" y2="100%">',
        '<stop offset="0%" stop-color="#60a5fa"/>',
        '<stop offset="100%" stop-color="#1e3a8a"/>',
      '</linearGradient>',
      /* terminal (green) */
      '<linearGradient id="gT' + i + '" x1="0%" y1="0%" x2="90%" y2="100%">',
        '<stop offset="0%" stop-color="#4ade80"/>',
        '<stop offset="100%" stop-color="#14532d"/>',
      '</linearGradient>',
      /* I/O      (amber) */
      '<linearGradient id="gI' + i + '" x1="0%" y1="0%" x2="90%" y2="100%">',
        '<stop offset="0%" stop-color="#fbbf24"/>',
        '<stop offset="100%" stop-color="#78350f"/>',
      '</linearGradient>',
      /* concept  (purple) */
      '<linearGradient id="gC' + i + '" x1="0%" y1="0%" x2="90%" y2="100%">',
        '<stop offset="0%" stop-color="#a78bfa"/>',
        '<stop offset="100%" stop-color="#3730a3"/>',
      '</linearGradient>',
      /* drop-shadow filter */
      '<filter id="sh' + i + '" x="-30%" y="-30%" width="180%" height="180%">',
        '<feDropShadow dx="5" dy="6" stdDeviation="5" flood-color="rgba(0,0,0,0.42)"/>',
      '</filter>'
    ].join('');

    /* ── 2. Style parallelogram nodes (I/O) ──────────────────────────── */
    svg.querySelectorAll('.node polygon').forEach(function (el) {
      el.setAttribute('fill',   'url(#gI' + i + ')');
      el.setAttribute('filter', 'url(#sh' + i + ')');
      el.setAttribute('stroke', '#fbbf24');
      el.setAttribute('stroke-width', '2');
    });

    /* ── 3. Style rect nodes — detect type by rx ─────────────────────── */
    svg.querySelectorAll('.node rect').forEach(function (el) {
      var rx = parseFloat(el.getAttribute('rx') || '0');
      var grad, stroke;
      if (rx >= 20) {
        /* stadium shape — START / END terminal */
        grad = 'gT' + i;  stroke = '#4ade80';
      } else if (rx >= 6) {
        /* rounded rect — conceptual node */
        grad = 'gC' + i;  stroke = '#a78bfa';
      } else {
        /* plain rect — process / computation step */
        grad = 'gP' + i;  stroke = '#60a5fa';
      }
      el.setAttribute('fill',   'url(#' + grad + ')');
      el.setAttribute('filter', 'url(#sh' + i + ')');
      el.setAttribute('stroke', stroke);
      el.setAttribute('stroke-width', '2');
    });

    /* ── 4. White text on all node labels ────────────────────────────── */
    /* HTML labels (htmlLabels:true) */
    svg.querySelectorAll('.node foreignObject *').forEach(function (el) {
      el.style.color      = '#ffffff';
      el.style.fill       = '#ffffff';
      el.style.fontWeight = '700';
    });
    /* SVG text fallback */
    svg.querySelectorAll('.node text, .node tspan').forEach(function (el) {
      el.setAttribute('fill', '#ffffff');
      el.style.fill       = '#ffffff';
      el.style.fontWeight = '700';
    });

    /* ── 5. Subgraph (cluster) styling ───────────────────────────────── */
    svg.querySelectorAll('.cluster rect').forEach(function (el) {
      el.setAttribute('fill',             'rgba(15,23,42,0.06)');
      el.setAttribute('stroke',           '#475569');
      el.setAttribute('stroke-dasharray', '6,3');
      el.setAttribute('stroke-width',     '1.5');
      el.setAttribute('rx', '10');
    });
    svg.querySelectorAll('.cluster .cluster-label text, .cluster .cluster-label tspan').forEach(function (el) {
      el.setAttribute('fill', '#334155');
      el.style.fontWeight = '700';
      el.style.fontSize   = '11px';
    });

    /* ── 6. Edge paths & arrowheads ──────────────────────────────────── */
    svg.querySelectorAll('.edgePath .path').forEach(function (el) {
      el.setAttribute('stroke',       '#3b82f6');
      el.setAttribute('stroke-width', '2.5');
    });
    svg.querySelectorAll('.arrowheadPath').forEach(function (el) {
      el.setAttribute('fill',   '#3b82f6');
      el.setAttribute('stroke', 'none');
    });

  }); /* end svgs.forEach */
}); /* end mermaid.run().then */
</script>
</body>
</html>"""


# Conceptual-flow variant: natural (unscaled) width, centred in the iframe.
# TD diagrams are naturally narrow (~280-320 px); scaling them to full page width
# (useMaxWidth:true) inflates height by 4-5× making it unframeable. This variant
# disables that scaling and centres the SVG instead.
_CONCEPT_TEMPLATE = (
    _TEMPLATE
    .replace("useMaxWidth: true", "useMaxWidth: false")
    .replace(
        '<div class="mermaid">%%DIAGRAM%%</div>',
        '<div style="display:flex;justify-content:center;">'
        '<div class="mermaid">%%DIAGRAM%%</div>'
        "</div>",
    )
)


def _render_mermaid(diagram: str, title: str, height: int, concept: bool = False) -> None:
    template = _CONCEPT_TEMPLATE if concept else _TEMPLATE
    html = template.replace("%%TITLE%%", title).replace("%%DIAGRAM%%", diagram)
    # scrolling=True on concept as a safety net for TOPSIS's 9-node TD chain
    components.html(html, height=height, scrolling=concept)


def render_diagram_pair(method: str) -> None:
    """Render implementation flow (full-width) then conceptual flow (centred, natural width)."""
    entry = _DIAGRAMS.get(method)
    if entry is None:
        return
    heights = _HEIGHTS[method]
    _render_mermaid(entry["impl"],    "Implementation Flow", heights["impl"])
    _render_mermaid(entry["concept"], "Conceptual Flow",     heights["concept"], concept=True)
