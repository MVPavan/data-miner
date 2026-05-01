"""Render :file:`labeling_config.xml` from a dataset's ``classes.txt``.

Single source of truth: the dataset's ``classes.txt`` (one class per line).
Hard-coding the palette in the XML drifted on every dataset switch — this
reads the file at build time and substitutes a 4-block label palette
(bbox + smart_click + smart_visual + smart_track) plus the page chrome.

Hotkey map: 1-0, q-w-e-r-t-y-u-i-o-p, a-s-d, **skipping "v"** (reserved
by the smart_visual toggle) and "f" (reserved as a future fallback).
That's 23 keys; classes beyond that get no hotkey (UI click still works).

Run::

    python -m manual_reviewer.configs.build_labeling_config \\
        --classes /path/to/classes.txt \\
        --out manual_reviewer/configs/labeling_config.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

# 23 distinct hex colors — one per class slot. Cycles if the dataset has more.
_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22",
    "#f1c40f", "#1abc9c", "#34495e", "#7f8c8d", "#c0392b",
    "#2980b9", "#27ae60", "#8e44ad", "#d35400", "#f39c12",
    "#16a085", "#2c3e50", "#95a5a6", "#e84393", "#fd79a8",
    "#a29bfe", "#fdcb6e", "#74b9ff",
]

# 23 hotkeys, "v" and "f" intentionally absent.
_HOTKEYS = list("1234567890qwertyuiopasd")

_TEMPLATE = """\
<!--
  Label Studio labeling interface for the aa_v4 manual review system.

  GENERATED FROM {classes_path}. Do not hand-edit; rerun
  ``python -m manual_reviewer.configs.build_labeling_config`` after
  the dataset's classes.txt changes.

  Phase A: <RectangleLabels name="bbox"> drives the rectangle tool and
  carries the class palette. <KeyPointLabels name="click"> mirrors the
  same palette for the smart_click tool so the reviewer's hotkey class
  rides every keypoint draft as value.keypointlabels.

  KeyPoint, TextArea and the V-tool Rectangle are gated by `smart="true"`
  so the ML backend handles those interactions.
-->
<View>
  <Header value="aa_v4 Manual Review"/>

  <View style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
    <Header value="Auto:"/>
    <Choices name="auto_tool" toName="image" choice="single" showInline="true">
      <Choice value="none"         hotkey="escape" selected="true"/>
      <Choice value="smart_visual" hotkey="shift+v"/>
      <Choice value="smart_click"  hotkey="shift+c"/>
      <Choice value="smart_search" hotkey="shift+s"/>
      <Choice value="smart_track"  hotkey="shift+t"/>
    </Choices>
  </View>

  <Image name="image" value="$image" zoom="true" zoomControl="true"
         rotateControl="false" brightnessControl="false"
         maxHeight="78vh"/>

  <View visibleWhen="choice-selected" whenTagName="auto_tool" whenChoiceValue="none">
  <RectangleLabels name="bbox" toName="image" opacity="0.05" strokeWidth="2">
{bbox_labels}
  </RectangleLabels>
  </View>

  <View visibleWhen="choice-selected" whenTagName="auto_tool" whenChoiceValue="smart_click">
  <KeyPointLabels name="click" toName="image" smart="true" smartOnly="true" strokeWidth="3">
{click_labels}
  </KeyPointLabels>
  </View>

  <View visibleWhen="choice-selected" whenTagName="auto_tool" whenChoiceValue="smart_visual">
  <RectangleLabels name="smart_visual" toName="image"
                   smart="true" smartOnly="true"
                   strokeColor="#ff7700" opacity="0.05" strokeWidth="2">
{visual_labels}
  </RectangleLabels>
  </View>

  <View visibleWhen="choice-selected" whenTagName="auto_tool" whenChoiceValue="smart_track">
  <RectangleLabels name="smart_track" toName="image"
                   smart="true" smartOnly="true"
                   strokeColor="#16a085" opacity="0.05" strokeWidth="2">
{track_labels}
  </RectangleLabels>
  </View>

  <View style="display: flex; gap: 16px; margin-top: 8px;">
    <View style="flex: 0 0 50%;">
      <View visibleWhen="choice-selected" whenTagName="auto_tool" whenChoiceValue="smart_search">
        <TextArea name="text_query" toName="image" smart="true"
                  editable="true"
                  placeholder="text prompt — e.g. 'forklift'"/>
      </View>

      <TextArea name="notes" toName="image" rows="2" maxSubmissions="1"
                placeholder="Notes for this image (optional)"/>
    </View>

    <View style="flex: 0 0 50%;">
      <Choices name="frame_state" toName="image" choice="single" required="true"
               showInline="true">
        <Choice value="clean" selected="true"/>
        <Choice value="needs_more_review"/>
        <Choice value="ambiguous_skip"/>
      </Choices>

      <TextArea name="track_id" toName="image" perRegion="true"
                placeholder="track_id" editable="true"/>
    </View>
  </View>
</View>
"""


def _read_classes(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    classes = [line.strip() for line in text.splitlines()]
    return [c for c in classes if c]


def _label_block(classes: list[str], indent: str = "    ") -> str:
    lines: list[str] = []
    for idx, name in enumerate(classes):
        color = _PALETTE[idx % len(_PALETTE)]
        hotkey = _HOTKEYS[idx] if idx < len(_HOTKEYS) else None
        attrs = f'value="{name}" background="{color}"'
        if hotkey is not None:
            attrs += f' hotkey="{hotkey}"'
        lines.append(f'{indent}<Label {attrs}/>')
    return "\n".join(lines)


def render(classes_path: Path) -> str:
    classes = _read_classes(classes_path)
    if not classes:
        raise ValueError(f"no classes parsed from {classes_path}")
    block = _label_block(classes)
    return _TEMPLATE.format(
        classes_path=classes_path,
        bbox_labels=block,
        click_labels=block,
        visual_labels=block,
        track_labels=block,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", required=True, type=Path,
                    help="classes.txt — one class name per line")
    ap.add_argument("--out", required=True, type=Path,
                    help="output XML path")
    args = ap.parse_args()

    xml = render(args.classes)
    args.out.write_text(xml, encoding="utf-8")
    n = len(_read_classes(args.classes))
    print(f"wrote {args.out} ({n} classes from {args.classes})")


if __name__ == "__main__":
    main()
