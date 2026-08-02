#!/usr/bin/env python3
"""
Generate a scalable Three.js gallery from ASCII mazes in samples.txt.

Unlike the earlier version, this generator uses:
    - ONE WebGLRenderer for the entire page
    - one independent scene and camera per maze
    - scissor/viewport rendering for visible maze cards only

This avoids the browser WebGL-context limit that causes older cards to
turn blank when displaying many mazes.

Usage:
    python generate_maze_gallery.py samples.txt
    python generate_maze_gallery.py samples.txt -o maze_gallery.html
    python generate_maze_gallery.py samples.txt -o maze_gallery.html --columns 8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SAMPLE_RE = re.compile(
    r"===== Sample\s+(\d+)\s*/\s*(\d+)\s*=====\s*"
    r"```[^\n]*\n"
    r"(.*?)"
    r"\n```",
    re.DOTALL,
)


def extract_samples(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for match in SAMPLE_RE.finditer(text):
        samples.append(
            {
                "number": int(match.group(1)),
                "total": int(match.group(2)),
                "maze": match.group(3).rstrip(),
            }
        )

    if not samples:
        raise ValueError(
            "No samples were found. Expected blocks in this format:\n\n"
            "===== Sample 1/100 =====\n"
            "```\n"
            "...maze...\n"
            "```"
        )

    samples.sort(key=lambda item: item["number"])
    return samples


def generate_html(samples: list[dict[str, Any]], columns: int) -> str:
    samples_json = json.dumps(samples, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ASCII Maze Gallery</title>

  <style>
    * {{
      box-sizing: border-box;
    }}

    html,
    body {{
      margin: 0;
      min-height: 100%;
      background: #0f141d;
      color: #ffffff;
      font-family: Inter, Arial, sans-serif;
    }}

    body {{
      padding: 14px;
    }}

    #sharedCanvas {{
      position: fixed;
      inset: 0;
      z-index: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }}

    .page {{
      position: relative;
      z-index: 1;
    }}

    .header {{
      margin-bottom: 12px;
    }}

    .header h1 {{
      margin: 0 0 4px;
      font-size: 20px;
    }}

    .header p {{
      margin: 0;
      color: #b9c4d3;
      font-size: 13px;
    }}

    .maze-grid {{
      display: grid;
      grid-template-columns: repeat({columns}, minmax(0, 1fr));
      gap: 8px;
    }}

    .maze-card {{
      min-width: 0;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 9px;
      background: transparent;
    }}

    .maze-title,
    .maze-status {{
      background: #171e29;
    }}

    .maze-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      padding: 6px 8px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      font-size: 11px;
      font-weight: 700;
    }}

    .rotation {{
      color: #91a0b4;
      font-size: 9px;
      font-weight: 500;
      white-space: nowrap;
    }}

    .maze-view {{
      position: relative;
      width: 100%;
      min-height: 112px;
      aspect-ratio: 4 / 3;
      background: transparent;
      cursor: default;
      touch-action: none;
    }}

    .maze-status {{
      display: flex;
      gap: 8px;
      min-height: 22px;
      padding: 5px 8px 7px;
      color: #cbd5e1;
      font-size: 9px;
    }}

    .status-item {{
      white-space: nowrap;
    }}

    .missing {{
      color: #f5b7be;
    }}

    .dot {{
      display: inline-block;
      width: 6px;
      height: 6px;
      margin-right: 4px;
      border-radius: 50%;
    }}

    .start-dot {{
      background: #39e879;
    }}

    .exit-dot {{
      background: #ff5267;
    }}

    @media (max-width: 1700px) {{
      .maze-grid {{
        grid-template-columns: repeat(6, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 1300px) {{
      .maze-grid {{
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 900px) {{
      .maze-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 650px) {{
      .maze-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 420px) {{
      .maze-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>

  <script type="importmap">
  {{
    "imports": {{
      "three": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"
    }}
  }}
  </script>
</head>

<body>
  <canvas id="sharedCanvas"></canvas>

  <div class="page">
    <header class="header">
      <h1>ASCII Maze Samples</h1>
      <p>
        {len(samples)} mazes · {columns} columns · one shared WebGL renderer ·
        orthographic top view · slow automatic centre rotation
      </p>
    </header>

    <main id="mazeGrid" class="maze-grid"></main>
  </div>

  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

    const SAMPLE_DATA = {samples_json};

    const COLS = 6;
    const ROWS = 4;
    const CELL = 1.45;
    const WALL_HEIGHT = 1.15;
    const WALL_THICKNESS = 0.32;

    const MAZE_WIDTH = COLS * CELL;
    const MAZE_DEPTH = ROWS * CELL;
    const ORTHO_HALF_HEIGHT = 6.7;

    // Slow automatic in-plane rotation for every maze.
    const AUTO_ROTATE_MIN_DPS = 4.0;
    const AUTO_ROTATE_MAX_DPS = 7.0;

    const canvas = document.getElementById("sharedCanvas");

    const renderer = new THREE.WebGLRenderer({{
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance"
    }});

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.autoClear = false;

    const viewers = [];

    // Shared geometry and materials keep memory and draw-call overhead low.
    const horizontalWallGeometry = new THREE.BoxGeometry(
      CELL + WALL_THICKNESS,
      WALL_HEIGHT,
      WALL_THICKNESS
    );

    const verticalWallGeometry = new THREE.BoxGeometry(
      WALL_THICKNESS,
      WALL_HEIGHT,
      CELL + WALL_THICKNESS
    );

    const floorGeometry = new THREE.BoxGeometry(
      MAZE_WIDTH + 0.10,
      0.14,
      MAZE_DEPTH + 0.10
    );

    const baseGeometry = new THREE.BoxGeometry(
      MAZE_WIDTH + 0.72,
      0.28,
      MAZE_DEPTH + 0.72
    );

    const markerRingGeometry = new THREE.TorusGeometry(
      0.27,
      0.065,
      16,
      40
    );

    const markerOrbGeometry = new THREE.SphereGeometry(
      0.12,
      18,
      18
    );

    const wallMaterial = new THREE.MeshStandardMaterial({{
      color: 0xdfe8f4,
      roughness: 0.72,
      metalness: 0.02
    }});

    const floorMaterial = new THREE.MeshStandardMaterial({{
      color: 0x2b384b,
      roughness: 0.92
    }});

    const baseMaterial = new THREE.MeshStandardMaterial({{
      color: 0x161d27,
      roughness: 0.96
    }});

    const startMarkerMaterial = new THREE.MeshStandardMaterial({{
      color: 0x39e879,
      emissive: 0x39e879,
      emissiveIntensity: 1.35
    }});

    const exitMarkerMaterial = new THREE.MeshStandardMaterial({{
      color: 0xff5267,
      emissive: 0xff5267,
      emissiveIntensity: 1.35
    }});

    const identityMatrix = new THREE.Matrix4();

    function localX(gridX) {{
      return (gridX - COLS / 2) * CELL;
    }}

    function localZ(gridY) {{
      return (gridY - ROWS / 2) * CELL;
    }}

    function parseMaze(mazeText) {{
      const lines = mazeText
        .split("\\n")
        .map((line) => line.replace(/\\r$/, ""));

      const borderLines = lines.filter((line) => line.includes("+"));
      const cellLines = lines.filter((line) => /^\\s*\\d+\\s*\\|/.test(line));

      if (borderLines.length < 5 || cellLines.length < 4) {{
        throw new Error(
          `Expected 5 border lines and 4 cell lines, received ` +
          `${{borderLines.length}} and ${{cellLines.length}}.`
        );
      }}

      const h = Array.from({{ length: 5 }}, () => Array(6).fill(0));
      const v = Array.from({{ length: 4 }}, () => Array(7).fill(0));

      let start = null;
      let exit = null;

      for (let boundary = 0; boundary < 5; boundary += 1) {{
        const fullLine = borderLines[boundary];
        const firstPlus = fullLine.indexOf("+");
        const mazePart = firstPlus >= 0
          ? fullLine.slice(firstPlus)
          : fullLine;

        for (let col = 0; col < 6; col += 1) {{
          const segment = mazePart.slice(
            col * 4 + 1,
            col * 4 + 4
          );

          h[boundary][col] = segment.includes("-") ? 1 : 0;
        }}
      }}

      for (let row = 0; row < 4; row += 1) {{
        const fullLine = cellLines[row];
        const firstPipe = fullLine.indexOf("|");
        const mazePart = firstPipe >= 0
          ? fullLine.slice(firstPipe)
          : fullLine;

        for (let boundary = 0; boundary < 7; boundary += 1) {{
          v[row][boundary] =
            mazePart[boundary * 4] === "|" ? 1 : 0;
        }}

        for (let col = 0; col < 6; col += 1) {{
          const content = mazePart.slice(
            col * 4 + 1,
            col * 4 + 4
          );

          if (content.includes("S")) {{
            start = {{ col, row }};
          }}

          if (content.includes("E")) {{
            exit = {{ col, row }};
          }}
        }}

        // Preserve malformed model outputs that place S or E to the right
        // of the nominal six-column boundary.
        const overflow = mazePart.slice(24);

        if (!start && overflow.includes("S")) {{
          start = {{ col: 6.15, row }};
        }}

        if (!exit && overflow.includes("E")) {{
          exit = {{ col: 6.15, row }};
        }}
      }}

      return {{ h, v, start, exit }};
    }}

    function createLabelTexture(text) {{
      const labelCanvas = document.createElement("canvas");
      labelCanvas.width = 256;
      labelCanvas.height = 128;

      const context = labelCanvas.getContext("2d");
      context.clearRect(0, 0, 256, 128);
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.font = "bold 82px Arial";
      context.fillStyle = "#ffffff";
      context.shadowColor = "rgba(0,0,0,.9)";
      context.shadowBlur = 12;
      context.fillText(text, 128, 64);

      const texture = new THREE.CanvasTexture(labelCanvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      return texture;
    }}

    const startLabelMaterial = new THREE.SpriteMaterial({{
      map: createLabelTexture("S"),
      transparent: true,
      depthTest: false
    }});

    const exitLabelMaterial = new THREE.SpriteMaterial({{
      map: createLabelTexture("E"),
      transparent: true,
      depthTest: false
    }});

    function createMarker(group, marker, type) {{
      if (!marker) {{
        return null;
      }}

      const isStart = type === "start";
      const material = isStart
        ? startMarkerMaterial
        : exitMarkerMaterial;

      const labelMaterial = isStart
        ? startLabelMaterial
        : exitLabelMaterial;

      const markerGroup = new THREE.Group();

      markerGroup.position.set(
        localX(marker.col + 0.5),
        0,
        localZ(marker.row + 0.5)
      );

      const ring = new THREE.Mesh(
        markerRingGeometry,
        material
      );

      ring.rotation.x = Math.PI / 2;
      ring.position.y = 0.06;
      markerGroup.add(ring);

      const orb = new THREE.Mesh(
        markerOrbGeometry,
        material
      );

      orb.position.y = 0.29;
      markerGroup.add(orb);

      const label = new THREE.Sprite(labelMaterial);
      label.position.y = 0.86;
      label.scale.set(0.72, 0.36, 1);
      markerGroup.add(label);

      group.add(markerGroup);
      return orb;
    }}

    function createWallInstances(parsed, group) {{
      const horizontalPositions = [];
      const verticalPositions = [];

      parsed.h.forEach((row, boundary) => {{
        row.forEach((hasWall, col) => {{
          if (hasWall) {{
            horizontalPositions.push([
              localX(col + 0.5),
              WALL_HEIGHT / 2,
              localZ(boundary)
            ]);
          }}
        }});
      }});

      parsed.v.forEach((row, rowIndex) => {{
        row.forEach((hasWall, boundary) => {{
          if (hasWall) {{
            verticalPositions.push([
              localX(boundary),
              WALL_HEIGHT / 2,
              localZ(rowIndex + 0.5)
            ]);
          }}
        }});
      }});

      if (horizontalPositions.length > 0) {{
        const horizontalWalls = new THREE.InstancedMesh(
          horizontalWallGeometry,
          wallMaterial,
          horizontalPositions.length
        );

        horizontalWalls.castShadow = true;
        horizontalWalls.receiveShadow = true;

        horizontalPositions.forEach((position, index) => {{
          identityMatrix.makeTranslation(...position);
          horizontalWalls.setMatrixAt(index, identityMatrix);
        }});

        horizontalWalls.instanceMatrix.needsUpdate = true;
        group.add(horizontalWalls);
      }}

      if (verticalPositions.length > 0) {{
        const verticalWalls = new THREE.InstancedMesh(
          verticalWallGeometry,
          wallMaterial,
          verticalPositions.length
        );

        verticalWalls.castShadow = true;
        verticalWalls.receiveShadow = true;

        verticalPositions.forEach((position, index) => {{
          identityMatrix.makeTranslation(...position);
          verticalWalls.setMatrixAt(index, identityMatrix);
        }});

        verticalWalls.instanceMatrix.needsUpdate = true;
        group.add(verticalWalls);
      }}
    }}

    function createMazeObject(parsed) {{
      const group = new THREE.Group();

      const base = new THREE.Mesh(
        baseGeometry,
        baseMaterial
      );

      base.position.y = -0.21;
      base.receiveShadow = true;
      group.add(base);

      const floor = new THREE.Mesh(
        floorGeometry,
        floorMaterial
      );

      floor.position.y = -0.07;
      floor.receiveShadow = true;
      group.add(floor);

      createWallInstances(parsed, group);

      const startOrb = createMarker(
        group,
        parsed.start,
        "start"
      );

      const exitOrb = createMarker(
        group,
        parsed.exit,
        "exit"
      );

      return {{
        group,
        startOrb,
        exitOrb
      }};
    }}

    function deterministicFraction(seed) {{
      const value = Math.sin(seed * 12.9898) * 43758.5453;
      return value - Math.floor(value);
    }}

    function deterministicAngle(sampleNumber) {{
      // Stable pseudo-random starting angle in the range [-45, 45].
      return -45 + deterministicFraction(sampleNumber) * 90;
    }}

    function deterministicRotationSpeed(sampleNumber) {{
      const speedDps =
        AUTO_ROTATE_MIN_DPS +
        deterministicFraction(sampleNumber + 1000) *
          (AUTO_ROTATE_MAX_DPS - AUTO_ROTATE_MIN_DPS);

      // Alternate clockwise and anticlockwise rotation.
      const direction = sampleNumber % 2 === 0 ? -1 : 1;

      return THREE.MathUtils.degToRad(speedDps * direction);
    }}

    function updateOrthographicCamera(camera, aspect) {{
      const halfHeight = ORTHO_HALF_HEIGHT;
      const halfWidth = halfHeight * Math.max(aspect, 0.01);

      camera.left = -halfWidth;
      camera.right = halfWidth;
      camera.top = halfHeight;
      camera.bottom = -halfHeight;
      camera.updateProjectionMatrix();
    }}

    function makeStatus(parsed) {{
      const startStatus = parsed.start
        ? `<span class="status-item"><i class="dot start-dot"></i>Start</span>`
        : `<span class="status-item missing">S missing</span>`;

      const exitStatus = parsed.exit
        ? `<span class="status-item"><i class="dot exit-dot"></i>Exit</span>`
        : `<span class="status-item missing">E missing</span>`;

      return startStatus + exitStatus;
    }}

    function buildViewer(sample, index) {{
      let parsed;

      try {{
        parsed = parseMaze(sample.maze);
      }} catch (error) {{
        console.error(
          `Failed to parse sample ${{sample.number}}`,
          error
        );

        const errorCard = document.createElement("section");
        errorCard.className = "maze-card";
        errorCard.innerHTML = `
          <div class="maze-title">
            <span>Sample ${{sample.number}}/${{sample.total}}</span>
          </div>
          <div class="maze-view"></div>
          <div class="maze-status">
            <span class="missing">Parse error</span>
          </div>
        `;
        document.getElementById("mazeGrid").appendChild(errorCard);
        return;
      }}

      const angleDegrees = deterministicAngle(sample.number);
      const rotationSpeed = deterministicRotationSpeed(sample.number);

      const card = document.createElement("section");
      card.className = "maze-card";

      const title = document.createElement("div");
      title.className = "maze-title";
      title.innerHTML = `
        <span>Sample ${{sample.number}}/${{sample.total}}</span>
        <span class="rotation">
          ${{rotationSpeed >= 0 ? "↻" : "↺"}}
          ${{Math.abs(
            THREE.MathUtils.radToDeg(rotationSpeed)
          ).toFixed(1)}}°/s
        </span>
      `;

      const view = document.createElement("div");
      view.className = "maze-view";

      const status = document.createElement("div");
      status.className = "maze-status";
      status.innerHTML = makeStatus(parsed);

      card.append(title, view, status);
      document.getElementById("mazeGrid").appendChild(card);

      const scene = new THREE.Scene();

      const camera = new THREE.OrthographicCamera(
        -ORTHO_HALF_HEIGHT,
        ORTHO_HALF_HEIGHT,
        ORTHO_HALF_HEIGHT,
        -ORTHO_HALF_HEIGHT,
        0.1,
        100
      );

      camera.position.set(0, 20, 0.001);
      camera.up.set(0, 0, -1);
      camera.lookAt(0, 0, 0);

      const controls = new OrbitControls(camera, view);
      controls.target.set(0, 0, 0);
      controls.enableDamping = true;
      controls.enablePan = false;
      controls.enableRotate = false;
      controls.enableZoom = true;
      controls.minZoom = 0.65;
      controls.maxZoom = 2.5;

      scene.add(
        new THREE.HemisphereLight(
          0xc8ddff,
          0x20242b,
          2.2
        )
      );

      const keyLight = new THREE.DirectionalLight(
        0xffffff,
        3.0
      );

      keyLight.position.set(8, 13, 9);
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.set(512, 512);
      keyLight.shadow.camera.left = -9;
      keyLight.shadow.camera.right = 9;
      keyLight.shadow.camera.top = 9;
      keyLight.shadow.camera.bottom = -9;
      scene.add(keyLight);

      const maze = createMazeObject(parsed);
      maze.group.rotation.y = THREE.MathUtils.degToRad(
        angleDegrees
      );

      scene.add(maze.group);

      viewers.push({{
        element: view,
        scene,
        camera,
        controls,
        startOrb: maze.startOrb,
        exitOrb: maze.exitOrb,
        mazeGroup: maze.group,
        initialRotation: THREE.MathUtils.degToRad(angleDegrees),
        rotationSpeed,
        index
      }});
    }}

    SAMPLE_DATA.forEach(buildViewer);

    function resizeRendererToDisplaySize() {{
      const width = window.innerWidth;
      const height = window.innerHeight;

      const expectedWidth = Math.floor(
        width * renderer.getPixelRatio()
      );

      const expectedHeight = Math.floor(
        height * renderer.getPixelRatio()
      );

      if (
        canvas.width !== expectedWidth ||
        canvas.height !== expectedHeight
      ) {{
        renderer.setSize(width, height, false);
      }}
    }}

    function renderViewer(viewer, elapsed) {{
      const rect = viewer.element.getBoundingClientRect();

      if (
        rect.bottom <= 0 ||
        rect.top >= window.innerHeight ||
        rect.right <= 0 ||
        rect.left >= window.innerWidth ||
        rect.width <= 0 ||
        rect.height <= 0
      ) {{
        return;
      }}

      const left = Math.max(0, rect.left);
      const right = Math.min(window.innerWidth, rect.right);
      const top = Math.max(0, rect.top);
      const bottom = Math.min(window.innerHeight, rect.bottom);

      const width = Math.max(1, right - left);
      const height = Math.max(1, bottom - top);

      const viewportBottom = window.innerHeight - bottom;

      renderer.setViewport(
        left,
        viewportBottom,
        width,
        height
      );

      renderer.setScissor(
        left,
        viewportBottom,
        width,
        height
      );

      renderer.setClearColor(0x111823, 1);
      renderer.clear(true, true, true);

      updateOrthographicCamera(
        viewer.camera,
        rect.width / rect.height
      );

      // Rotate around the maze group's local origin, which is its centre.
      // Elapsed-time rotation keeps motion continuous after scrolling off-screen.
      viewer.mazeGroup.rotation.y =
        viewer.initialRotation +
        elapsed * viewer.rotationSpeed;

      if (viewer.startOrb) {{
        viewer.startOrb.position.y =
          0.29 +
          Math.sin(
            elapsed * 2.4 +
            viewer.index * 0.2
          ) * 0.05;
      }}

      if (viewer.exitOrb) {{
        viewer.exitOrb.position.y =
          0.29 +
          Math.sin(
            elapsed * 2.4 +
            viewer.index * 0.2 +
            Math.PI
          ) * 0.05;
      }}

      viewer.controls.update();
      renderer.render(
        viewer.scene,
        viewer.camera
      );
    }}

    const clock = new THREE.Clock();

    function animate() {{
      resizeRendererToDisplaySize();

      renderer.setScissorTest(false);
      renderer.setClearColor(0x000000, 0);
      renderer.clear(true, true, true);
      renderer.setScissorTest(true);

      const elapsed = clock.getElapsedTime();

      viewers.forEach((viewer) => {{
        renderViewer(viewer, elapsed);
      }});

      requestAnimationFrame(animate);
    }}

    animate();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a scalable Three.js maze gallery using one "
            "shared WebGL renderer."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to samples.txt",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("maze_gallery.html"),
        help="Output HTML file. Default: maze_gallery.html",
    )

    parser.add_argument(
        "--columns",
        type=int,
        default=8,
        help="Number of columns on wide screens. Default: 8",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.columns < 1:
        raise SystemExit("--columns must be at least 1")

    if not args.input.is_file():
        raise SystemExit(f"Input file does not exist: {args.input}")

    text = args.input.read_text(encoding="utf-8")
    samples = extract_samples(text)
    output_html = generate_html(samples, args.columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_html, encoding="utf-8")

    print(f"Extracted {len(samples)} maze samples")
    print("Renderer mode: one shared WebGL context")
    print(f"Generated: {args.output.resolve()}")


if __name__ == "__main__":
    main()
