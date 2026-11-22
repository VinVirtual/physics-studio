---
name: physics-studio
description: Turn a robot (URDF or MJCF) and a plain-language task into a narrated physics simulation clip, made on this machine with MuJoCo, for free. Writes a scene recipe (table, objects, drop, push, pick-and-place with scripted inverse kinematics, pendulum, cart-pole), builds the MJCF scene, runs it deterministically with a fixed timestep, draws contact-force arrows, centre of mass, motor-effort bars and a time/step counter, voices a script written from the run log with Kokoro, and encodes an MP4 (landscape, vertical or square) with captions. Also writes the generated MJCF, a JSON log of the run (contacts, peak forces, events, pass or fail) and a WebVTT caption file. Use whenever the user wants a physics simulation video, a robot "picking up a cube" or "pushing a block" clip, a MuJoCo demo, a pendulum or cart-pole animation, a drop test, contact forces shown on a video, a narrated sim for a talk, a lesson or social media, or wants to check whether a robot arm can reach and place something.
---

# physics-studio

Recipe (JSON) -> **MJCF scene** -> **MuJoCo** run (fixed timestep, seeded, CPU) -> run **log** -> narration script **from the log** -> **Kokoro** voice and word timings -> frames rendered offscreen (EGL or OSMesa) with **overlays** -> **ffmpeg** MP4 + VTT.

Skill folder (written `$P` below): this folder. Every command is `$P/physics-studio <command> ...`; `$P/physics-studio help` lists them. `README.md` documents every recipe field and the log format.

## Workflow

1. `$P/check_env.sh`: MuJoCo, offscreen OpenGL, ffmpeg, Kokoro. Fix every MISS line. With no Kokoro, `--no-voice` still makes a captioned video.
2. `$P/physics-studio new <dir> --task <pick_place|push|drop|pendulum|cartpole>`: a working recipe to edit. Arm tasks also copy `robot/tern.urdf`, a small five-joint arm with a two-finger gripper written for this skill.
3. **Turn the user's words into the recipe** (rules below). For the user's own robot, put its URDF or MJCF and meshes in `<dir>/robot/` and run `$P/physics-studio inspect <dir>/robot/<file>` to read the real joint names, limits and body names. Use only those names.
4. `$P/physics-studio validate <dir>/recipe.json`. Fix every problem it lists (it lists all of them at once).
5. `$P/physics-studio sim <dir>/recipe.json` (seconds). Read `out/<name>.log.json`: `task.success` and `task.reason`, `events`, `plan.unreachable` (IK waypoints the arm could not reach), `warnings`, `actuators[].saturated_pct`. If the task failed because the recipe is wrong (object out of reach, gripper too narrow), fix the recipe and run again. If it failed for a physical reason, that is a result: keep it and let the narration say so. Never tune a recipe until a failure "passes" and then present it as typical.
6. `$P/physics-studio sheet <dir>/recipe.json` and **Read the PNG** (`out/<name>-sheet.png`). Check: the robot and objects are in frame, the arrows point at the contacts, the panels do not cover the action, captions are readable. Adjust `camera` (`azimuth`, `elevation`, `distance`, `lookat`) and look again. Do not say it looks right without viewing it.
7. `$P/physics-studio build <dir>/recipe.json` (about a minute for 15 to 20 seconds of 720p video). `--format vertical|square` for other shapes, `--no-voice` for captions only.
8. Report: paths of the MP4, poster, VTT, scene and log; the result and reason from the log; the playback speed it chose.

## From plain language to a recipe

- "pick up the cube and put it on the left" -> `pick_place`, a box object, `goal.place` in metres (x forward from the robot, y to its left, z up). Heights are measured from the surface below: `pos: [x, y, 0]` puts an object on the table.
- "push the block 15 cm to the left" -> `push`, `goal.direction_deg` (0 = +x, 90 = +y) and `goal.distance`.
- "drop three things on a table" -> `drop`, objects with `pos[2]` as the drop height.
- "a pendulum from 40 degrees" / "a double pendulum" -> `pendulum` (`links` 1 or 2). "balance a pole on a cart and shove it" -> `cartpole` with `push`.
- The arm needs `end_effector` (a body near the gripper and the offset of the grasp point in that body's frame), and for `pick_place` a `gripper` with `open` and `closed` joint values. `closed` should be a little past touching so the fingers squeeze; `force` caps the squeeze.
- Keep objects inside the arm's reach (Tern reaches about 0.4 m from its base) and on the table. `jitter` adds seeded randomness to start positions; same `seed`, same run.
- Anything the user did not say: leave the defaults.

## Narration rules

- Leave `"narration": "auto"` unless the user wants their own words. The auto script is written from the log after the run and only states what happened: grasp time, finger force, lift height, final error, impact speed, measured period, pass or fail.
- Custom lines may only use numbers through placeholders from the log, such as `{task.final_error_cm}`, `{events.grasp:1}`, `{contacts.max_normal_N:0}`. Never type a measured value into a line by hand, and never claim anything the log does not show. A line tied to an event that did not happen (`"at": "event:grasp"`) is left out automatically and listed in `out/<name>.script.json` under `skipped`.
- `"video": {"speed": "auto"}` (the default) slows playback (1x, 0.8x ... 0.25x) until every line starts within 1.5 s of the event it describes. The HUD shows the speed.

## Reporting back honestly

- Say it is a simulation: MuJoCo with soft contacts, a fixed timestep and the masses, frictions and motor limits in the files. It is not a claim about a real robot. The video carries a "SIMULATION" label.
- Peak contact forces are single-step values and depend on the timestep and contact softness; say "in this simulation".
- You cannot hear the audio. Check it numerically (`ffmpeg -i out/<name>.mp4 -af volumedetect -f null -`) and say you did.
- If the robot model is someone else's, check its licence allows a published video.

## Troubleshooting

- **offscreen rendering MISS**: install Mesa EGL (`apt install libegl1 libgl1-mesa-dri`) or OSMesa (`apt install libosmesa6`) and `export MUJOCO_GL=egl` (or `osmesa`). No GPU is needed.
- **`plan.unreachable` is not empty**: the target or a waypoint is out of reach, or the gripper cannot point straight down there. Move the object or the place target closer, or raise `end_effector.offset` to match the real grasp point.
- **Grasp never happens**: `gripper.closed` does not squeeze (fingers stop before the object), the object is wider than `open`, or the grasp point is too high. Look at the sheet at the `close` event.
- **The object slips**: raise `gripper.force` or the object's `friction`, or lower its mass.
- **"the second run differs from the first"**: the run is not deterministic on this machine; report it, do not hide it.
- **Robot file refused**: DOCTYPE/ENTITY, `<include>`, `<plugin>`, paths with `..` or outside the robot's folder, and mesh types other than STL, OBJ, MSH are refused on purpose. Xacro must be expanded first.
