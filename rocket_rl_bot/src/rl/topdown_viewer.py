from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List

import numpy as np
import pygame

FIELD_X = 4096.0
FIELD_Y = 5120.0
GOAL_DEPTH = 880.0
GOAL_HALF_WIDTH = 893.0
GOAL_WIDTH = GOAL_HALF_WIDTH * 2.0
CORNER_OFFSET = 1152.0
SIDE_WALL_Y = FIELD_Y - CORNER_OFFSET
GOAL_BACK_Y = FIELD_Y + GOAL_DEPTH
FIELD_MARGIN = 28
PANEL_WIDTH = 360
WINDOW_WIDTH = 1480
WINDOW_HEIGHT = 900
TOTAL_HALF_Y = GOAL_BACK_Y

TEAM_COLORS = {
    0: (65, 155, 255),
    1: (255, 145, 60),
}


class TopDown2DViewer:
    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT, title: str = "Rocket RL Top Down") -> None:
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.field_width = width - PANEL_WIDTH
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 24, bold=True)
        self.car_trails: Dict[int, Deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=60))
        self.ball_trail: Deque[tuple[int, int]] = deque(maxlen=60)

    def draw(self, frame: Dict[str, Any], fps: int = 15) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill((15, 18, 24))
        self._draw_field(frame)
        self._draw_side_panel(frame)
        pygame.display.flip()
        self.clock.tick(max(1, fps))
        return True

    def close(self) -> None:
        pygame.display.quit()
        pygame.font.quit()
        pygame.quit()

    def _draw_field(self, frame: Dict[str, Any]) -> None:
        canvas_rect = pygame.Rect(0, 0, self.field_width, self.height)
        self._draw_field_background(canvas_rect)
        arena_rect = self._compute_arena_rect(canvas_rect)

        self._draw_soccar_surface(arena_rect)
        self._draw_reference_lines(arena_rect)
        self._draw_goals(arena_rect)
        self._draw_boost_pads(frame.get("boost_pads", []), arena_rect)
        self._draw_ball(frame.get("ball", {}), arena_rect)
        self._draw_cars(frame.get("cars", []), arena_rect)

        score = frame.get("score", {})
        scoreboard = f"Blue {score.get('blue', 0)} - {score.get('orange', 0)} Orange"
        self._blit_text(self.title_font, scoreboard, (20, 18), (245, 248, 252))

        meta = frame.get("meta", {})
        if meta:
            progress = f"Match {int(meta.get('completed_matches', 0))}/{int(meta.get('target_matches', 0))}"
            self._blit_text(self.font, progress, (20, 50), (226, 230, 235))

    def _draw_field_background(self, canvas_rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (10, 42, 39), canvas_rect)
        glow_rect = canvas_rect.inflate(-18, -18)
        pygame.draw.rect(self.screen, (13, 61, 54), glow_rect, border_radius=28)
        inner_glow = glow_rect.inflate(-34, -34)
        pygame.draw.rect(self.screen, (15, 72, 61), inner_glow, border_radius=26)

    def _compute_arena_rect(self, canvas_rect: pygame.Rect) -> pygame.Rect:
        available = canvas_rect.inflate(-FIELD_MARGIN * 2, -FIELD_MARGIN * 2)
        arena_aspect = (FIELD_X * 2.0) / (TOTAL_HALF_Y * 2.0)
        available_aspect = available.width / max(1, available.height)

        if available_aspect > arena_aspect:
            arena_height = available.height
            arena_width = int(arena_height * arena_aspect)
        else:
            arena_width = available.width
            arena_height = int(arena_width / arena_aspect)

        arena_rect = pygame.Rect(0, 0, arena_width, arena_height)
        arena_rect.center = available.center
        return arena_rect

    def _draw_soccar_surface(self, arena_rect: pygame.Rect) -> None:
        outer_poly = self._world_poly_to_screen(self._soccar_outline_world(), arena_rect)
        pygame.draw.polygon(self.screen, (29, 92, 61), outer_poly)
        pygame.draw.polygon(self.screen, (33, 103, 68), outer_poly)
        pygame.draw.lines(self.screen, (224, 234, 226), True, outer_poly, width=3)

    def _draw_reference_lines(self, arena_rect: pygame.Rect) -> None:
        center_top = self._world_to_screen(0.0, FIELD_Y, arena_rect)
        center_bottom = self._world_to_screen(0.0, -FIELD_Y, arena_rect)
        center = self._world_to_screen(0.0, 0.0, arena_rect)
        center_radius = int(abs(self._world_to_screen(0.0, 0.0, arena_rect)[1] - self._world_to_screen(0.0, 915.0, arena_rect)[1]))
        pygame.draw.line(self.screen, (224, 234, 226), center_top, center_bottom, width=2)
        pygame.draw.circle(self.screen, (224, 234, 226), center, max(8, center_radius), width=2)

        dotted_color = (160, 168, 174)
        path_points = [
            (-3210.0, 4200.0),
            (-3800.0, 1200.0),
            (-3800.0, -1200.0),
            (-3210.0, -4200.0),
            (0.0, -4700.0),
            (3210.0, -4200.0),
            (3800.0, -1200.0),
            (3800.0, 1200.0),
            (3210.0, 4200.0),
            (0.0, 4700.0),
            (-3210.0, 4200.0),
        ]
        self._draw_dotted_polyline(self._world_poly_to_screen(path_points, arena_rect), dotted_color, radius=2, spacing=12)

    def _draw_goals(self, arena_rect: pygame.Rect) -> None:
        top_goal = self._goal_world_rect(team_num=1)
        bottom_goal = self._goal_world_rect(team_num=0)
        self._draw_goal_box(top_goal, TEAM_COLORS[0], arena_rect, opening_toward_center=True)
        self._draw_goal_box(bottom_goal, TEAM_COLORS[1], arena_rect, opening_toward_center=True)

    def _draw_goal_box(
        self,
        world_rect: tuple[float, float, float, float],
        color: tuple[int, int, int],
        arena_rect: pygame.Rect,
        opening_toward_center: bool,
    ) -> None:
        left, top_y, right, bottom_y = world_rect
        top_left = self._world_to_screen(left, top_y, arena_rect)
        bottom_right = self._world_to_screen(right, bottom_y, arena_rect)
        rect = pygame.Rect(top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
        rect.normalize()

        fill_color = tuple(max(0, int(channel * 0.18)) for channel in color)
        border_color = tuple(min(255, int(channel * 0.85 + 30)) for channel in color)
        net_color = (220, 228, 234)
        pygame.draw.rect(self.screen, fill_color, rect, border_radius=10)
        pygame.draw.rect(self.screen, border_color, rect, width=3, border_radius=10)

        for idx in range(1, 5):
            y = int(rect.top + idx * rect.height / 5)
            pygame.draw.line(self.screen, net_color, (rect.left + 5, y), (rect.right - 5, y), width=1)
        for idx in range(1, 6):
            x = int(rect.left + idx * rect.width / 6)
            pygame.draw.line(self.screen, net_color, (x, rect.top + 5), (x, rect.bottom - 5), width=1)

        mouth_y = rect.bottom if opening_toward_center and rect.centery < self.height / 2 else rect.top
        pygame.draw.line(self.screen, border_color, (rect.left, mouth_y), (rect.right, mouth_y), width=4)

    def _draw_boost_pads(self, boost_pads: Iterable[Dict[str, Any]], arena_rect: pygame.Rect) -> None:
        for pad in boost_pads:
            position = pad.get("position", [0.0, 0.0, 0.0])
            x, y = self._world_to_screen(position[0], position[1], arena_rect)
            radius = 10 if pad.get("is_large", False) else 6
            color = (247, 205, 77) if pad.get("active", False) else (92, 84, 52)
            pygame.draw.circle(self.screen, color, (x, y), radius)
            if pad.get("is_large", False):
                pygame.draw.circle(self.screen, (250, 230, 130), (x, y), radius, width=1)

    def _draw_ball(self, ball: Dict[str, Any], arena_rect: pygame.Rect) -> None:
        position = ball.get("position", [0.0, 0.0, 0.0])
        x, y = self._world_to_screen(position[0], position[1], arena_rect)
        self.ball_trail.append((x, y))
        self._draw_trail(list(self.ball_trail), (255, 184, 80))
        height = float(ball.get("height", position[2] if len(position) > 2 else 0.0))
        shadow_radius = max(5, 12 - int(min(height, 1200.0) / 150.0))
        pygame.draw.circle(self.screen, (50, 45, 40), (x, y), shadow_radius)
        ball_radius = 9 + int(min(height, 1800.0) / 400.0)
        pygame.draw.circle(self.screen, (255, 169, 51), (x, y), ball_radius)
        self._blit_text(self.small_font, f"z={height:.0f}", (x + 12, y - 18), (255, 236, 186))

    def _draw_cars(self, cars: Iterable[Dict[str, Any]], arena_rect: pygame.Rect) -> None:
        for car in cars:
            position = car.get("position", [0.0, 0.0, 0.0])
            x, y = self._world_to_screen(position[0], position[1], arena_rect)
            car_id = int(car.get("car_id", 0))
            self.car_trails[car_id].append((x, y))
            color = TEAM_COLORS.get(int(car.get("team_num", 0)), (220, 220, 220))
            self._draw_trail(list(self.car_trails[car_id]), color)

            if car.get("is_demoed", False):
                pygame.draw.circle(self.screen, (255, 80, 80), (x, y), 16, width=3)
                pygame.draw.line(self.screen, (255, 80, 80), (x - 12, y - 12), (x + 12, y + 12), width=3)
                pygame.draw.line(self.screen, (255, 80, 80), (x + 12, y - 12), (x - 12, y + 12), width=3)
                continue

            yaw = float(car.get("yaw", 0.0))
            height = float(car.get("height", position[2] if len(position) > 2 else 0.0))
            scale = 18 + int(min(height, 900.0) / 300.0)
            points = self._car_triangle(x, y, yaw, scale)
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.polygon(self.screen, (18, 22, 26), points, width=2)
            self._blit_text(self.small_font, f"z={height:.0f}", (x - 16, y - scale - 18), (240, 245, 249))

    def _draw_boost_gauge(self, center: tuple[int, int], boost_amount: float) -> None:
        boost_amount = max(0.0, min(1.0, float(boost_amount)))
        radius = 24
        thickness = 4
        start_deg = 135.0
        sweep_deg = 270.0
        segments = 18
        gap_deg = 3.0
        active_segments = int(round(boost_amount * segments))

        core_color = (90, 22, 22)
        rim_color = (235, 240, 245)
        pygame.draw.circle(self.screen, core_color, center, radius - 7)
        pygame.draw.circle(self.screen, rim_color, center, radius - 7, width=1)

        for index in range(segments):
            seg_start = start_deg + index * (sweep_deg / segments)
            seg_end = seg_start + (sweep_deg / segments) - gap_deg
            active = index < active_segments
            color = (255, 187, 55) if active else (88, 94, 108)
            self._draw_arc_segment(center, radius, seg_start, seg_end, color, thickness)

        boost_text = f"{int(round(boost_amount * 100.0)):02d}"
        text_surface = self.font.render(boost_text, True, (255, 242, 214))
        text_rect = text_surface.get_rect(center=(center[0], center[1] - 2))
        self.screen.blit(text_surface, text_rect)
        boost_label = self.small_font.render("BOOST", True, (255, 170, 120))
        boost_rect = boost_label.get_rect(center=(center[0], center[1] + 13))
        self.screen.blit(boost_label, boost_rect)

    def _draw_arc_segment(
        self,
        center: tuple[int, int],
        radius: int,
        start_deg: float,
        end_deg: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        points = []
        for deg in np.linspace(start_deg, end_deg, 6):
            rad = np.deg2rad(deg)
            px = int(center[0] + np.cos(rad) * radius)
            py = int(center[1] - np.sin(rad) * radius)
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(self.screen, color, False, points, width=thickness)

    def _draw_side_panel(self, frame: Dict[str, Any]) -> None:
        panel_rect = pygame.Rect(self.field_width, 0, PANEL_WIDTH, self.height)
        pygame.draw.rect(self.screen, (22, 26, 34), panel_rect)
        pygame.draw.line(self.screen, (56, 64, 78), (self.field_width, 0), (self.field_width, self.height), width=2)

        self._blit_text(self.title_font, "2D Match View", (self.field_width + 20, 18), (245, 248, 252))
        ball = frame.get("ball", {})
        self._blit_text(
            self.font,
            f"Ball: z={ball.get('height', 0):.0f} speed={ball.get('speed', 0):.0f}",
            (self.field_width + 20, 56),
            (229, 233, 238),
        )
        self._blit_text(
            self.font,
            f"Last touch: {frame.get('last_touch', 0)}",
            (self.field_width + 20, 84),
            (180, 188, 202),
        )

        y = 132
        for car in frame.get("cars", []):
            y = self._draw_car_panel(car, y)

    def _draw_car_panel(self, car: Dict[str, Any], y: int) -> int:
        panel_x = self.field_width + 18
        panel_w = PANEL_WIDTH - 36
        color = TEAM_COLORS.get(int(car.get("team_num", 0)), (220, 220, 220))
        bg = pygame.Rect(panel_x, y, panel_w, 154)
        pygame.draw.rect(self.screen, (30, 35, 44), bg, border_radius=12)
        pygame.draw.rect(self.screen, color, bg, width=2, border_radius=12)

        title = f"Car {car.get('car_id', '?')} {'BLUE' if int(car.get('team_num', 0)) == 0 else 'ORANGE'}"
        self._blit_text(self.font, title, (panel_x + 12, y + 10), color)
        self._blit_text(
            self.small_font,
            f"speed={car.get('speed', 0.0):.0f} z={car.get('height', 0.0):.0f}",
            (panel_x + 12, y + 36),
            (228, 231, 236),
        )

        inputs = car.get("inputs", {})
        self._draw_axis_bar(panel_x + 12, y + 64, "throttle", float(inputs.get("throttle", 0.0)), (-1.0, 1.0), color)
        self._draw_axis_bar(panel_x + 12, y + 88, "steer", float(inputs.get("steer", 0.0)), (-1.0, 1.0), color)
        self._draw_axis_bar(panel_x + 12, y + 112, "pitch", float(inputs.get("pitch", 0.0)), (-1.0, 1.0), color)
        self._draw_axis_bar(panel_x + 12, y + 136, "yaw", float(inputs.get("yaw", 0.0)), (-1.0, 1.0), color)
        self._draw_binary_flags(panel_x + 196, y + 64, inputs)
        self._draw_boost_gauge((panel_x + 286, y + 92), float(car.get("boost_amount", 0.0)))
        return y + 168

    def _draw_axis_bar(self, x: int, y: int, label: str, value: float, limits: tuple[float, float], color) -> None:
        self._blit_text(self.small_font, label, (x, y), (226, 230, 235))
        bar_rect = pygame.Rect(x + 72, y + 2, 104, 14)
        pygame.draw.rect(self.screen, (63, 72, 86), bar_rect, border_radius=6)
        center = bar_rect.x + bar_rect.w // 2
        pygame.draw.line(self.screen, (129, 138, 151), (center, bar_rect.y), (center, bar_rect.bottom), width=1)
        min_v, max_v = limits
        value = max(min(value, max_v), min_v)
        if value >= 0:
            width = int((value / max_v) * (bar_rect.w / 2)) if max_v != 0 else 0
            rect = pygame.Rect(center, bar_rect.y, width, bar_rect.h)
        else:
            width = int((abs(value) / abs(min_v)) * (bar_rect.w / 2)) if min_v != 0 else 0
            rect = pygame.Rect(center - width, bar_rect.y, width, bar_rect.h)
        pygame.draw.rect(self.screen, color, rect, border_radius=6)

    def _draw_binary_flags(self, x: int, y: int, inputs: Dict[str, Any]) -> None:
        flags = [
            ("jump", bool(inputs.get("jump", False))),
            ("boost", bool(inputs.get("boost", False))),
            ("hb", bool(inputs.get("handbrake", False))),
        ]
        for index, (label, active) in enumerate(flags):
            pill = pygame.Rect(x, y + index * 24, 78, 18)
            pygame.draw.rect(self.screen, (63, 72, 86), pill, border_radius=9)
            if active:
                pygame.draw.rect(self.screen, (255, 214, 110), pill, border_radius=9)
                text_color = (18, 22, 26)
            else:
                text_color = (208, 214, 221)
            self._blit_text(self.small_font, label, (x + 22, y + 2 + index * 24), text_color)

    def _draw_height_indicator(self, x: int, y: int, height: float, on_ground: bool) -> None:
        bar = pygame.Rect(x, y, 18, 88)
        pygame.draw.rect(self.screen, (63, 72, 86), bar, border_radius=8)
        filled = int(max(0.0, min(height, 2044.0)) / 2044.0 * bar.h)
        fill_rect = pygame.Rect(bar.x, bar.bottom - filled, bar.w, filled)
        color = (109, 206, 164) if on_ground else (255, 173, 94)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=8)
        self._blit_text(self.small_font, "z", (x + 5, y - 18), (230, 234, 238))

    def _draw_trail(self, points: List[tuple[int, int]], color) -> None:
        if len(points) < 2:
            return
        for index in range(1, len(points)):
            alpha = index / len(points)
            trail_color = (
                int(color[0] * alpha),
                int(color[1] * alpha),
                int(color[2] * alpha),
            )
            pygame.draw.line(self.screen, trail_color, points[index - 1], points[index], width=2)

    def _draw_dotted_polyline(self, points: List[tuple[int, int]], color, radius: int = 2, spacing: int = 10) -> None:
        if len(points) < 2:
            return
        for start, end in zip(points[:-1], points[1:]):
            vec = pygame.Vector2(end[0] - start[0], end[1] - start[1])
            length = vec.length()
            if length <= 0:
                continue
            direction = vec.normalize()
            dot_count = max(1, int(length // spacing))
            for idx in range(dot_count + 1):
                pos = pygame.Vector2(start) + direction * min(length, idx * spacing)
                pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y)), radius)

    def _soccar_outline_world(self) -> list[tuple[float, float]]:
        return [
            (-GOAL_HALF_WIDTH, GOAL_BACK_Y),
            (GOAL_HALF_WIDTH, GOAL_BACK_Y),
            (GOAL_HALF_WIDTH, FIELD_Y),
            (FIELD_X - CORNER_OFFSET, FIELD_Y),
            (FIELD_X, SIDE_WALL_Y),
            (FIELD_X, -SIDE_WALL_Y),
            (FIELD_X - CORNER_OFFSET, -FIELD_Y),
            (GOAL_HALF_WIDTH, -FIELD_Y),
            (GOAL_HALF_WIDTH, -GOAL_BACK_Y),
            (-GOAL_HALF_WIDTH, -GOAL_BACK_Y),
            (-GOAL_HALF_WIDTH, -FIELD_Y),
            (-(FIELD_X - CORNER_OFFSET), -FIELD_Y),
            (-FIELD_X, -SIDE_WALL_Y),
            (-FIELD_X, SIDE_WALL_Y),
            (-(FIELD_X - CORNER_OFFSET), FIELD_Y),
            (-GOAL_HALF_WIDTH, FIELD_Y),
        ]

    def _goal_world_rect(self, team_num: int) -> tuple[float, float, float, float]:
        if team_num == 1:
            return (-GOAL_HALF_WIDTH, GOAL_BACK_Y, GOAL_HALF_WIDTH, FIELD_Y)
        return (-GOAL_HALF_WIDTH, -FIELD_Y, GOAL_HALF_WIDTH, -GOAL_BACK_Y)

    def _world_poly_to_screen(self, points: List[tuple[float, float]], arena_rect: pygame.Rect) -> list[tuple[int, int]]:
        return [self._world_to_screen(x, y, arena_rect) for x, y in points]

    def _world_to_screen(self, world_x: float, world_y: float, arena_rect: pygame.Rect) -> tuple[int, int]:
        normalized_x = (world_x + FIELD_X) / (FIELD_X * 2.0)
        normalized_y = 1.0 - ((world_y + TOTAL_HALF_Y) / (TOTAL_HALF_Y * 2.0))
        screen_x = int(arena_rect.left + normalized_x * arena_rect.width)
        screen_y = int(arena_rect.top + normalized_y * arena_rect.height)
        return screen_x, screen_y

    def _car_triangle(self, x: int, y: int, yaw: float, scale: int) -> list[tuple[int, int]]:
        forward = pygame.Vector2(np.cos(yaw), -np.sin(yaw))
        right = pygame.Vector2(forward.y, -forward.x)
        nose = pygame.Vector2(x, y) + forward * scale
        rear_left = pygame.Vector2(x, y) - forward * (scale * 0.7) + right * (scale * 0.6)
        rear_right = pygame.Vector2(x, y) - forward * (scale * 0.7) - right * (scale * 0.6)
        return [(int(nose.x), int(nose.y)), (int(rear_left.x), int(rear_left.y)), (int(rear_right.x), int(rear_right.y))]

    def _blit_text(self, font: pygame.font.Font, text: str, position: tuple[int, int], color) -> None:
        surface = font.render(text, True, color)
        self.screen.blit(surface, position)


def save_trajectory(frames: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frames, indent=2), encoding="utf-8")


def load_trajectory(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def play_trajectory(path: Path, fps: int = 15, loop: bool = False) -> None:
    frames = load_trajectory(path)
    viewer = TopDown2DViewer(title=f"Replay - {path.name}")
    running = True
    while running:
        for frame in frames:
            running = viewer.draw(frame, fps=fps)
            if not running:
                break
        if not loop:
            break
    viewer.close()



