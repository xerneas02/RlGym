from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pygame

from src.rl.topdown_viewer import (
    FIELD_X,
    FIELD_Y,
    GOAL_BACK_Y,
    TEAM_COLORS,
    TOTAL_HALF_Y,
    TopDown2DViewer,
)

BALL_Z_DEFAULT = 93.0
CAR_Z_DEFAULT = 17.0
EDITOR_PANEL_WIDTH = 520
EDITOR_WIDTH = 1640
EDITOR_HEIGHT = 920
GRID_COLOR = (70, 88, 96)
SELECTION_COLOR = (245, 245, 210)
RANDOM_BOX_COLORS = {
    "ball": (255, 184, 80),
    "car_blue": TEAM_COLORS[0],
    "car_orange": TEAM_COLORS[1],
}
ANGLE_CONE_COLOR = (250, 238, 170)
VELOCITY_COLOR = (165, 255, 214)
TEXT_COLOR = (229, 233, 238)
MUTED_TEXT = (164, 174, 184)
INPUT_BG = (34, 39, 48)
INPUT_BG_ACTIVE = (46, 54, 66)
BUTTON_BG = (46, 53, 65)
BUTTON_ACTIVE = (74, 103, 145)
FIELD_HEIGHT = 24
FIELD_LABEL_WIDTH = 116
FIELD_INPUT_WIDTH = 86
FIELD_GAP = 8
RIGHT_SECTION_X = 270


@dataclass
class Range3D:
    enabled: bool = False
    min_x: float = 0.0
    max_x: float = 0.0
    min_y: float = 0.0
    max_y: float = 0.0
    min_z: float = 0.0
    max_z: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "min_x": float(self.min_x),
            "max_x": float(self.max_x),
            "min_y": float(self.min_y),
            "max_y": float(self.max_y),
            "min_z": float(self.min_z),
            "max_z": float(self.max_z),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], defaults: Tuple[float, float, float]) -> "Range3D":
        x, y, z = defaults
        payload = data or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            min_x=float(payload.get("min_x", x)),
            max_x=float(payload.get("max_x", x)),
            min_y=float(payload.get("min_y", y)),
            max_y=float(payload.get("max_y", y)),
            min_z=float(payload.get("min_z", z)),
            max_z=float(payload.get("max_z", z)),
        )

    def normalize(self) -> None:
        self.min_x, self.max_x = sorted((float(self.min_x), float(self.max_x)))
        self.min_y, self.max_y = sorted((float(self.min_y), float(self.max_y)))
        self.min_z, self.max_z = sorted((float(self.min_z), float(self.max_z)))


@dataclass
class Range1D:
    enabled: bool = False
    value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "value": float(self.value),
            "min": float(self.min_value),
            "max": float(self.max_value),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], default: float) -> "Range1D":
        payload = data or {}
        value = float(payload.get("value", default))
        return cls(
            enabled=bool(payload.get("enabled", False)),
            value=value,
            min_value=float(payload.get("min", value)),
            max_value=float(payload.get("max", value)),
        )

    def normalize(self) -> None:
        self.min_value, self.max_value = sorted((float(self.min_value), float(self.max_value)))


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z)}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], default: Tuple[float, float, float]) -> "Vector3":
        dx, dy, dz = default
        payload = data or {}
        return cls(
            x=float(payload.get("x", dx)),
            y=float(payload.get("y", dy)),
            z=float(payload.get("z", dz)),
        )


@dataclass
class BallSpec:
    position: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, BALL_Z_DEFAULT))
    position_random: Range3D = field(default_factory=lambda: Range3D(False, 0.0, 0.0, 0.0, 0.0, BALL_Z_DEFAULT, BALL_Z_DEFAULT))
    velocity: Vector3 = field(default_factory=Vector3)
    angular_velocity: Vector3 = field(default_factory=Vector3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "position_random": self.position_random.to_dict(),
            "velocity": self.velocity.to_dict(),
            "angular_velocity": self.angular_velocity.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BallSpec":
        payload = data or {}
        position = Vector3.from_dict(payload.get("position"), (0.0, 0.0, BALL_Z_DEFAULT))
        return cls(
            position=position,
            position_random=Range3D.from_dict(payload.get("position_random"), (position.x, position.y, position.z)),
            velocity=Vector3.from_dict(payload.get("velocity"), (0.0, 0.0, 0.0)),
            angular_velocity=Vector3.from_dict(payload.get("angular_velocity"), (0.0, 0.0, 0.0)),
        )


@dataclass
class CarSpec:
    name: str
    team: int
    position: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, CAR_Z_DEFAULT))
    position_random: Range3D = field(default_factory=lambda: Range3D(False, 0.0, 0.0, 0.0, 0.0, CAR_Z_DEFAULT, CAR_Z_DEFAULT))
    velocity: Vector3 = field(default_factory=Vector3)
    yaw: Range1D = field(default_factory=lambda: Range1D(False, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0))
    boost: Range1D = field(default_factory=lambda: Range1D(False, 0.33, 0.33, 0.33))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "team": int(self.team),
            "position": self.position.to_dict(),
            "position_random": self.position_random.to_dict(),
            "velocity": self.velocity.to_dict(),
            "yaw": self.yaw.to_dict(),
            "boost": self.boost.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CarSpec":
        position = Vector3.from_dict(data.get("position"), (0.0, 0.0, CAR_Z_DEFAULT))
        default_yaw = math.pi / 2.0 if int(data.get("team", 0)) == 0 else -math.pi / 2.0
        return cls(
            name=str(data.get("name", "car")),
            team=int(data.get("team", 0)),
            position=position,
            position_random=Range3D.from_dict(data.get("position_random"), (position.x, position.y, position.z)),
            velocity=Vector3.from_dict(data.get("velocity"), (0.0, 0.0, 0.0)),
            yaw=Range1D.from_dict(data.get("yaw"), default_yaw),
            boost=Range1D.from_dict(data.get("boost"), 0.33),
        )


@dataclass
class ScenarioSpec:
    name: str
    ball: BallSpec
    cars: List[CarSpec]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ball": self.ball.to_dict(),
            "cars": [car.to_dict() for car in self.cars],
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ScenarioSpec":
        payload = data or {}
        cars = payload.get("cars") or []
        if not cars:
            cars = default_scenario().cars
        return cls(
            name=str(payload.get("name", "custom_setup")),
            ball=BallSpec.from_dict(payload.get("ball")),
            cars=[CarSpec.from_dict(item) if not isinstance(item, CarSpec) else item for item in cars],
        )


@dataclass
class FieldBinding:
    label: str
    getter: Callable[[], Any]
    setter: Callable[[str], None]
    rect: pygame.Rect
    visible: Callable[[], bool] = lambda: True


@dataclass
class ToggleBinding:
    label: str
    getter: Callable[[], bool]
    setter: Callable[[bool], None]
    rect: pygame.Rect
    visible: Callable[[], bool] = lambda: True


@dataclass
class EditorState:
    selected_kind: str = "ball"
    selected_index: int = 0
    active_field: Optional[int] = None
    edit_buffer: str = ""
    dragging_position: bool = False
    dragging_velocity: bool = False
    drawing_random_box: bool = False
    adjusting_yaw: bool = False
    adjusting_yaw_range: bool = False
    random_box_anchor: Optional[Tuple[float, float]] = None
    yaw_anchor: Optional[float] = None
    status: str = "Left drag: move | Ctrl+drag: velocity | Alt+drag: yaw | Alt+right drag: yaw range | Right drag: random zone"

class ScenarioEditor(TopDown2DViewer):
    def __init__(self, scenario_path: Path, scenario: ScenarioSpec) -> None:
        super().__init__(width=EDITOR_WIDTH, height=EDITOR_HEIGHT, title=f"Scenario Editor - {scenario.name}")
        self.field_width = self.width - EDITOR_PANEL_WIDTH
        self.scenario_path = scenario_path
        self.scenario = scenario
        self.state = EditorState()
        self.font = pygame.font.SysFont("consolas", 17)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.title_font = pygame.font.SysFont("consolas", 24, bold=True)
        self.help_font = pygame.font.SysFont("consolas", 13)
        self.fields: List[FieldBinding] = []
        self.toggles: List[ToggleBinding] = []
        self._rebuild_bindings()

    def run(self) -> None:
        running = True
        while running:
            running = self._handle_events()
            self._draw_editor()
            pygame.display.flip()
            self.clock.tick(60)
        self.close()

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)
            elif event.type == pygame.KEYDOWN:
                if not self._handle_key_down(event):
                    return False
        return True

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        mouse_pos = tuple(event.pos)
        if mouse_pos[0] >= self.field_width:
            self._handle_panel_click(mouse_pos)
            return

        arena_rect = self._current_arena_rect()
        selected = self._pick_object(mouse_pos, arena_rect)
        modifiers = pygame.key.get_mods()
        ctrl = bool(modifiers & pygame.KMOD_CTRL)
        alt = bool(modifiers & pygame.KMOD_ALT)

        if selected is not None:
            self._select(selected[0], selected[1])

        if event.button == 1:
            if ctrl:
                self.state.dragging_velocity = True
                self.state.status = "Dragging velocity vector"
                self._set_velocity_from_mouse(mouse_pos, arena_rect)
            elif alt and self.state.selected_kind == "car":
                self.state.adjusting_yaw = True
                self.state.status = "Dragging yaw"
                self._set_yaw_from_mouse(mouse_pos, arena_rect)
            else:
                self.state.dragging_position = True
                self.state.status = "Dragging object position"
                self._set_position_from_mouse(mouse_pos, arena_rect)
        elif event.button == 3:
            if alt and self.state.selected_kind == "car":
                self.state.adjusting_yaw_range = True
                self.state.yaw_anchor = self._current_yaw()
                self._set_yaw_range_from_mouse(mouse_pos, arena_rect)
                self.state.status = "Dragging yaw range"
            else:
                self.state.drawing_random_box = True
                self.state.random_box_anchor = self._screen_to_world(mouse_pos[0], mouse_pos[1], arena_rect)
                self._ensure_random_box_matches_position()
                self._set_random_box_from_drag(self.state.random_box_anchor, self.state.random_box_anchor)
                self.state.status = "Drawing random position box"
        elif event.button == 4:
            self._adjust_selected_height(40.0)
        elif event.button == 5:
            self._adjust_selected_height(-40.0)

    def _handle_mouse_up(self, event: pygame.event.Event) -> None:
        if event.button in (1, 3):
            self.state.dragging_position = False
            self.state.dragging_velocity = False
            self.state.drawing_random_box = False
            self.state.adjusting_yaw = False
            self.state.adjusting_yaw_range = False
            self.state.random_box_anchor = None
            self.state.yaw_anchor = None
            self.state.status = "Ready"

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        mouse_pos = tuple(event.pos)
        if mouse_pos[0] >= self.field_width:
            return
        arena_rect = self._current_arena_rect()
        if self.state.dragging_position:
            self._set_position_from_mouse(mouse_pos, arena_rect)
        elif self.state.dragging_velocity:
            self._set_velocity_from_mouse(mouse_pos, arena_rect)
        elif self.state.drawing_random_box and self.state.random_box_anchor is not None:
            current = self._screen_to_world(mouse_pos[0], mouse_pos[1], arena_rect)
            self._set_random_box_from_drag(self.state.random_box_anchor, current)
        elif self.state.adjusting_yaw:
            self._set_yaw_from_mouse(mouse_pos, arena_rect)
        elif self.state.adjusting_yaw_range:
            self._set_yaw_range_from_mouse(mouse_pos, arena_rect)

    def _handle_key_down(self, event: pygame.event.Event) -> bool:
        if self.state.active_field is not None:
            return self._handle_text_input(event)

        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_TAB:
            self._cycle_selection(backward=bool(event.mod & pygame.KMOD_SHIFT))
        elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
            self.save()
        elif event.key == pygame.K_l and event.mod & pygame.KMOD_CTRL:
            self.reload()
        elif event.key == pygame.K_e and event.mod & pygame.KMOD_CTRL:
            self.export_python()
        elif event.key == pygame.K_b:
            self._adjust_boost(0.05)
        elif event.key == pygame.K_n:
            self._adjust_boost(-0.05)
        elif event.key == pygame.K_q:
            self._adjust_yaw(-0.08)
        elif event.key == pygame.K_e:
            self._adjust_yaw(0.08)
        elif event.key == pygame.K_r:
            box = self._selected_position_random()
            box.enabled = not box.enabled
            self.state.status = f"Random position {'enabled' if box.enabled else 'disabled'}"
        elif event.key == pygame.K_t and self.state.selected_kind == "car":
            yaw = self._selected_car().yaw
            yaw.enabled = not yaw.enabled
            self.state.status = f"Random yaw {'enabled' if yaw.enabled else 'disabled'}"
        elif event.key == pygame.K_p and event.mod & pygame.KMOD_CTRL:
            self._add_car(team=0)
        elif event.key == pygame.K_o and event.mod & pygame.KMOD_CTRL:
            self._add_car(team=1)
        elif event.key == pygame.K_DELETE:
            self._delete_selected_car()
        elif event.key == pygame.K_UP:
            self._adjust_selected_height(40.0)
        elif event.key == pygame.K_DOWN:
            self._adjust_selected_height(-40.0)
        return True

    def _handle_text_input(self, event: pygame.event.Event) -> bool:
        if self.state.active_field is None:
            return True
        field = self.fields[self.state.active_field]
        if event.key == pygame.K_ESCAPE:
            self.state.active_field = None
            self.state.edit_buffer = ""
            self.state.status = "Edit cancelled"
            return True
        if event.key == pygame.K_RETURN:
            try:
                field.setter(self.state.edit_buffer)
                self.state.status = f"Updated {field.label}"
            except ValueError as exc:
                self.state.status = f"Invalid value for {field.label}: {exc}"
            self.state.active_field = None
            self.state.edit_buffer = ""
            return True
        if event.key == pygame.K_BACKSPACE:
            self.state.edit_buffer = self.state.edit_buffer[:-1]
            return True
        if event.unicode and event.unicode in "-+.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_":
            self.state.edit_buffer += event.unicode
        return True

    def _draw_editor(self) -> None:
        self.screen.fill((15, 18, 24))
        self._draw_editor_field()
        self._draw_editor_panel()

    def _draw_editor_field(self) -> None:
        canvas_rect = pygame.Rect(0, 0, self.field_width, self.height)
        self._draw_field_background(canvas_rect)
        arena_rect = self._compute_arena_rect(canvas_rect)
        self._draw_soccar_surface(arena_rect)
        self._draw_grid(arena_rect)
        self._draw_reference_lines(arena_rect)
        self._draw_goals(arena_rect)
        self._draw_scenario_entities(arena_rect)
        self._blit_text(self.title_font, self.scenario.name, (20, 18), (245, 248, 252))
        self._blit_text(self.font, self.scenario_path.name, (20, 50), MUTED_TEXT)
        self._blit_text(self.help_font, self.state.status, (20, self.height - 26), (214, 220, 226))

    def _draw_grid(self, arena_rect: pygame.Rect) -> None:
        for x in range(-4000, 4001, 1000):
            start = self._world_to_screen(x, -TOTAL_HALF_Y, arena_rect)
            end = self._world_to_screen(x, TOTAL_HALF_Y, arena_rect)
            pygame.draw.line(self.screen, GRID_COLOR, start, end, width=1)
        for y in range(-5000, 5001, 1000):
            start = self._world_to_screen(-FIELD_X, y, arena_rect)
            end = self._world_to_screen(FIELD_X, y, arena_rect)
            pygame.draw.line(self.screen, GRID_COLOR, start, end, width=1)

    def _draw_scenario_entities(self, arena_rect: pygame.Rect) -> None:
        self._draw_ball_spec(self.scenario.ball, arena_rect, selected=self.state.selected_kind == "ball")
        for index, car in enumerate(self.scenario.cars):
            self._draw_car_spec(car, arena_rect, selected=self.state.selected_kind == "car" and self.state.selected_index == index)

    def _draw_ball_spec(self, ball: BallSpec, arena_rect: pygame.Rect, selected: bool) -> None:
        color = RANDOM_BOX_COLORS["ball"]
        self._draw_random_box(ball.position_random, color, arena_rect)
        self._draw_velocity_arrow(ball.position, ball.velocity, arena_rect)
        x, y = self._world_to_screen(ball.position.x, ball.position.y, arena_rect)
        pygame.draw.circle(self.screen, color, (x, y), 11)
        pygame.draw.circle(self.screen, (20, 24, 28), (x, y), 11, width=2)
        if selected:
            pygame.draw.circle(self.screen, SELECTION_COLOR, (x, y), 17, width=2)
        self._draw_random_samples(ball.position_random, color, arena_rect)

    def _draw_car_spec(self, car: CarSpec, arena_rect: pygame.Rect, selected: bool) -> None:
        color = TEAM_COLORS.get(int(car.team), (220, 220, 220))
        random_color = RANDOM_BOX_COLORS["car_blue" if int(car.team) == 0 else "car_orange"]
        self._draw_random_box(car.position_random, random_color, arena_rect)
        self._draw_yaw_cone(car, arena_rect, selected)
        self._draw_velocity_arrow(car.position, car.velocity, arena_rect)
        x, y = self._world_to_screen(car.position.x, car.position.y, arena_rect)
        points = self._car_triangle(x, y, car.yaw.value, 18)
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (18, 22, 26), points, width=2)
        if selected:
            pygame.draw.circle(self.screen, SELECTION_COLOR, (x, y), 22, width=2)
        self._blit_text(self.small_font, car.name, (x + 16, y - 16), TEXT_COLOR)
        self._blit_text(self.small_font, f"boost={int(round(car.boost.value * 100))}", (x + 16, y + 2), MUTED_TEXT)
        self._draw_random_samples(car.position_random, random_color, arena_rect)

    def _draw_random_samples(self, position_random: Range3D, color, arena_rect: pygame.Rect) -> None:
        if not position_random.enabled:
            return
        samples = [
            (position_random.min_x, position_random.min_y),
            (position_random.max_x, position_random.min_y),
            (position_random.min_x, position_random.max_y),
            (position_random.max_x, position_random.max_y),
            ((position_random.min_x + position_random.max_x) * 0.5, (position_random.min_y + position_random.max_y) * 0.5),
        ]
        for sx, sy in samples:
            px, py = self._world_to_screen(sx, sy, arena_rect)
            pygame.draw.circle(self.screen, color, (px, py), 3)

    def _draw_random_box(self, random_box: Range3D, color, arena_rect: pygame.Rect) -> None:
        if not random_box.enabled:
            return
        random_box.normalize()
        top_left = self._world_to_screen(random_box.min_x, random_box.max_y, arena_rect)
        bottom_right = self._world_to_screen(random_box.max_x, random_box.min_y, arena_rect)
        rect = pygame.Rect(top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
        rect.normalize()
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill((color[0], color[1], color[2], 36))
        self.screen.blit(overlay, rect.topleft)
        pygame.draw.rect(self.screen, color, rect, width=2, border_radius=6)

    def _draw_yaw_cone(self, car: CarSpec, arena_rect: pygame.Rect, selected: bool) -> None:
        origin = self._world_to_screen(car.position.x, car.position.y, arena_rect)
        radius = 48
        if car.yaw.enabled:
            min_end = self._yaw_endpoint(origin, car.yaw.min_value, radius)
            max_end = self._yaw_endpoint(origin, car.yaw.max_value, radius)
            pygame.draw.line(self.screen, ANGLE_CONE_COLOR, origin, min_end, width=2)
            pygame.draw.line(self.screen, ANGLE_CONE_COLOR, origin, max_end, width=2)
            self._draw_arc(origin, radius, car.yaw.min_value, car.yaw.max_value, ANGLE_CONE_COLOR)
        if selected:
            current_end = self._yaw_endpoint(origin, car.yaw.value, radius + 12)
            pygame.draw.line(self.screen, SELECTION_COLOR, origin, current_end, width=3)

    def _draw_velocity_arrow(self, position: Vector3, velocity: Vector3, arena_rect: pygame.Rect) -> None:
        start = self._world_to_screen(position.x, position.y, arena_rect)
        scale = 0.08
        end = self._world_to_screen(position.x + velocity.x * scale, position.y + velocity.y * scale, arena_rect)
        pygame.draw.line(self.screen, VELOCITY_COLOR, start, end, width=2)
        direction = pygame.Vector2(end[0] - start[0], end[1] - start[1])
        if direction.length() > 0:
            direction = direction.normalize()
            left = pygame.Vector2(-direction.y, direction.x)
            tip = pygame.Vector2(end)
            arrow = [tip, tip - direction * 10 + left * 5, tip - direction * 10 - left * 5]
            pygame.draw.polygon(self.screen, VELOCITY_COLOR, [(int(p.x), int(p.y)) for p in arrow])

    def _draw_arc(self, center: Tuple[int, int], radius: int, start_yaw: float, end_yaw: float, color) -> None:
        sweep = end_yaw - start_yaw
        if sweep < 0:
            start_yaw, end_yaw = end_yaw, start_yaw
            sweep = -sweep
        points = []
        for idx in range(13):
            angle = start_yaw + sweep * idx / 12.0
            points.append(self._yaw_endpoint(center, angle, radius))
        if len(points) >= 2:
            pygame.draw.lines(self.screen, color, False, points, width=2)

    def _draw_editor_panel(self) -> None:
        panel_rect = pygame.Rect(self.field_width, 0, EDITOR_PANEL_WIDTH, self.height)
        pygame.draw.rect(self.screen, (22, 26, 34), panel_rect)
        pygame.draw.line(self.screen, (56, 64, 78), (self.field_width, 0), (self.field_width, self.height), width=2)
        self._blit_text(self.title_font, "Scenario Setup", (self.field_width + 18, 16), (245, 248, 252))
        self._blit_text(self.small_font, "Ctrl+S save | Ctrl+L reload | Ctrl+E export", (self.field_width + 18, 48), MUTED_TEXT)
        self._blit_text(self.small_font, "Ctrl+P blue | Ctrl+O orange | Del remove", (self.field_width + 18, 66), MUTED_TEXT)
        self._blit_text(self.font, f"Selected: {self._selected_label()}", (self.field_width + 18, 98), TEXT_COLOR)

        for toggle in self.toggles:
            if not toggle.visible():
                continue
            active = bool(toggle.getter())
            color = BUTTON_ACTIVE if active else BUTTON_BG
            pygame.draw.rect(self.screen, color, toggle.rect, border_radius=6)
            self._blit_text(self.small_font, f"{toggle.label}: {'ON' if active else 'OFF'}", (toggle.rect.x + 8, toggle.rect.y + 5), TEXT_COLOR)

        for index, binding in enumerate(self.fields):
            if not binding.visible():
                continue
            pygame.draw.rect(self.screen, INPUT_BG_ACTIVE if self.state.active_field == index else INPUT_BG, binding.rect, border_radius=5)
            pygame.draw.rect(self.screen, (72, 80, 94), binding.rect, width=1, border_radius=5)
            value = self.state.edit_buffer if self.state.active_field == index else self._format_value(binding)
            self._blit_text(self.small_font, binding.label, (binding.rect.x - FIELD_LABEL_WIDTH, binding.rect.y + 5), MUTED_TEXT)
            self._blit_text(self.small_font, value, (binding.rect.x + 6, binding.rect.y + 5), TEXT_COLOR)

        self._draw_object_list()

    def _draw_object_list(self) -> None:
        y = self.height - 186
        self._blit_text(self.font, "Objects", (self.field_width + 18, y), TEXT_COLOR)
        y += 28
        objects = [("ball", "Ball", None)] + [("car", car.name, index) for index, car in enumerate(self.scenario.cars)]
        for kind, label, index in objects:
            selected = self.state.selected_kind == kind and (kind == "ball" or self.state.selected_index == index)
            rect = pygame.Rect(self.field_width + 18, y, EDITOR_PANEL_WIDTH - 36, 28)
            pygame.draw.rect(self.screen, BUTTON_ACTIVE if selected else BUTTON_BG, rect, border_radius=6)
            self._blit_text(self.small_font, label, (rect.x + 10, rect.y + 7), TEXT_COLOR)
            y += 34

    def _handle_panel_click(self, mouse_pos: Tuple[int, int]) -> None:
        self.state.active_field = None
        self.state.edit_buffer = ""
        for toggle in self.toggles:
            if toggle.visible() and toggle.rect.collidepoint(mouse_pos):
                toggle.setter(not toggle.getter())
                self.state.status = f"Toggled {toggle.label}"
                return
        for index, binding in enumerate(self.fields):
            if binding.visible() and binding.rect.collidepoint(mouse_pos):
                self.state.active_field = index
                self.state.edit_buffer = self._format_value(binding)
                self.state.status = f"Editing {binding.label}"
                return
        y = self.height - 158
        if pygame.Rect(self.field_width + 18, y, EDITOR_PANEL_WIDTH - 36, 28).collidepoint(mouse_pos):
            self._select("ball", 0)
            return
        y += 34
        for index, _car in enumerate(self.scenario.cars):
            if pygame.Rect(self.field_width + 18, y, EDITOR_PANEL_WIDTH - 36, 28).collidepoint(mouse_pos):
                self._select("car", index)
                return
            y += 34

    def _rebuild_bindings(self) -> None:
        self.fields = []
        self.toggles = []
        base_x = self.field_width + FIELD_LABEL_WIDTH + 26
        right_x = self.field_width + RIGHT_SECTION_X + FIELD_LABEL_WIDTH
        y = 138

        def add_field(label: str, getter: Callable[[], Any], setter: Callable[[str], None], *, right: bool = False, visible: Optional[Callable[[], bool]] = None) -> None:
            nonlocal y
            rect = pygame.Rect(right_x if right else base_x, y, FIELD_INPUT_WIDTH, FIELD_HEIGHT)
            self.fields.append(FieldBinding(label=label, getter=getter, setter=setter, rect=rect, visible=visible or (lambda: True)))
            if right:
                y += FIELD_HEIGHT + FIELD_GAP

        def add_pair(left_label: str, left_getter, left_setter, right_label: str, right_getter, right_setter, *, visible: Optional[Callable[[], bool]] = None) -> None:
            add_field(left_label, left_getter, left_setter, right=False, visible=visible)
            add_field(right_label, right_getter, right_setter, right=True, visible=visible)

        def add_toggle(label: str, getter: Callable[[], bool], setter: Callable[[bool], None], *, right: bool = False, visible: Optional[Callable[[], bool]] = None) -> None:
            nonlocal y
            rect = pygame.Rect(self.field_width + 26 if not right else self.field_width + 270, y, 210, FIELD_HEIGHT)
            self.toggles.append(ToggleBinding(label=label, getter=getter, setter=setter, rect=rect, visible=visible or (lambda: True)))
            if right:
                y += FIELD_HEIGHT + FIELD_GAP

        visible_car = lambda: self.state.selected_kind == "car"
        visible_ball = lambda: self.state.selected_kind == "ball"

        add_toggle("Random Pos", lambda: self._selected_position_random().enabled, lambda value: self._set_bool(self._selected_position_random(), "enabled", value))
        add_toggle("Random Yaw", lambda: self._selected_car().yaw.enabled if self.state.selected_kind == "car" else False, lambda value: self._set_bool(self._selected_car().yaw, "enabled", value), right=True, visible=visible_car)
        add_toggle("Ball", lambda: self.state.selected_kind == "ball", lambda _value: None, visible=visible_ball)
        add_toggle("Car", lambda: self.state.selected_kind == "car", lambda _value: None, right=True, visible=visible_car)

        add_pair("pos_x", lambda: self._selected_position().x, lambda value: self._set_float(self._selected_position(), "x", value), "pos_y", lambda: self._selected_position().y, lambda value: self._set_float(self._selected_position(), "y", value))
        add_pair("pos_z", lambda: self._selected_position().z, lambda value: self._set_float(self._selected_position(), "z", value), "boost", lambda: self._selected_car().boost.value if self.state.selected_kind == "car" else 0.0, lambda value: self._set_float(self._selected_car().boost, "value", value, clamp=(0.0, 1.0)), visible=visible_car)
        add_pair("vel_x", lambda: self._selected_velocity().x, lambda value: self._set_float(self._selected_velocity(), "x", value), "vel_y", lambda: self._selected_velocity().y, lambda value: self._set_float(self._selected_velocity(), "y", value))
        add_pair("vel_z", lambda: self._selected_velocity().z, lambda value: self._set_float(self._selected_velocity(), "z", value), "yaw", lambda: self._selected_car().yaw.value if self.state.selected_kind == "car" else 0.0, lambda value: self._set_float(self._selected_car().yaw, "value", value), visible=visible_car)
        add_pair("rand_x_min", lambda: self._selected_position_random().min_x, lambda value: self._set_float(self._selected_position_random(), "min_x", value), "rand_x_max", lambda: self._selected_position_random().max_x, lambda value: self._set_float(self._selected_position_random(), "max_x", value))
        add_pair("rand_y_min", lambda: self._selected_position_random().min_y, lambda value: self._set_float(self._selected_position_random(), "min_y", value), "rand_y_max", lambda: self._selected_position_random().max_y, lambda value: self._set_float(self._selected_position_random(), "max_y", value))
        add_pair("rand_z_min", lambda: self._selected_position_random().min_z, lambda value: self._set_float(self._selected_position_random(), "min_z", value), "rand_z_max", lambda: self._selected_position_random().max_z, lambda value: self._set_float(self._selected_position_random(), "max_z", value))
        add_pair("yaw_min", lambda: self._selected_car().yaw.min_value if self.state.selected_kind == "car" else 0.0, lambda value: self._set_float(self._selected_car().yaw, "min_value", value), "yaw_max", lambda: self._selected_car().yaw.max_value if self.state.selected_kind == "car" else 0.0, lambda value: self._set_float(self._selected_car().yaw, "max_value", value), visible=visible_car)
        add_pair("boost_min", lambda: self._selected_car().boost.min_value if self.state.selected_kind == "car" else 0.0, lambda value: self._set_float(self._selected_car().boost, "min_value", value, clamp=(0.0, 1.0)), "boost_max", lambda: self._selected_car().boost.max_value if self.state.selected_kind == "car" else 0.0, lambda value: self._set_float(self._selected_car().boost, "max_value", value, clamp=(0.0, 1.0)), visible=visible_car)
        add_pair("ball_ang_x", lambda: self.scenario.ball.angular_velocity.x, lambda value: self._set_float(self.scenario.ball.angular_velocity, "x", value), "ball_ang_y", lambda: self.scenario.ball.angular_velocity.y, lambda value: self._set_float(self.scenario.ball.angular_velocity, "y", value), visible=visible_ball)
        add_pair("ball_ang_z", lambda: self.scenario.ball.angular_velocity.z, lambda value: self._set_float(self.scenario.ball.angular_velocity, "z", value), "name", lambda: self._selected_car().name if self.state.selected_kind == "car" else self.scenario.name, lambda value: self._set_name(value))

    def _format_value(self, binding: FieldBinding) -> str:
        value = binding.getter()
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def _set_name(self, raw: str) -> None:
        if self.state.selected_kind == "car":
            self._selected_car().name = raw.strip() or self._selected_car().name
        else:
            self.scenario.name = raw.strip() or self.scenario.name

    def _set_bool(self, target: Any, attr: str, value: bool) -> None:
        setattr(target, attr, bool(value))

    def _set_float(self, target: Any, attr: str, raw: str, clamp: Optional[Tuple[float, float]] = None) -> None:
        value = float(raw)
        if clamp is not None:
            value = max(clamp[0], min(clamp[1], value))
        setattr(target, attr, value)
        if isinstance(target, Range1D):
            target.normalize()
        elif isinstance(target, Range3D):
            target.normalize()

    def _pick_object(self, mouse_pos: Tuple[int, int], arena_rect: pygame.Rect) -> Optional[Tuple[str, int]]:
        candidates: List[Tuple[float, Tuple[str, int]]] = []
        bx, by = self._world_to_screen(self.scenario.ball.position.x, self.scenario.ball.position.y, arena_rect)
        candidates.append((pygame.Vector2(mouse_pos).distance_to((bx, by)), ("ball", 0)))
        for index, car in enumerate(self.scenario.cars):
            cx, cy = self._world_to_screen(car.position.x, car.position.y, arena_rect)
            candidates.append((pygame.Vector2(mouse_pos).distance_to((cx, cy)), ("car", index)))
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1] if candidates and candidates[0][0] <= 40 else None

    def _select(self, kind: str, index: int) -> None:
        self.state.selected_kind = kind
        self.state.selected_index = index
        self.state.active_field = None
        self.state.edit_buffer = ""

    def _cycle_selection(self, backward: bool = False) -> None:
        objects = [("ball", 0)] + [("car", idx) for idx in range(len(self.scenario.cars))]
        current = (self.state.selected_kind, 0 if self.state.selected_kind == "ball" else self.state.selected_index)
        if current not in objects:
            self._select("ball", 0)
            return
        next_index = (objects.index(current) - 1 if backward else objects.index(current) + 1) % len(objects)
        self._select(*objects[next_index])

    def _current_arena_rect(self) -> pygame.Rect:
        return self._compute_arena_rect(pygame.Rect(0, 0, self.field_width, self.height))

    def _screen_to_world(self, screen_x: int, screen_y: int, arena_rect: pygame.Rect) -> Tuple[float, float]:
        normalized_x = (screen_x - arena_rect.left) / max(1, arena_rect.width)
        normalized_y = (screen_y - arena_rect.top) / max(1, arena_rect.height)
        world_x = normalized_x * (FIELD_X * 2.0) - FIELD_X
        world_y = ((1.0 - normalized_y) * (TOTAL_HALF_Y * 2.0)) - TOTAL_HALF_Y
        return max(-FIELD_X, min(FIELD_X, world_x)), max(-GOAL_BACK_Y, min(GOAL_BACK_Y, world_y))

    def _selected_label(self) -> str:
        return "ball" if self.state.selected_kind == "ball" else self._selected_car().name

    def _selected_position(self) -> Vector3:
        return self.scenario.ball.position if self.state.selected_kind == "ball" else self._selected_car().position

    def _selected_velocity(self) -> Vector3:
        return self.scenario.ball.velocity if self.state.selected_kind == "ball" else self._selected_car().velocity

    def _selected_position_random(self) -> Range3D:
        return self.scenario.ball.position_random if self.state.selected_kind == "ball" else self._selected_car().position_random

    def _selected_car(self) -> CarSpec:
        return self.scenario.cars[self.state.selected_index]

    def _current_yaw(self) -> float:
        return self._selected_car().yaw.value if self.state.selected_kind == "car" else 0.0

    def _set_position_from_mouse(self, mouse_pos: Tuple[int, int], arena_rect: pygame.Rect) -> None:
        world_x, world_y = self._screen_to_world(mouse_pos[0], mouse_pos[1], arena_rect)
        position = self._selected_position()
        position.x = world_x
        position.y = world_y
        if self._selected_position_random().enabled:
            self._center_random_box_on_position()

    def _set_velocity_from_mouse(self, mouse_pos: Tuple[int, int], arena_rect: pygame.Rect) -> None:
        pos = self._selected_position()
        world_x, world_y = self._screen_to_world(mouse_pos[0], mouse_pos[1], arena_rect)
        velocity = self._selected_velocity()
        velocity.x = (world_x - pos.x) / 0.08
        velocity.y = (world_y - pos.y) / 0.08

    def _set_yaw_from_mouse(self, mouse_pos: Tuple[int, int], arena_rect: pygame.Rect) -> None:
        if self.state.selected_kind != "car":
            return
        pos = self._selected_position()
        world_x, world_y = self._screen_to_world(mouse_pos[0], mouse_pos[1], arena_rect)
        self._selected_car().yaw.value = math.atan2(-(world_y - pos.y), world_x - pos.x)

    def _set_yaw_range_from_mouse(self, mouse_pos: Tuple[int, int], arena_rect: pygame.Rect) -> None:
        if self.state.selected_kind != "car":
            return
        car = self._selected_car()
        car.yaw.enabled = True
        pos = self._selected_position()
        world_x, world_y = self._screen_to_world(mouse_pos[0], mouse_pos[1], arena_rect)
        yaw = math.atan2(-(world_y - pos.y), world_x - pos.x)
        anchor = car.yaw.value if self.state.yaw_anchor is None else self.state.yaw_anchor
        car.yaw.min_value = min(anchor, yaw)
        car.yaw.max_value = max(anchor, yaw)
        car.yaw.normalize()

    def _set_random_box_from_drag(self, anchor: Tuple[float, float], current: Tuple[float, float]) -> None:
        random_box = self._selected_position_random()
        random_box.enabled = True
        random_box.min_x = min(anchor[0], current[0])
        random_box.max_x = max(anchor[0], current[0])
        random_box.min_y = min(anchor[1], current[1])
        random_box.max_y = max(anchor[1], current[1])
        pos = self._selected_position()
        random_box.min_z = pos.z
        random_box.max_z = pos.z
        random_box.normalize()

    def _ensure_random_box_matches_position(self) -> None:
        random_box = self._selected_position_random()
        pos = self._selected_position()
        random_box.enabled = True
        random_box.min_x = pos.x
        random_box.max_x = pos.x
        random_box.min_y = pos.y
        random_box.max_y = pos.y
        random_box.min_z = pos.z
        random_box.max_z = pos.z

    def _center_random_box_on_position(self) -> None:
        random_box = self._selected_position_random()
        pos = self._selected_position()
        width = random_box.max_x - random_box.min_x
        height = random_box.max_y - random_box.min_y
        random_box.min_x = pos.x - width * 0.5
        random_box.max_x = pos.x + width * 0.5
        random_box.min_y = pos.y - height * 0.5
        random_box.max_y = pos.y + height * 0.5
        random_box.min_z = pos.z
        random_box.max_z = pos.z
        random_box.normalize()

    def _adjust_selected_height(self, delta: float) -> None:
        pos = self._selected_position()
        pos.z = max(0.0, pos.z + delta)
        random_box = self._selected_position_random()
        if random_box.enabled:
            random_box.min_z = pos.z
            random_box.max_z = pos.z

    def _adjust_boost(self, delta: float) -> None:
        if self.state.selected_kind != "car":
            return
        car = self._selected_car()
        car.boost.value = max(0.0, min(1.0, car.boost.value + delta))
        car.boost.min_value = max(0.0, min(1.0, car.boost.min_value + delta))
        car.boost.max_value = max(0.0, min(1.0, car.boost.max_value + delta))

    def _adjust_yaw(self, delta: float) -> None:
        if self.state.selected_kind != "car":
            return
        car = self._selected_car()
        car.yaw.value += delta
        car.yaw.min_value += delta
        car.yaw.max_value += delta

    def _yaw_endpoint(self, origin: Tuple[int, int], yaw: float, radius: int) -> Tuple[int, int]:
        return int(origin[0] + math.cos(yaw) * radius), int(origin[1] - math.sin(yaw) * radius)

    def _add_car(self, team: int) -> None:
        name_prefix = "blue" if team == 0 else "orange"
        count = sum(1 for car in self.scenario.cars if car.team == team)
        yaw = math.pi / 2.0 if team == 0 else -math.pi / 2.0
        y = -2000.0 if team == 0 else 2000.0
        new_car = CarSpec(
            name=f"{name_prefix}_{count}",
            team=team,
            position=Vector3(0.0, y, CAR_Z_DEFAULT),
            position_random=Range3D(False, 0.0, 0.0, y, y, CAR_Z_DEFAULT, CAR_Z_DEFAULT),
            velocity=Vector3(0.0, 0.0, 0.0),
            yaw=Range1D(False, yaw, yaw, yaw),
            boost=Range1D(False, 0.33, 0.33, 0.33),
        )
        self.scenario.cars.append(new_car)
        self._select("car", len(self.scenario.cars) - 1)
        self.state.status = f"Added {new_car.name}"

    def _delete_selected_car(self) -> None:
        if self.state.selected_kind != "car" or not self.scenario.cars:
            return
        removed = self.scenario.cars.pop(self.state.selected_index)
        if not self.scenario.cars:
            self._select("ball", 0)
        else:
            self._select("car", max(0, min(self.state.selected_index, len(self.scenario.cars) - 1)))
        self.state.status = f"Deleted {removed.name}"

    def save(self) -> None:
        self.scenario_path.parent.mkdir(parents=True, exist_ok=True)
        self.scenario_path.write_text(json.dumps(self.scenario.to_dict(), indent=2), encoding="utf-8")
        self.state.status = f"Saved to {self.scenario_path}"

    def reload(self) -> None:
        self.scenario = load_scenario(self.scenario_path)
        self._select("ball", 0)
        self.state.status = f"Reloaded {self.scenario_path}"

    def export_python(self) -> None:
        export_path = self.scenario_path.with_suffix(".py")
        export_path.write_text(render_python_scenario(self.scenario), encoding="utf-8")
        self.state.status = f"Exported Python snippet to {export_path}"


def default_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        name="custom_setup",
        ball=BallSpec(
            position=Vector3(0.0, 0.0, BALL_Z_DEFAULT),
            position_random=Range3D(False, 0.0, 0.0, 0.0, 0.0, BALL_Z_DEFAULT, BALL_Z_DEFAULT),
            velocity=Vector3(0.0, 0.0, 0.0),
            angular_velocity=Vector3(0.0, 0.0, 0.0),
        ),
        cars=[
            CarSpec(
                name="blue_0",
                team=0,
                position=Vector3(0.0, -2048.0, CAR_Z_DEFAULT),
                position_random=Range3D(False, 0.0, 0.0, -2048.0, -2048.0, CAR_Z_DEFAULT, CAR_Z_DEFAULT),
                velocity=Vector3(0.0, 0.0, 0.0),
                yaw=Range1D(False, math.pi / 2.0, math.pi / 2.0, math.pi / 2.0),
                boost=Range1D(False, 0.33, 0.33, 0.33),
            ),
            CarSpec(
                name="orange_0",
                team=1,
                position=Vector3(0.0, 2048.0, CAR_Z_DEFAULT),
                position_random=Range3D(False, 0.0, 0.0, 2048.0, 2048.0, CAR_Z_DEFAULT, CAR_Z_DEFAULT),
                velocity=Vector3(0.0, 0.0, 0.0),
                yaw=Range1D(False, -math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0),
                boost=Range1D(False, 0.33, 0.33, 0.33),
            ),
        ],
    )


def load_scenario(path: Path) -> ScenarioSpec:
    if not path.exists():
        scenario = default_scenario()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scenario.to_dict(), indent=2), encoding="utf-8")
        return scenario
    return ScenarioSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def render_python_scenario(scenario: ScenarioSpec) -> str:
    return (
        "from __future__ import annotations\n\n"
        "# Generated by scenario_editor.py\n"
        "SCENARIO_SPEC = "
        f"{json.dumps(scenario.to_dict(), indent=2)}\n"
    )


def run_editor(scenario_path: Path) -> None:
    scenario = load_scenario(scenario_path)
    editor = ScenarioEditor(scenario_path=scenario_path, scenario=scenario)
    editor.run()


