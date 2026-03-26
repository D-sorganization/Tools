# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""3D Renderer
===============

OpenGL-based renderer for the solar system visualization.
Provides high-quality rendering of planets, orbits, trajectories,
and UI elements.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

try:
    import pygame
    from pygame.locals import (
        DOUBLEBUF,
        FULLSCREEN,
        GL_MULTISAMPLEBUFFERS,
        GL_MULTISAMPLESAMPLES,
        OPENGL,
    )

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from OpenGL.GL import (
        GL_AMBIENT,
        GL_AMBIENT_AND_DIFFUSE,
        GL_BLEND,
        GL_COLOR_ARRAY,
        GL_COLOR_BUFFER_BIT,
        GL_COLOR_MATERIAL,
        GL_COMPILE,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_DIFFUSE,
        GL_FLOAT,
        GL_FRAGMENT_SHADER,
        GL_FRONT_AND_BACK,
        GL_LEQUAL,
        GL_LIGHT0,
        GL_LIGHTING,
        GL_LINE_LOOP,
        GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT,
        GL_LINE_STRIP,
        GL_LINES,
        GL_MODELVIEW,
        GL_MODELVIEW_MATRIX,
        GL_NICEST,
        GL_NORMALIZE,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINT_SMOOTH,
        GL_POINT_SMOOTH_HINT,
        GL_POINTS,
        GL_POSITION,
        GL_PROJECTION,
        GL_PROJECTION_MATRIX,
        GL_QUAD_STRIP,
        GL_SPECULAR,
        GL_SRC_ALPHA,
        GL_TEXTURE_2D,
        GL_TRIANGLE_FAN,
        GL_VERTEX_ARRAY,
        GL_VERTEX_SHADER,
        GL_VIEWPORT,
        glAttachShader,
        glBegin,
        glBlendFunc,
        glCallList,
        glClear,
        glClearColor,
        glColor3f,
        glColor4f,
        glColorMaterial,
        glColorPointer,
        glCompileShader,
        glCreateProgram,
        glCreateShader,
        glDeleteLists,
        glDepthFunc,
        glDisable,
        glDisableClientState,
        glDrawArrays,
        glEnable,
        glEnableClientState,
        glEnd,
        glEndList,
        glGenLists,
        glGetDoublev,
        glGetIntegerv,
        glHint,
        glLightfv,
        glLineWidth,
        glLinkProgram,
        glLoadIdentity,
        glMatrixMode,
        glNewList,
        glNormal3f,
        glPointSize,
        glPopMatrix,
        glPushMatrix,
        glScalef,
        glShaderSource,
        glTexCoord2f,
        glTranslatef,
        glUseProgram,
        glVertex3f,
        glVertexPointer,
        glViewport,
    )
    from OpenGL.GLU import (
        gluLookAt,
        gluPerspective,
        gluProject,
    )

    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

from ..core.celestial_body import BodyType, CelestialBody, StateVector
from ..core.constants import AU
from ..data.star_catalog import iter_catalog
from .camera import Camera, CameraState
from .starfield import StarVertex, build_star_vertices, point_size_from_magnitude
from .textures import TextureManager
from .ui_renderer import UIRenderer


@dataclass
class RenderSettings:
    """Settings for the renderer."""

    window_width: int = 1600
    window_height: int = 900
    fullscreen: bool = False
    vsync: bool = True
    antialiasing: bool = True
    show_orbits: bool = True
    show_labels: bool = True
    show_grid: bool = False
    show_axes: bool = False
    use_textures: bool = True
    use_shaders: bool = True
    stereo_view: bool = False
    orbit_segments: int = 360
    planet_segments: int = 32
    background_color: tuple[float, float, float, float] = (0.02, 0.02, 0.05, 1.0)


class Renderer:
    """
    OpenGL renderer for the solar system.

    Handles all rendering including:
    - Planets and sun with proper colors and sizes
    - Orbital paths
    - Trajectory lines
    - Star field background
    - UI overlays (delegated to UIRenderer)
    """

    def __init__(self, settings: RenderSettings | None = None) -> None:
        """
        Initialize the renderer.

        Args:
            settings: Render settings configuration
        """
        if not PYGAME_AVAILABLE or not OPENGL_AVAILABLE:
            raise ImportError(
                "PyGame and PyOpenGL are required for visualization. "
                "Install with: pip install pygame PyOpenGL PyOpenGL_accelerate"
            )

        self.settings = settings or RenderSettings()
        self.camera = Camera()

        # Display state
        self.display: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.running = False

        # Rendering data
        self._sphere_list: int | None = None
        self._ring_list: int | None = None
        self._circle_list: int | None = None
        self.star_batches: list[tuple[float, int, np.ndarray, np.ndarray]] = []
        self.star_vertices: list[StarVertex] = []

        # Scale factor for visualization
        self.distance_scale = 1e-9  # Convert meters to viewable units
        self.size_scale = 5e-7  # Scale for body sizes
        self.min_body_size = 0.05
        self.max_body_size = 0.8
        self.sun_size = 1.0

        # UI state
        self.selected_body: CelestialBody | None = None
        self.hovered_body: CelestialBody | None = None
        self.ui_renderer: UIRenderer | None = None

        # Textures and shaders
        assets_root = pathlib.Path(__file__).resolve().parent.parent
        self.texture_manager = TextureManager(assets_root, auto_download=True)
        self._shaders_enabled = False
        self._shader_program: int | None = None

    def initialize(self) -> bool:
        """
        Initialize the rendering system.

        Returns:
            True if initialization successful
        """
        # Initialize pygame
        pygame.init()
        pygame.font.init()

        # Set up display
        flags = DOUBLEBUF | OPENGL
        if self.settings.fullscreen:
            flags |= FULLSCREEN

        if self.settings.antialiasing:
            pygame.display.gl_set_attribute(GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES, 4)

        self.display = pygame.display.set_mode(
            (self.settings.window_width, self.settings.window_height), flags
        )
        pygame.display.set_caption("Solar System Simulation")

        self.clock = pygame.time.Clock()

        # Initialize UI Renderer
        self.ui_renderer = UIRenderer(self.settings.window_width, self.settings.window_height)

        # OpenGL setup
        self._setup_opengl()

        # Create display lists for common objects
        self._create_display_lists()

        # Generate star field from catalog
        self._generate_stars()

        self.running = True
        return True

    def _setup_opengl(self) -> None:
        """Configure OpenGL state."""
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Enable smooth lines
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Enable point smoothing
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # Set up lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Light at origin (sun)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 0, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

        # Normalize normals
        glEnable(GL_NORMALIZE)

        # Background color
        glClearColor(*self.settings.background_color)

        # Set up viewport
        glViewport(0, 0, self.settings.window_width, self.settings.window_height)

        # Modern shading pipeline
        self._setup_shaders()

    def _create_display_lists(self) -> None:
        """Create OpenGL display lists for common objects."""
        # Sphere for planets
        self._sphere_list = glGenLists(1)
        glNewList(self._sphere_list, GL_COMPILE)
        self._draw_sphere(1.0, self.settings.planet_segments)
        glEndList()

        # Circle for orbits
        self._circle_list = glGenLists(1)
        glNewList(self._circle_list, GL_COMPILE)
        self._draw_circle(1.0, self.settings.orbit_segments)
        glEndList()

    def _setup_shaders(self) -> None:
        """Compile a minimal Lambert shader for per-pixel lighting."""

        if not self.settings.use_shaders:
            return

        try:
            vertex_shader = glCreateShader(GL_VERTEX_SHADER)
            fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)

            vertex_src = """
            varying vec3 vNormal;
            varying vec3 vPosition;
            void main() {
                vNormal = normalize(gl_NormalMatrix * gl_Normal);
                vPosition = vec3(gl_ModelViewMatrix * gl_Vertex);
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                gl_TexCoord[0] = gl_MultiTexCoord0;
            }
            """

            fragment_src = """
            varying vec3 vNormal;
            varying vec3 vPosition;
            void main() {
                vec3 lightDir = normalize(vec3(0.0, 0.0, 0.0) - vPosition);
                float diffuse = max(dot(vNormal, lightDir), 0.2);
                vec4 baseColor = gl_Color;
                if (gl_TexCoord[0].s > 0.0) {
                    baseColor *= texture2D(gl_Texture_2D, gl_TexCoord[0].st);
                }
                gl_FragColor = vec4(baseColor.rgb * diffuse, baseColor.a);
            }
            """

            glShaderSource(vertex_shader, vertex_src)
            glCompileShader(vertex_shader)
            glShaderSource(fragment_shader, fragment_src)
            glCompileShader(fragment_shader)

            program = glCreateProgram()
            glAttachShader(program, vertex_shader)
            glAttachShader(program, fragment_shader)
            glLinkProgram(program)
            glUseProgram(program)
            self._shader_program = program
            self._shaders_enabled = True
        except (KeyError, ValueError, TypeError):
            self._shader_program = None
            self._shaders_enabled = False

    def _draw_sphere(self, radius: float, segments: int) -> None:
        """Draw a unit sphere using immediate mode."""
        for i in range(segments):
            lat0 = math.pi * (-0.5 + float(i) / segments)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)

            lat1 = math.pi * (-0.5 + float(i + 1) / segments)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)

            glBegin(GL_QUAD_STRIP)
            for j in range(segments + 1):
                lng = 2 * math.pi * float(j) / segments
                x = math.cos(lng)
                y = math.sin(lng)

                u = float(j) / segments
                v0 = 0.5 + (lat0 / math.pi)
                v1 = 0.5 + (lat1 / math.pi)

                glTexCoord2f(u, v0)
                glNormal3f(x * zr0, y * zr0, z0)
                glVertex3f(radius * x * zr0, radius * y * zr0, radius * z0)

                glTexCoord2f(u, v1)
                glNormal3f(x * zr1, y * zr1, z1)
                glVertex3f(radius * x * zr1, radius * y * zr1, radius * z1)
            glEnd()

    def _draw_circle(self, radius: float, segments: int) -> None:
        """Draw a circle in the XY plane."""
        if not (radius is not None):
            raise ValueError("radius must be provided")
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            glVertex3f(radius * math.cos(angle), 0, radius * math.sin(angle))
        glEnd()

    def _generate_stars(self) -> None:
        """Build a star field from the curated catalog using Vertex Arrays."""
        self.star_vertices = build_star_vertices(iter_catalog())

        # Group stars by integer point size for batching
        stars_by_size: dict[int, list[StarVertex]] = {}
        for star in self.star_vertices:
            size = int(point_size_from_magnitude(star.magnitude))
            if size not in stars_by_size:
                stars_by_size[size] = []
            stars_by_size[size].append(star)

        self.star_batches = []
        for size, stars in stars_by_size.items():
            if not stars:
                continue

            # Optimize array creation - pre-allocate and fill directly
            n_stars = len(stars)
            coords = np.empty((n_stars, 3), dtype=np.float32)
            colors = np.empty((n_stars, 3), dtype=np.float32)

            for i, star in enumerate(stars):
                coords[i] = star.position
                colors[i] = star.color

            self.star_batches.append((float(size), n_stars, coords, colors))

    def begin_frame(self, camera_state: CameraState | None = None, clear: bool = True) -> None:
        """Begin a new frame."""
        if not (clear is not None):
            raise ValueError("clear must be provided")
        if self.ui_renderer:
            self.ui_renderer.drawn_labels.clear()
        if clear:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        active_camera = camera_state or self.camera
        aspect = self.settings.window_width / self.settings.window_height
        gluPerspective(active_camera.fov, aspect, active_camera.near, active_camera.far)

        # Set up view
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Apply camera
        gluLookAt(*active_camera.position, *active_camera.target, *active_camera.up)

    def end_frame(self) -> None:
        """End current frame and swap buffers."""
        pygame.display.flip()
        if self.clock:
            self.clock.tick(60)  # Cap at 60 FPS

    def render_stars(self) -> None:
        """Render the star field background."""
        if not self.star_batches:
            return

        glDisable(GL_LIGHTING)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        for size, count, coords, colors in self.star_batches:
            glPointSize(size)
            glVertexPointer(3, GL_FLOAT, 0, coords)
            glColorPointer(3, GL_FLOAT, 0, colors)
            glDrawArrays(GL_POINTS, 0, count)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glEnable(GL_LIGHTING)

    def render_body(self, body: CelestialBody, julian_date: float, highlight: bool = False) -> None:
        """
        Render a celestial body.

        Args:
            body: The body to render
            julian_date: Current simulation time
            highlight: Whether to highlight (selected/hovered)
        """
        if not (body is not None):
            raise ValueError("body must be provided")
        state = body.get_state_at_time(julian_date)
        position = state.position * self.distance_scale

        # Calculate visual size
        if body.body_type == BodyType.STAR:
            size = self.sun_size
        else:
            # Scale based on actual radius but with limits
            actual_size = body.radius * self.size_scale * 1e6
            size = np.clip(actual_size, self.min_body_size, self.max_body_size)

            # Make larger planets more visible
            if body.radius > 20000:  # Gas giants
                size = max(size, 0.15)

        glPushMatrix()
        glTranslatef(*position)

        # Set color
        color = body.color

        if body.body_type == BodyType.STAR:
            # Sun is emissive - disable lighting
            glDisable(GL_LIGHTING)
            glColor3f(*color)
        else:
            glEnable(GL_LIGHTING)
            glColor3f(*color)

        # Highlight effect
        if highlight:
            glColor3f(
                min(color[0] + 0.3, 1.0),
                min(color[1] + 0.3, 1.0),
                min(color[2] + 0.3, 1.0),
            )

        texturing_active = False
        if self.settings.use_textures:
            texturing_active = self.texture_manager.bind(body.name)
            if texturing_active:
                glEnable(GL_TEXTURE_2D)

        # Draw sphere
        glPushMatrix()
        glScalef(size, size, size)
        glCallList(self._sphere_list)
        glPopMatrix()

        if body.body_type == BodyType.STAR:
            self._render_star_glow(size, color)

        if highlight:
            self._render_selection_ring(size)

        # Draw rings for Saturn, etc.
        if hasattr(body, "has_rings") and body.has_rings:
            self._render_rings(body, size)

        glPopMatrix()

        # Re-enable lighting
        glEnable(GL_LIGHTING)
        if texturing_active:
            glDisable(GL_TEXTURE_2D)

    def _render_star_glow(self, body_size: float, color: tuple[float, float, float]) -> None:
        """Render a soft halo to make the Sun feel more luminous."""
        if not (body_size is not None):
            raise ValueError("body_size must be provided")
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glow_radius = body_size * 1.9
        glColor4f(color[0], color[1], min(color[2] + 0.1, 1.0), 0.18)

        segments = 48
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0.0, 0.0, 0.0)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(glow_radius * math.cos(angle), 0.0, glow_radius * math.sin(angle))
        glEnd()
        glDisable(GL_BLEND)

    def _render_selection_ring(self, body_size: float) -> None:
        """Render an orbit-plane selection ring for the chosen body."""
        if not (body_size is not None):
            raise ValueError("body_size must be provided")
        glDisable(GL_LIGHTING)
        glColor4f(0.95, 0.95, 0.6, 0.85)
        glLineWidth(2.0)

        ring_radius = body_size * 1.8
        segments = 64
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            glVertex3f(ring_radius * math.cos(angle), 0.0, ring_radius * math.sin(angle))
        glEnd()

    def _render_rings(self, body: CelestialBody, body_size: float) -> None:
        """Render planetary rings."""
        # Ring color (semi-transparent)
        if not (body is not None):
            raise ValueError("body must be provided")
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)

        ring_inner = body_size * 1.4
        ring_outer = body_size * 2.3

        # Draw multiple ring bands
        glColor4f(0.8, 0.75, 0.6, 0.5)

        segments = 64
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            glVertex3f(ring_inner * cos_a, 0, ring_inner * sin_a)
            glVertex3f(ring_outer * cos_a, 0, ring_outer * sin_a)
        glEnd()

        glEnable(GL_LIGHTING)

    def render_orbit(
        self,
        body: CelestialBody,
        julian_date: float,
        color: tuple[float, float, float, float] | None = None,
    ) -> None:
        """
        Render the orbital path of a body using Vertex Arrays.

        Args:
            body: The body whose orbit to render
            julian_date: Current time for element calculation
            color: Optional override color (RGBA)
        """
        if not (body is not None):
            raise ValueError("body must be provided")
        if body.orbital_elements is None:
            return

        # Get orbit points
        points = body.get_orbit_points(julian_date, self.settings.orbit_segments)
        points = points * self.distance_scale
        points = points.astype(np.float32)

        glDisable(GL_LIGHTING)
        glLineWidth(1.0)

        if color is None:
            # Use body color with reduced alpha
            body_color = body.color
            glColor4f(body_color[0] * 0.6, body_color[1] * 0.6, body_color[2] * 0.6, 0.4)
        else:
            glColor4f(*color)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, points)
        glDrawArrays(GL_LINE_LOOP, 0, len(points))
        glDisableClientState(GL_VERTEX_ARRAY)

        glEnable(GL_LIGHTING)

    def render_trajectory(
        self,
        points: list[StateVector],
        color: tuple[float, float, float, float] = (0.0, 1.0, 0.5, 0.8),
        line_width: float = 2.0,
    ) -> None:
        """
        Render a spacecraft trajectory using Vertex Arrays.

        Args:
            points: List of state vectors defining the trajectory
            color: Line color (RGBA)
            line_width: Width of the trajectory line
        """
        if not (points is not None):
            raise ValueError("points must be provided")
        if len(points) < 2:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(line_width)
        glColor4f(*color)

        # Extract positions using pre-allocated array for better performance
        n_points = len(points)
        pos_array = np.empty((n_points, 3), dtype=np.float32)

        # Fill array directly to avoid list comprehension overhead (2x speedup)
        for i, state in enumerate(points):
            pos_array[i] = state.position * self.distance_scale

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, pos_array)
        glDrawArrays(GL_LINE_STRIP, 0, len(pos_array))
        glDisableClientState(GL_VERTEX_ARRAY)

        glEnable(GL_LIGHTING)

    def render_asteroid_belt(self, belt_points_au: np.ndarray) -> None:
        """Render a faint asteroid belt based on pre-generated particle positions."""

        if not (belt_points_au is not None):
            raise ValueError("belt_points_au must be provided")
        if belt_points_au.size == 0:
            return

        glDisable(GL_LIGHTING)
        glPointSize(1.2)
        glColor4f(0.7, 0.7, 0.7, 0.35)

        # Use Vertex Array for belt
        points = (belt_points_au * AU * self.distance_scale).astype(np.float32)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, points)
        glDrawArrays(GL_POINTS, 0, len(points))
        glDisableClientState(GL_VERTEX_ARRAY)

        glEnable(GL_LIGHTING)

    def render_grid(self, size: float = 10.0, divisions: int = 20) -> None:
        """Render a reference grid in the ecliptic plane."""
        if not (size is not None):
            raise ValueError("size must be provided")
        if not self.settings.show_grid:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor4f(0.2, 0.2, 0.3, 0.3)

        step = size / divisions

        # Build grid lines
        vertices = []
        for i in range(-divisions, divisions + 1):
            # Parallel to X
            vertices.extend([-size, 0, i * step])
            vertices.extend([size, 0, i * step])
            # Parallel to Z
            vertices.extend([i * step, 0, -size])
            vertices.extend([i * step, 0, size])

        v_array = np.array(vertices, dtype=np.float32)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, v_array)
        glDrawArrays(GL_LINES, 0, len(vertices) // 3)
        glDisableClientState(GL_VERTEX_ARRAY)

        glEnable(GL_LIGHTING)

    def render_axes(self, size: float = 2.0) -> None:
        """Render coordinate axes for reference."""
        if not (size is not None):
            raise ValueError("size must be provided")
        if not self.settings.show_axes:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(2.0)

        glBegin(GL_LINES)
        # X axis - red
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(size, 0, 0)

        # Y axis - green
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, size, 0)

        # Z axis - blue
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, size)
        glEnd()

        glEnable(GL_LIGHTING)

    # Distance thresholds per priority level (Issue #811)
    # Priority 3 = planets (always visible), 2 = important, 1 = minor
    _LABEL_MAX_DISTANCE = {1: 15.0, 2: 80.0, 3: 500.0}

    def render_label(
        self,
        text: str,
        position_3d: np.ndarray,
        color: tuple[int, int, int] = (255, 255, 255),
        offset: tuple[int, int] = (10, -10),
        priority: int = 1,
    ) -> None:
        """Render a text label at a 3D position with priority-based visibility.

        Priority levels control distance-based visibility and font size:
            3 = planets/Sun (large font, visible from far away)
            2 = important bodies (medium font, moderate distance)
            1 = minor bodies (small font, only when nearby)
        """
        if not (text is not None):
            raise ValueError("text must be provided")
        if not self.settings.show_labels:
            return

        # Calculate distance for fading (Issue #811)
        cam_pos = np.array(self.camera.position)
        dist = float(np.linalg.norm(position_3d - cam_pos))

        # Distance clipping: lower priority labels disappear further away
        max_dist = self._LABEL_MAX_DISTANCE.get(priority, 20.0)
        if dist > max_dist:
            return

        # Project 3D to 2D screen coordinates
        screen_pos = self._project_to_screen(position_3d)

        if screen_pos is None:
            return

        # Fade alpha based on distance -- reference distance of 5.0 for full
        # brightness, with a priority multiplier so important labels stay bright
        ref_dist = 5.0 * priority
        alpha_scale = float(max(0.2, min(1.0, ref_dist / dist)))

        # Apply fading to color
        faded_color = tuple(int(c * alpha_scale) for c in color)

        x, y = screen_pos
        x += offset[0]
        y += offset[1]

        # Choose font name based on priority (Issue #811 - font size scaling)
        # priority 3 = planets -> "default" (large font)
        # priority 2 = dwarf planets, spacecraft -> "default" (medium)
        # priority 1 = moons, asteroids -> "small" (small font)
        font_name = "small" if priority <= 1 else "default"

        # Render text using UI renderer with collision avoidance
        if self.ui_renderer:
            self.ui_renderer.render_label_2d(
                text,
                (x, y),
                cast(tuple[int, int, int], faded_color),
                font_name=font_name,
            )

    def _project_to_screen(self, position_3d: np.ndarray) -> tuple[int, int] | None:
        """Project 3D position to 2D screen coordinates."""
        if not (position_3d is not None):
            raise ValueError("position_3d must be provided")
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        try:
            x, y, z = gluProject(
                position_3d[0],
                position_3d[1],
                position_3d[2],
                modelview,
                projection,
                viewport,
            )

            # Check if in front of camera
            if z < 0 or z > 1:
                return None

            # Flip Y for pygame coordinates
            return int(x), int(self.settings.window_height - y)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return None

    def render_info_panel(self, info: dict[str, Any], position: tuple[int, int] = (20, 20)) -> None:
        """Render info panel (Delegated to UIRenderer)."""
        pass  # UI Renderer handles panels now via render_sidebar or similar

    def render_status_bar(self, text: str) -> None:
        """Render status bar (Delegated to UIRenderer)."""
        if self.ui_renderer:
            self.ui_renderer.render_status_bar(text)

    def render_help_overlay(self, help_data: dict[str, Any]) -> None:
        """Render help overlay (Delegated to UIRenderer)."""
        if self.ui_renderer:
            self.ui_renderer.render_help_overlay(help_data)

    def render_date_picker(self, picker_data: dict[str, Any]) -> None:
        """Render date picker (Delegated to UIRenderer)."""
        if self.ui_renderer:
            self.ui_renderer.render_date_picker(picker_data)

    def render_educational_panel(self, edu_data: dict[str, Any]) -> None:
        """Render educational panel (Delegated to UIRenderer)."""
        if self.ui_renderer:
            self.ui_renderer.render_educational_panel(edu_data)

    def render_historical_events(self, events_data: dict[str, Any]) -> None:
        """Render historical events (Delegated to UIRenderer)."""
        if self.ui_renderer:
            self.ui_renderer.render_historical_events(events_data)

    def render_immersion_checklist(self, checklist_data: dict[str, Any]) -> None:
        """Render immersion checklist (Delegated to UIRenderer)."""
        if self.ui_renderer:
            self.ui_renderer.render_immersion_checklist(checklist_data)

    def cleanup(self) -> None:
        """Clean up OpenGL resources."""
        if self._sphere_list:
            glDeleteLists(self._sphere_list, 1)
        if self._circle_list:
            glDeleteLists(self._circle_list, 1)

        pygame.quit()

    def get_fps(self) -> float:
        """Get current frames per second."""
        return self.clock.get_fps() if self.clock else 0.0

    def render_sidebar(
        self, sidebar_data: dict[str, Any], content_data: dict[str, Any] | None
    ) -> None:
        """Render sidebar (Delegated to UIRenderer)."""
        if self.ui_renderer:
            self.ui_renderer.render_sidebar(sidebar_data, content_data)

    def render_unified_controls(self, ctrl_data: dict[str, Any], time_data: dict[str, Any]) -> None:
        """Render unified controls (Delegated to UIRenderer)."""
        if self.ui_renderer:
            self.ui_renderer.render_unified_controls(ctrl_data, time_data)

    def render_speed_indicator(self, time_warp: float) -> None:
        """Render speed indicator bar."""
        if self.ui_renderer:
            self.ui_renderer.render_speed_indicator(time_warp)

    def render_compass(self, camera_yaw: float) -> None:
        """Render compass."""
        if self.ui_renderer:
            self.ui_renderer.render_compass(camera_yaw)
