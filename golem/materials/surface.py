from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from golem.materials.core import AppearanceMaterialId, LinearRgb


class SurfaceMaterialKind(StrEnum):
    """Closed material-class vocabulary (wave-3, R8).

    Every surface color decodes into exactly one class. Absence of a
    `class` key decodes as OPAQUE with the pre-class parameterization, so
    specs authored before the vocabulary existed compile byte-identical.
    """

    OPAQUE = "opaque"
    EMISSIVE = "emissive"
    TRANSLUCENT = "translucent"


@dataclass(frozen=True)
class OpaqueSurfaceClass:
    kind: SurfaceMaterialKind = SurfaceMaterialKind.OPAQUE


@dataclass(frozen=True)
class EmissiveSurfaceClass:
    """Shading-exempt surface rendered at full authored display luminance."""

    intensity: float
    kind: SurfaceMaterialKind = SurfaceMaterialKind.EMISSIVE


@dataclass(frozen=True)
class TranslucentSurfaceClass:
    """Surface composited over what lies behind at authored opacity."""

    opacity: float
    kind: SurfaceMaterialKind = SurfaceMaterialKind.TRANSLUCENT


type SurfaceClass = (
    OpaqueSurfaceClass | EmissiveSurfaceClass | TranslucentSurfaceClass
)


class SurfaceColorVariationKind(StrEnum):
    TRIPLANAR_VALUE_NOISE = "triplanar_value_noise"


class SurfaceColorOwner(StrEnum):
    LOCAL = "local"
    APPEARANCE_PALETTE = "appearance_palette"


class SurfaceColorRule(StrEnum):
    SURFACE_COLOR_OBJECT = "surface_color_object"
    UNKNOWN_FIELD = "unknown_field"
    REQUIRED_FIELD = "required_field"
    FINITE_NUMBER = "finite_number"
    BOUNDED_NUMBER = "bounded_number"
    RGB_CHANNEL_COUNT = "rgb_channel_count"
    VARIATION_OBJECT = "variation_object"
    VARIATION_KIND = "variation_kind"
    MATERIAL_CLASS = "material_class"


class AppearancePaletteRule(StrEnum):
    PALETTE_OBJECT = "palette_object"
    NON_EMPTY_PALETTE = "non_empty_palette"
    KNOWN_MATERIAL = "known_material"
    DISTINCT_LOCAL_COLOR_OWNER = "distinct_local_color_owner"


@dataclass(frozen=True)
class SurfaceColorVariation:
    kind: SurfaceColorVariationKind
    wavelength_world: float
    amplitude: float


@dataclass(frozen=True)
class SurfaceColorRecipe:
    tint_linear_rgb: LinearRgb
    variation: SurfaceColorVariation | None = None


@dataclass(frozen=True)
class SeededSurfaceColor:
    recipe: SurfaceColorRecipe
    seed: int
    owner: SurfaceColorOwner = SurfaceColorOwner.LOCAL
    surface_class: SurfaceClass = OpaqueSurfaceClass()


@dataclass(frozen=True)
class ClassedSurfaceColor:
    """A decoded recipe paired with its material class."""

    recipe: SurfaceColorRecipe
    surface_class: SurfaceClass = OpaqueSurfaceClass()


@dataclass(frozen=True)
class AppearancePaletteEntry[color]:
    material_id: AppearanceMaterialId
    color: color


@dataclass(frozen=True)
class AppearancePalette[color]:
    entries: tuple[AppearancePaletteEntry[color], ...]

    def color_for(
        self,
        material_id: AppearanceMaterialId | str,
    ) -> color | None:
        entry = next(
            filter(
                lambda candidate: candidate.material_id.value == material_id,
                self.entries,
            ),
            None,
        )
        return entry.color if entry is not None else None


@dataclass(frozen=True)
class SurfaceColorObstruction:
    address: str
    rule: SurfaceColorRule
    authored: object
    required: object


@dataclass(frozen=True)
class AcceptedSurfaceColor:
    recipe: SurfaceColorRecipe
    surface_class: SurfaceClass = OpaqueSurfaceClass()


@dataclass(frozen=True)
class RejectedSurfaceColor:
    obstructions: tuple[SurfaceColorObstruction, ...]


@dataclass(frozen=True)
class AppearancePaletteObstruction:
    address: str
    rule: AppearancePaletteRule
    authored: object
    required: object


@dataclass(frozen=True)
class AcceptedAppearancePalette:
    palette: AppearancePalette[ClassedSurfaceColor]


@dataclass(frozen=True)
class RejectedAppearancePalette:
    obstructions: tuple[
        AppearancePaletteObstruction | SurfaceColorObstruction,
        ...,
    ]


type SurfaceColorDecodeResult = AcceptedSurfaceColor | RejectedSurfaceColor
type AppearancePaletteDecodeResult = (
    AcceptedAppearancePalette | RejectedAppearancePalette
)
type SeededAppearancePalette = AppearancePalette[SeededSurfaceColor]
type _NumberResult = float | SurfaceColorObstruction

_LOCAL_SURFACE_COLOR_FIELDS = frozenset(("tint_linear_rgb", "variation"))
_CLASS_SURFACE_COLOR_FIELDS = {
    SurfaceMaterialKind.OPAQUE: frozenset(
        ("class", "tint_linear_rgb", "variation")
    ),
    SurfaceMaterialKind.EMISSIVE: frozenset(
        ("class", "tint_linear_rgb", "intensity")
    ),
    SurfaceMaterialKind.TRANSLUCENT: frozenset(
        ("class", "tint_linear_rgb", "variation", "opacity")
    ),
}
_ANY_CLASS_SURFACE_COLOR_FIELDS = frozenset(
    ("class", "tint_linear_rgb", "variation", "intensity", "opacity")
)


def _unknown_field_obstructions(
    payload: Mapping[object, object],
    allowed: frozenset[str],
    address: str,
) -> tuple[SurfaceColorObstruction, ...]:
    return tuple(
        map(
            lambda key: SurfaceColorObstruction(
                f"{address}/{key}",
                SurfaceColorRule.UNKNOWN_FIELD,
                key,
                tuple(sorted(allowed)),
            ),
            sorted(map(str, frozenset(payload).difference(allowed))),
        )
    )


def _required_field_obstruction(
    payload: Mapping[object, object],
    field: str,
    address: str,
) -> tuple[SurfaceColorObstruction, ...]:
    return (
        ()
        if field in payload
        else (
            SurfaceColorObstruction(
                f"{address}/{field}",
                SurfaceColorRule.REQUIRED_FIELD,
                tuple(sorted(map(str, payload))),
                field,
            ),
        )
    )


def _bounded_number(
    value: object,
    address: str,
    *,
    minimum: float,
    maximum: float | None,
    exclusive_minimum: bool = False,
) -> _NumberResult:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return SurfaceColorObstruction(
            address,
            SurfaceColorRule.FINITE_NUMBER,
            value,
            {"finite": True},
        )
    number = float(value)
    if not isfinite(number):
        return SurfaceColorObstruction(
            address,
            SurfaceColorRule.FINITE_NUMBER,
            value,
            {"finite": True},
        )
    minimum_satisfied = (
        number > minimum if exclusive_minimum else number >= minimum
    )
    maximum_satisfied = maximum is None or number <= maximum
    if not minimum_satisfied or not maximum_satisfied:
        return SurfaceColorObstruction(
            address,
            SurfaceColorRule.BOUNDED_NUMBER,
            number,
            {
                "minimum": minimum,
                "maximum": maximum,
                "exclusive_minimum": exclusive_minimum,
            },
        )
    return number


def _decode_tint(
    value: object,
    address: str,
) -> LinearRgb | tuple[SurfaceColorObstruction, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return (
            SurfaceColorObstruction(
                address,
                SurfaceColorRule.RGB_CHANNEL_COUNT,
                value,
                {"sequence": True, "length": 3},
            ),
        )
    decoded = tuple(
        map(
            lambda item: _bounded_number(
                item[1],
                f"{address}/{item[0]}",
                minimum=0.0,
                maximum=1.0,
            ),
            enumerate(value),
        )
    )
    obstructions = tuple(
        filter(lambda item: isinstance(item, SurfaceColorObstruction), decoded)
    )
    return (
        obstructions
        if obstructions
        else (float(decoded[0]), float(decoded[1]), float(decoded[2]))
    )


def _decode_variation(
    value: object,
    address: str,
) -> SurfaceColorVariation | tuple[SurfaceColorObstruction, ...]:
    if not isinstance(value, Mapping):
        return (
            SurfaceColorObstruction(
                address,
                SurfaceColorRule.VARIATION_OBJECT,
                value,
                "object",
            ),
        )
    required = ("kind", "wavelength_world", "amplitude")
    structural_obstructions = (
        *_unknown_field_obstructions(value, frozenset(required), address),
        *tuple(
            chain.from_iterable(
                map(
                    lambda field: _required_field_obstruction(
                        value,
                        field,
                        address,
                    ),
                    required,
                )
            )
        ),
    )
    kind = (
        SurfaceColorVariationKind.TRIPLANAR_VALUE_NOISE
        if value.get("kind")
        == SurfaceColorVariationKind.TRIPLANAR_VALUE_NOISE.value
        else SurfaceColorObstruction(
            f"{address}/kind",
            SurfaceColorRule.VARIATION_KIND,
            value.get("kind"),
            tuple(kind.value for kind in SurfaceColorVariationKind),
        )
        if "kind" in value
        else None
    )
    wavelength = (
        _bounded_number(
            value["wavelength_world"],
            f"{address}/wavelength_world",
            minimum=0.0,
            maximum=None,
            exclusive_minimum=True,
        )
        if "wavelength_world" in value
        else None
    )
    amplitude = (
        _bounded_number(
            value["amplitude"],
            f"{address}/amplitude",
            minimum=0.0,
            maximum=1.0,
        )
        if "amplitude" in value
        else None
    )
    obstructions = (
        *structural_obstructions,
        *tuple(
            filter(
                lambda item: isinstance(item, SurfaceColorObstruction),
                (kind, wavelength, amplitude),
            )
        ),
    )
    return (
        obstructions
        if obstructions
        else SurfaceColorVariation(
            SurfaceColorVariationKind(kind),
            float(wavelength),
            float(amplitude),
        )
    )


def _decode_material_class(
    payload: Mapping[object, object],
    address: str,
) -> SurfaceMaterialKind | SurfaceColorObstruction:
    authored = payload.get("class", SurfaceMaterialKind.OPAQUE.value)
    legal = tuple(sorted(kind.value for kind in SurfaceMaterialKind))
    return (
        SurfaceMaterialKind(authored)
        if authored in legal
        else SurfaceColorObstruction(
            f"{address}/class",
            SurfaceColorRule.MATERIAL_CLASS,
            authored,
            legal,
        )
    )


def decode_surface_color(
    payload: object,
    address: str = "/surface_color",
    *,
    allow_class: bool = False,
) -> SurfaceColorDecodeResult:
    """Decode one surface-color payload.

    `allow_class` opens the material-class vocabulary (wave-3, R8); it is
    legal on appearance-palette entries only. A local element surface
    color keeps the pre-class field set, so a `class` key there is an
    unknown-field obstruction rather than a silently dropped declaration.
    """
    if not isinstance(payload, Mapping):
        return RejectedSurfaceColor(
            (
                SurfaceColorObstruction(
                    address,
                    SurfaceColorRule.SURFACE_COLOR_OBJECT,
                    payload,
                    "object",
                ),
            )
        )
    material_class = (
        _decode_material_class(payload, address)
        if allow_class
        else SurfaceMaterialKind.OPAQUE
    )
    kind = (
        material_class
        if isinstance(material_class, SurfaceMaterialKind)
        else None
    )
    allowed = (
        _LOCAL_SURFACE_COLOR_FIELDS
        if not allow_class
        else (
            _CLASS_SURFACE_COLOR_FIELDS[kind]
            if kind is not None
            else _ANY_CLASS_SURFACE_COLOR_FIELDS
        )
    )
    structural_obstructions = (
        *(
            (material_class,)
            if isinstance(material_class, SurfaceColorObstruction)
            else ()
        ),
        *_unknown_field_obstructions(payload, allowed, address),
        *_required_field_obstruction(payload, "tint_linear_rgb", address),
        *(
            _required_field_obstruction(payload, "opacity", address)
            if kind is SurfaceMaterialKind.TRANSLUCENT
            else ()
        ),
    )
    tint = (
        _decode_tint(
            payload["tint_linear_rgb"],
            f"{address}/tint_linear_rgb",
        )
        if "tint_linear_rgb" in payload
        else None
    )
    variation = (
        _decode_variation(payload["variation"], f"{address}/variation")
        if "variation" in payload and "variation" in allowed
        else None
    )
    intensity = (
        _bounded_number(
            payload["intensity"],
            f"{address}/intensity",
            minimum=0.0,
            maximum=None,
            exclusive_minimum=True,
        )
        if kind is SurfaceMaterialKind.EMISSIVE and "intensity" in payload
        else None
    )
    opacity = (
        _bounded_number(
            payload["opacity"],
            f"{address}/opacity",
            minimum=0.0,
            maximum=1.0,
        )
        if kind is SurfaceMaterialKind.TRANSLUCENT and "opacity" in payload
        else None
    )
    obstructions = (
        *structural_obstructions,
        *(
            tint
            if isinstance(tint, tuple)
            and tint
            and isinstance(tint[0], SurfaceColorObstruction)
            else ()
        ),
        *(variation if isinstance(variation, tuple) else ()),
        *(
            (intensity,)
            if isinstance(intensity, SurfaceColorObstruction)
            else ()
        ),
        *(
            (opacity,)
            if isinstance(opacity, SurfaceColorObstruction)
            else ()
        ),
    )
    surface_class: SurfaceClass = (
        OpaqueSurfaceClass()
        if kind is not SurfaceMaterialKind.EMISSIVE
        and kind is not SurfaceMaterialKind.TRANSLUCENT
        else (
            EmissiveSurfaceClass(
                float(intensity)
                if isinstance(intensity, float)
                else 1.0
            )
            if kind is SurfaceMaterialKind.EMISSIVE
            else TranslucentSurfaceClass(
                float(opacity) if isinstance(opacity, float) else 1.0
            )
        )
    )
    return (
        RejectedSurfaceColor(obstructions)
        if obstructions
        else AcceptedSurfaceColor(
            SurfaceColorRecipe(
                (float(tint[0]), float(tint[1]), float(tint[2])),
                variation if isinstance(variation, SurfaceColorVariation) else None,
            ),
            surface_class,
        )
    )


def decode_appearance_palette(
    payload: object,
    address: str = "/appearance_palette",
) -> AppearancePaletteDecodeResult:
    if not isinstance(payload, Mapping):
        return RejectedAppearancePalette(
            (
                AppearancePaletteObstruction(
                    address,
                    AppearancePaletteRule.PALETTE_OBJECT,
                    payload,
                    "object",
                ),
            )
        )
    if not payload:
        return RejectedAppearancePalette(
            (
                AppearancePaletteObstruction(
                    address,
                    AppearancePaletteRule.NON_EMPTY_PALETTE,
                    payload,
                    {"object": True, "minimum_property_count": 1},
                ),
            )
        )
    known_material_ids = tuple(AppearanceMaterialId)
    known_identifiers = frozenset(
        map(lambda material_id: material_id.value, known_material_ids)
    )
    unknown_obstructions = tuple(
        map(
            lambda identifier: AppearancePaletteObstruction(
                f"{address}/{identifier}",
                AppearancePaletteRule.KNOWN_MATERIAL,
                identifier,
                tuple(sorted(known_identifiers)),
            ),
            sorted(map(str, frozenset(payload).difference(known_identifiers))),
        )
    )
    decoded_entries = tuple(
        map(
            lambda material_id: (
                material_id,
                decode_surface_color(
                    payload[material_id.value],
                    f"{address}/{material_id.value}",
                    allow_class=True,
                ),
            ),
            filter(
                lambda material_id: material_id.value in payload,
                known_material_ids,
            ),
        )
    )
    color_obstructions = tuple(
        chain.from_iterable(
            map(
                lambda entry: (
                    entry[1].obstructions
                    if isinstance(entry[1], RejectedSurfaceColor)
                    else ()
                ),
                decoded_entries,
            )
        )
    )
    obstructions = (*unknown_obstructions, *color_obstructions)
    return (
        RejectedAppearancePalette(obstructions)
        if obstructions
        else AcceptedAppearancePalette(
            AppearancePalette(
                tuple(
                    map(
                        lambda entry: AppearancePaletteEntry(
                            entry[0],
                            ClassedSurfaceColor(
                                entry[1].recipe,
                                entry[1].surface_class,
                            ),
                        ),
                        decoded_entries,
                    )
                )
            )
        )
    )


def seed_surface_color(
    recipe: SurfaceColorRecipe,
    source_digest: str,
    record_id: str,
    *,
    owner: SurfaceColorOwner = SurfaceColorOwner.LOCAL,
    surface_class: SurfaceClass = OpaqueSurfaceClass(),
) -> SeededSurfaceColor:
    seed_bytes = hashlib.sha256(
        b"golem.surface-color.v1\0"
        + bytes.fromhex(source_digest)
        + b"\0"
        + record_id.encode("utf-8")
    ).digest()
    return SeededSurfaceColor(
        recipe,
        int.from_bytes(seed_bytes[:8], "big"),
        owner,
        surface_class,
    )


def seed_appearance_palette(
    palette: AppearancePalette[ClassedSurfaceColor],
    source_digest: str,
    element_id: str,
) -> SeededAppearancePalette:
    return AppearancePalette(
        tuple(
            map(
                lambda entry: AppearancePaletteEntry(
                    entry.material_id,
                    seed_surface_color(
                        entry.color.recipe,
                        source_digest,
                        f"{element_id}:{entry.material_id.value}",
                        owner=SurfaceColorOwner.APPEARANCE_PALETTE,
                        surface_class=entry.color.surface_class,
                    ),
                ),
                palette.entries,
            )
        )
    )


def _lattice_values(
    first: np.ndarray,
    second: np.ndarray,
    seed: int,
) -> np.ndarray:
    first_bits = np.asarray(first, dtype=np.int64).astype(np.uint64)
    second_bits = np.asarray(second, dtype=np.int64).astype(np.uint64)
    combined = (
        np.uint64(seed)
        ^ first_bits * np.uint64(0x9E3779B185EBCA87)
        ^ second_bits * np.uint64(0xC2B2AE3D27D4EB4F)
    )
    folded = (combined ^ (combined >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    scrambled = (folded ^ (folded >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    hashed = scrambled ^ (scrambled >> np.uint64(31))
    return (hashed >> np.uint64(40)).astype(np.float64) / float(1 << 24)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def _value_noise_2d(
    first: np.ndarray,
    second: np.ndarray,
    seed: int,
) -> np.ndarray:
    first_floor = np.floor(first).astype(np.int64)
    second_floor = np.floor(second).astype(np.int64)
    first_weight = _smoothstep(first - first_floor)
    second_weight = _smoothstep(second - second_floor)
    lower = (
        _lattice_values(first_floor, second_floor, seed)
        * (1.0 - first_weight)
        + _lattice_values(first_floor + 1, second_floor, seed) * first_weight
    )
    upper = (
        _lattice_values(first_floor, second_floor + 1, seed)
        * (1.0 - first_weight)
        + _lattice_values(first_floor + 1, second_floor + 1, seed)
        * first_weight
    )
    return lower * (1.0 - second_weight) + upper * second_weight


def _triplanar_noise(
    vertices: np.ndarray,
    normals: np.ndarray,
    variation: SurfaceColorVariation,
    seed: int,
) -> np.ndarray:
    coordinates = np.asarray(vertices, dtype=np.float64) / variation.wavelength_world
    normal_weights = np.abs(np.asarray(normals, dtype=np.float64))
    weight_sums = normal_weights.sum(axis=1, keepdims=True)
    normalized_weights = np.where(
        weight_sums > 0.0,
        normal_weights / np.where(weight_sums > 0.0, weight_sums, 1.0),
        np.full_like(normal_weights, 1.0 / 3.0),
    )
    yz = _value_noise_2d(
        coordinates[:, 1],
        coordinates[:, 2],
        seed ^ 0xA24BAED4963EE407,
    )
    xz = _value_noise_2d(
        coordinates[:, 0],
        coordinates[:, 2],
        seed ^ 0x9FB21C651E98DF25,
    )
    xy = _value_noise_2d(
        coordinates[:, 0],
        coordinates[:, 1],
        seed ^ 0xB7E151628AED2A6B,
    )
    return (
        yz * normalized_weights[:, 0]
        + xz * normalized_weights[:, 1]
        + xy * normalized_weights[:, 2]
    )


def derive_linear_vertex_rgb(
    vertices: NDArray[np.float64],
    normals: NDArray[np.float64],
    surface_color: SeededSurfaceColor,
) -> NDArray[np.float64]:
    recipe = surface_color.recipe
    variation = recipe.variation
    intensity = (
        np.ones(len(vertices), dtype=np.float64)
        if variation is None
        else 1.0
        + variation.amplitude
        * (
            2.0
            * _triplanar_noise(
                vertices,
                normals,
                variation,
                surface_color.seed,
            )
            - 1.0
        )
    )
    return (
        np.asarray(recipe.tint_linear_rgb, dtype=np.float64)[None, :]
        * intensity[:, None]
    )


def derive_vertex_colors(
    vertices: NDArray[np.float64],
    normals: NDArray[np.float64],
    surface_color: SeededSurfaceColor,
) -> NDArray[np.uint8]:
    """Quantize linear-light RGB plus class alpha for glTF COLOR_0 export."""
    linear_rgb = derive_linear_vertex_rgb(vertices, normals, surface_color)
    encoded_rgb = np.floor(
        np.clip(linear_rgb, 0.0, 1.0) * 255.0 + 0.5
    ).astype(np.uint8)
    surface_class = surface_color.surface_class
    alpha = (
        int(np.floor(surface_class.opacity * 255.0 + 0.5))
        if isinstance(surface_class, TranslucentSurfaceClass)
        else 255
    )
    colors = np.concatenate(
        (encoded_rgb, np.full((len(encoded_rgb), 1), alpha, dtype=np.uint8)),
        axis=1,
    )
    return np.frombuffer(colors.tobytes(), dtype=np.uint8).reshape(colors.shape)


def surface_color_to_dict(
    surface_color: SeededSurfaceColor,
) -> dict[str, object]:
    variation = surface_color.recipe.variation
    surface_class = surface_color.surface_class
    return {
        "tint_linear_rgb": surface_color.recipe.tint_linear_rgb,
        **(
            {
                "variation": {
                    "kind": variation.kind.value,
                    "wavelength_world": variation.wavelength_world,
                    "amplitude": variation.amplitude,
                }
            }
            if variation is not None
            else {}
        ),
        **(
            {"class": "emissive", "intensity": surface_class.intensity}
            if isinstance(surface_class, EmissiveSurfaceClass)
            else (
                {"class": "translucent", "opacity": surface_class.opacity}
                if isinstance(surface_class, TranslucentSurfaceClass)
                else {}
            )
        ),
        "derived_seed": f"{surface_color.seed:016x}",
    }
