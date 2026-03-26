from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from config import ExperimentConfig, RegionConfig
from core_ops import subsystem_dimensions_from_qubits


# ============================================================
# Small helpers
# ============================================================

RegionLike = Union[str, int, RegionConfig]
PairLike = Union[Tuple[int, int], Tuple[str, str]]


def _ensure_nonnegative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _ensure_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _sorted_pair(i: int, j: int) -> Tuple[int, int]:
    i = int(i)
    j = int(j)
    return (i, j) if i <= j else (j, i)


def _as_region_obj(region: RegionLike, cfg: ExperimentConfig) -> RegionConfig:
    if isinstance(region, RegionConfig):
        return region
    if isinstance(region, str):
        return cfg.region_by_name(region)
    if isinstance(region, int):
        idx = int(region)
        if idx < 0 or idx >= cfg.num_regions:
            raise ValueError(f"Region index must be in [0, {cfg.num_regions - 1}], got {idx}.")
        return cfg.regions[idx]
    raise TypeError(f"Unsupported region specifier type: {type(region)!r}")


def _as_region_index(region: RegionLike, cfg: ExperimentConfig) -> int:
    if isinstance(region, int):
        idx = int(region)
        if idx < 0 or idx >= cfg.num_regions:
            raise ValueError(f"Region index must be in [0, {cfg.num_regions - 1}], got {idx}.")
        return idx
    if isinstance(region, str):
        return cfg.region_index(region)
    if isinstance(region, RegionConfig):
        return cfg.region_index(region.name)
    raise TypeError(f"Unsupported region specifier type: {type(region)!r}")


def _validate_pair(pair: Tuple[int, int], cfg: ExperimentConfig) -> Tuple[int, int]:
    if len(pair) != 2:
        raise ValueError(f"pair must have length 2, got {pair}.")
    i, j = int(pair[0]), int(pair[1])
    if i == j:
        raise ValueError(f"pair must contain two distinct region indices, got {pair}.")
    if i < 0 or i >= cfg.num_regions or j < 0 or j >= cfg.num_regions:
        raise ValueError(
            f"pair entries must lie in [0, {cfg.num_regions - 1}], got {pair}."
        )
    return _sorted_pair(i, j)


# ============================================================
# Data containers
# ============================================================

@dataclass(frozen=True)
class RegionInfo:
    """
    Derived structural information for one region.

    Attributes
    ----------
    index :
        Region index in cfg.regions.
    name :
        Region name.
    sites :
        Global site indices belonging to the region.
    qubits :
        Total number of qubits in the region.
    dim :
        Hilbert-space dimension of the region.
    site_dims :
        Site Hilbert-space dimensions inside the region.
    shots :
        Number of measurement shots assigned to the region.
    povm_type :
        POVM family name for the region.
    povm_num_outcomes :
        Requested number of POVM outcomes, if specified.
    neighbors :
        Tuple of neighboring region indices that overlap with this region.
    """
    index: int
    name: str
    sites: Tuple[int, ...]
    qubits: int
    dim: int
    site_dims: Tuple[int, ...]
    shots: int
    povm_type: str
    povm_num_outcomes: int | None
    neighbors: Tuple[int, ...]


@dataclass(frozen=True)
class OverlapInfo:
    """
    Structural information for one overlapping region pair.

    All ordering conventions are canonical:
    the overlap sites are listed in increasing global site index order.

    Attributes
    ----------
    pair :
        Region-index pair (i, j) with i < j.
    region_names :
        Corresponding region names.
    overlap_sites :
        Shared global site indices, in increasing order.
    overlap_qubits :
        Total number of qubits in the overlap.
    overlap_dim :
        Hilbert-space dimension of the overlap subsystem.
    overlap_site_dims :
        Site Hilbert-space dimensions of the overlap subsystem.
    local_keep_i :
        Local subsystem indices inside region i that correspond to overlap_sites.
    local_keep_j :
        Local subsystem indices inside region j that correspond to overlap_sites.
    """
    pair: Tuple[int, int]
    region_names: Tuple[str, str]
    overlap_sites: Tuple[int, ...]
    overlap_qubits: int
    overlap_dim: int
    overlap_site_dims: Tuple[int, ...]
    local_keep_i: Tuple[int, ...]
    local_keep_j: Tuple[int, ...]


# ============================================================
# Main structure object
# ============================================================

class RegionGraph:
    """
    Structural wrapper around an ExperimentConfig.

    This object centralizes all region / overlap metadata so later modules
    do not have to repeatedly recompute it.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

        self.num_sites: int = cfg.num_sites
        self.num_regions: int = cfg.num_regions
        self.qubits_per_site: Tuple[int, ...] = tuple(int(q) for q in cfg.qubits_per_site)
        self.site_dims: Tuple[int, ...] = tuple(subsystem_dimensions_from_qubits(self.qubits_per_site))
        self.total_qubits: int = int(sum(self.qubits_per_site))

        self.region_infos: Tuple[RegionInfo, ...] = self._build_region_infos()
        self.region_info_by_index: Dict[int, RegionInfo] = {
            info.index: info for info in self.region_infos
        }
        self.region_info_by_name: Dict[str, RegionInfo] = {
            info.name: info for info in self.region_infos
        }

        self.overlap_infos: Tuple[OverlapInfo, ...] = self._build_overlap_infos()
        self.overlap_info_by_pair: Dict[Tuple[int, int], OverlapInfo] = {
            info.pair: info for info in self.overlap_infos
        }
        self.overlap_info_by_names: Dict[Tuple[str, str], OverlapInfo] = {
            _sorted_name_pair(*info.region_names): info for info in self.overlap_infos
        }

    # --------------------------------------------------------
    # Internal builders
    # --------------------------------------------------------

    def _build_region_infos(self) -> Tuple[RegionInfo, ...]:
        infos: List[RegionInfo] = []
        for idx, region in enumerate(self.cfg.regions):
            qubits = self.cfg.region_qubits(region)
            dim = self.cfg.region_dimension(region)
            site_dims = self.cfg.region_site_dimensions(region)
            neighbors = self.cfg.neighbors(idx)

            info = RegionInfo(
                index=idx,
                name=region.name,
                sites=tuple(region.sites),
                qubits=int(qubits),
                dim=int(dim),
                site_dims=tuple(int(d) for d in site_dims),
                shots=int(region.shots),
                povm_type=str(region.povm_type),
                povm_num_outcomes=(
                    None if region.povm_num_outcomes is None else int(region.povm_num_outcomes)
                ),
                neighbors=tuple(int(j) for j in neighbors),
            )
            infos.append(info)
        return tuple(infos)

    def _build_overlap_infos(self) -> Tuple[OverlapInfo, ...]:
        infos: List[OverlapInfo] = []

        for i, j in self.cfg.overlap_pairs():
            region_i = self.cfg.regions[i]
            region_j = self.cfg.regions[j]

            overlap_sites = self.cfg.region_overlap_sites(region_i, region_j)
            overlap_qubits = self.cfg.region_overlap_qubits(region_i, region_j)
            overlap_dim = 2 ** overlap_qubits if overlap_qubits > 0 else 1
            overlap_site_dims = tuple(2 ** self.qubits_per_site[s] for s in overlap_sites)

            local_keep_i = tuple(region_i.sites.index(s) for s in overlap_sites)
            local_keep_j = tuple(region_j.sites.index(s) for s in overlap_sites)

            info = OverlapInfo(
                pair=(int(i), int(j)),
                region_names=(region_i.name, region_j.name),
                overlap_sites=tuple(int(s) for s in overlap_sites),
                overlap_qubits=int(overlap_qubits),
                overlap_dim=int(overlap_dim),
                overlap_site_dims=tuple(int(d) for d in overlap_site_dims),
                local_keep_i=tuple(int(k) for k in local_keep_i),
                local_keep_j=tuple(int(k) for k in local_keep_j),
            )
            infos.append(info)

        return tuple(infos)

    # --------------------------------------------------------
    # Region accessors
    # --------------------------------------------------------

    def region_info(self, region: RegionLike) -> RegionInfo:
        """Return RegionInfo for a region specified by name, index, or RegionConfig."""
        idx = _as_region_index(region, self.cfg)
        return self.region_info_by_index[idx]

    def region_name(self, region: RegionLike) -> str:
        """Return canonical region name."""
        return self.region_info(region).name

    def region_index(self, region: RegionLike) -> int:
        """Return canonical region index."""
        return self.region_info(region).index

    def region_sites(self, region: RegionLike) -> Tuple[int, ...]:
        """Return global sites belonging to the region."""
        return self.region_info(region).sites

    def region_qubits(self, region: RegionLike) -> int:
        """Return total number of qubits in the region."""
        return self.region_info(region).qubits

    def region_dim(self, region: RegionLike) -> int:
        """Return Hilbert-space dimension of the region."""
        return self.region_info(region).dim

    def region_site_dims(self, region: RegionLike) -> Tuple[int, ...]:
        """Return site Hilbert-space dimensions inside the region."""
        return self.region_info(region).site_dims

    def region_shots(self, region: RegionLike) -> int:
        """Return number of measurement shots assigned to the region."""
        return self.region_info(region).shots

    def region_povm_type(self, region: RegionLike) -> str:
        """Return POVM type string for the region."""
        return self.region_info(region).povm_type

    def region_povm_num_outcomes(self, region: RegionLike) -> int | None:
        """Return configured number of POVM outcomes, if specified."""
        return self.region_info(region).povm_num_outcomes

    def neighbors(self, region: RegionLike) -> Tuple[int, ...]:
        """Return neighboring region indices."""
        return self.region_info(region).neighbors

    def neighbor_names(self, region: RegionLike) -> Tuple[str, ...]:
        """Return neighboring region names."""
        return tuple(self.region_info_by_index[j].name for j in self.neighbors(region))

    # --------------------------------------------------------
    # Overlap accessors
    # --------------------------------------------------------

    def has_overlap(self, region_a: RegionLike, region_b: RegionLike) -> bool:
        """Return True iff the two regions overlap."""
        i = _as_region_index(region_a, self.cfg)
        j = _as_region_index(region_b, self.cfg)
        if i == j:
            return False
        return _sorted_pair(i, j) in self.overlap_info_by_pair

    def overlap_info(self, region_a: RegionLike, region_b: RegionLike) -> OverlapInfo:
        """Return OverlapInfo for a pair of overlapping regions."""
        i = _as_region_index(region_a, self.cfg)
        j = _as_region_index(region_b, self.cfg)
        if i == j:
            raise ValueError("A region does not form an overlap pair with itself.")
        pair = _sorted_pair(i, j)
        if pair not in self.overlap_info_by_pair:
            name_i = self.cfg.regions[pair[0]].name
            name_j = self.cfg.regions[pair[1]].name
            raise ValueError(f"Regions '{name_i}' and '{name_j}' do not overlap.")
        return self.overlap_info_by_pair[pair]

    def overlap_info_from_pair(self, pair: Tuple[int, int]) -> OverlapInfo:
        """Return OverlapInfo from an explicit region-index pair."""
        key = _validate_pair(pair, self.cfg)
        if key not in self.overlap_info_by_pair:
            name_i = self.cfg.regions[key[0]].name
            name_j = self.cfg.regions[key[1]].name
            raise ValueError(f"Regions '{name_i}' and '{name_j}' do not overlap.")
        return self.overlap_info_by_pair[key]

    def overlap_sites(self, region_a: RegionLike, region_b: RegionLike) -> Tuple[int, ...]:
        """Return shared global site indices in canonical increasing order."""
        return self.overlap_info(region_a, region_b).overlap_sites

    def overlap_qubits(self, region_a: RegionLike, region_b: RegionLike) -> int:
        """Return number of qubits in the overlap."""
        return self.overlap_info(region_a, region_b).overlap_qubits

    def overlap_dim(self, region_a: RegionLike, region_b: RegionLike) -> int:
        """Return overlap Hilbert-space dimension."""
        return self.overlap_info(region_a, region_b).overlap_dim

    def overlap_site_dims(self, region_a: RegionLike, region_b: RegionLike) -> Tuple[int, ...]:
        """Return overlap site Hilbert-space dimensions."""
        return self.overlap_info(region_a, region_b).overlap_site_dims

    def local_keep_indices(
        self,
        region_a: RegionLike,
        region_b: RegionLike,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Return local subsystem indices needed to reduce each region onto the overlap.
        """
        info = self.overlap_info(region_a, region_b)

        i = _as_region_index(region_a, self.cfg)
        j = _as_region_index(region_b, self.cfg)

        if (i, j) == info.pair:
            return info.local_keep_i, info.local_keep_j
        return info.local_keep_j, info.local_keep_i

    # --------------------------------------------------------
    # Collection helpers
    # --------------------------------------------------------

    def region_names(self) -> Tuple[str, ...]:
        """Return all region names in config order."""
        return tuple(info.name for info in self.region_infos)

    def region_indices(self) -> Tuple[int, ...]:
        """Return all region indices."""
        return tuple(info.index for info in self.region_infos)

    def overlap_pairs(self) -> Tuple[Tuple[int, int], ...]:
        """Return all overlapping region-index pairs."""
        return tuple(info.pair for info in self.overlap_infos)

    def overlap_name_pairs(self) -> Tuple[Tuple[str, str], ...]:
        """Return all overlapping region-name pairs."""
        return tuple(info.region_names for info in self.overlap_infos)

    def regions_touching_site(self, site: int) -> Tuple[int, ...]:
        """Return region indices whose support contains the given site."""
        site = _ensure_nonnegative_int(site, "site")
        if site >= self.num_sites:
            raise ValueError(f"site must be in [0, {self.num_sites - 1}], got {site}.")
        out = [info.index for info in self.region_infos if site in info.sites]
        return tuple(out)

    def region_name_to_index(self) -> Dict[str, int]:
        """Return mapping region name -> region index."""
        return {info.name: info.index for info in self.region_infos}

    def index_to_region_name(self) -> Dict[int, str]:
        """Return mapping region index -> region name."""
        return {info.index: info.name for info in self.region_infos}

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate_region_mapping_keys(self, mapping: Mapping[str, object], name: str) -> None:
        """
        Validate that a dictionary keyed by region name matches cfg.regions exactly.
        """
        expected = set(self.region_names())
        provided = set(mapping.keys())
        missing = expected - provided
        extra = provided - expected
        if missing:
            raise ValueError(f"{name} is missing entries for regions: {sorted(missing)}.")
        if extra:
            raise ValueError(f"{name} has unexpected region names: {sorted(extra)}.")

    def validate_pair_mapping_keys(
        self,
        mapping: Mapping[Tuple[str, str], object] | Mapping[Tuple[int, int], object],
        name: str,
    ) -> None:
        """
        Validate that a pair-keyed dictionary matches the overlap pairs exactly.
        """
        keys = set(mapping.keys())

        valid_index_pairs = set(self.overlap_pairs())
        valid_name_pairs = set(self.overlap_name_pairs())

        if keys == valid_index_pairs or keys == valid_name_pairs:
            return

        raise ValueError(
            f"{name} keys do not match overlap pairs. "
            f"Expected either {sorted(valid_index_pairs)} or {sorted(valid_name_pairs)}, got {sorted(keys)}."
        )

    # --------------------------------------------------------
    # Pretty printing
    # --------------------------------------------------------

    def summary_dict(self) -> Dict[str, object]:
        """Return a compact summary dictionary."""
        return {
            "num_sites": self.num_sites,
            "site_dims": self.site_dims,
            "total_qubits": self.total_qubits,
            "num_regions": self.num_regions,
            "region_names": self.region_names(),
            "overlap_pairs": self.overlap_pairs(),
            "overlap_name_pairs": self.overlap_name_pairs(),
        }

    def pretty_print(self) -> None:
        """Print a readable summary of regions and overlaps."""
        print("=" * 72)
        print("RegionGraph")
        print("-" * 72)
        print(f"Sites: {self.num_sites}")
        print(f"Site dims: {self.site_dims}")
        print(f"Total qubits: {self.total_qubits}")
        print(f"Regions: {self.num_regions}")
        print(f"Overlaps: {len(self.overlap_infos)}")
        print("-" * 72)
        for info in self.region_infos:
            print(
                f"[Region {info.index}] name={info.name}, sites={info.sites}, "
                f"qubits={info.qubits}, dim={info.dim}, shots={info.shots}, "
                f"neighbors={info.neighbors}"
            )
        if len(self.overlap_infos) > 0:
            print("-" * 72)
            for info in self.overlap_infos:
                print(
                    f"[Overlap {info.pair}] names={info.region_names}, "
                    f"sites={info.overlap_sites}, qubits={info.overlap_qubits}, "
                    f"dim={info.overlap_dim}, keep_i={info.local_keep_i}, keep_j={info.local_keep_j}"
                )
        print("=" * 72)


# ============================================================
# Standalone convenience helpers
# ============================================================

def _sorted_name_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def build_region_graph(cfg: ExperimentConfig) -> RegionGraph:
    """Convenience constructor."""
    return RegionGraph(cfg)


def region_name_to_index_map(cfg: ExperimentConfig) -> Dict[str, int]:
    """Return mapping region name -> index."""
    return RegionGraph(cfg).region_name_to_index()


def region_index_to_name_map(cfg: ExperimentConfig) -> Dict[int, str]:
    """Return mapping region index -> name."""
    return RegionGraph(cfg).index_to_region_name()


def all_region_names(cfg: ExperimentConfig) -> Tuple[str, ...]:
    """Return all region names in config order."""
    return RegionGraph(cfg).region_names()


def all_overlap_pairs(cfg: ExperimentConfig) -> Tuple[Tuple[int, int], ...]:
    """Return all overlapping region-index pairs."""
    return RegionGraph(cfg).overlap_pairs()


def all_overlap_name_pairs(cfg: ExperimentConfig) -> Tuple[Tuple[str, str], ...]:
    """Return all overlapping region-name pairs."""
    return RegionGraph(cfg).overlap_name_pairs()


def regions_touching_site(cfg: ExperimentConfig, site: int) -> Tuple[int, ...]:
    """Return region indices containing the given site."""
    return RegionGraph(cfg).regions_touching_site(site)


# ============================================================
# Lightweight self-tests
# ============================================================

def _self_test_basic_graph() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)

    assert graph.num_sites == 3
    assert graph.num_regions == 2
    assert graph.region_names() == ("R0", "R1")
    assert graph.overlap_pairs() == ((0, 1),)
    assert graph.overlap_name_pairs() == (("R0", "R1"),)

    info0 = graph.region_info("R0")
    info1 = graph.region_info(1)

    assert info0.sites == (0, 1)
    assert info1.sites == (1, 2)
    assert info0.qubits == 2
    assert info1.dim == 4

    ov = graph.overlap_info("R0", "R1")
    assert ov.overlap_sites == (1,)
    assert ov.overlap_qubits == 1
    assert ov.overlap_dim == 2
    assert ov.local_keep_i == (1,)
    assert ov.local_keep_j == (0,)


def _self_test_site_membership() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)

    assert graph.regions_touching_site(0) == (0,)
    assert graph.regions_touching_site(1) == (0, 1)
    assert graph.regions_touching_site(2) == (1,)


def _self_test_mapping_validation() -> None:
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)

    graph.validate_region_mapping_keys({"R0": 1, "R1": 2}, "dummy_region_map")
    graph.validate_pair_mapping_keys({("R0", "R1"): 1.0}, "dummy_pair_map")
    graph.validate_pair_mapping_keys({(0, 1): 1.0}, "dummy_pair_map_idx")


def run_self_tests(verbose: bool = True) -> None:
    tests = [
        ("basic region graph", _self_test_basic_graph),
        ("site membership", _self_test_site_membership),
        ("mapping validation", _self_test_mapping_validation),
    ]
    for name, fn in tests:
        fn()
        if verbose:
            print(f"[PASS] {name}")
    if verbose:
        print("All regions self-tests passed.")


if __name__ == "__main__":
    run_self_tests(verbose=True)
    from config import make_default_experiment_config

    cfg = make_default_experiment_config()
    graph = RegionGraph(cfg)
    graph.pretty_print()
