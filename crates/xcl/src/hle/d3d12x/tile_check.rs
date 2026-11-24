//! Validation of this port's own tile mapping calls before they reach D3D12.
//!
//! A malformed `ID3D12CommandQueue::UpdateTileMappings` does not return an error: the device
//! is removed with `DXGI_ERROR_INVALID_CALL` (0x887A0001). The documented causes are a
//! resource that was not created with `CreateReservedResource`, region tiles that leave the
//! resource, heap ranges that leave the heap, and region and range tile totals that differ.
//! The checks here read only values the caller already has, so a call that would remove the
//! device is logged and skipped, and a valid call goes out unchanged.
//!
//! `UpdateTileMappings` and `CopyTileMappings` are methods of `ID3D12CommandQueue` with no
//! queue type restriction: DIRECT, COMPUTE and COPY all accept them, so the queue type is
//! not a rejection reason ([`queue_allows_tile_mapping`] records this).

/// The command queue types that can issue a tile mapping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QueueKind {
    Direct,
    Compute,
    Copy,
}

/// Whether `UpdateTileMappings` may be issued on this queue type. All three can.
pub fn queue_allows_tile_mapping(_queue: QueueKind) -> bool {
    true
}

/// The size of one region, as `D3D12_TILE_REGION_SIZE` describes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegionKind {
    /// `UseBox` false: `NumTiles` tiles, walked in x, y, z and then subresource order.
    Tiles(u64),
    /// `UseBox` true: a box of `width * height * depth` tiles.
    Box { width: u64, height: u64, depth: u64 },
}

/// One tile region: its start tile in the resource's overall 64 KiB tile order and its size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Region {
    pub start: u64,
    pub kind: RegionKind,
}

/// A D3D12 tile range flag.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RangeFlag {
    /// Sequential tiles in the heap, starting at `offset`.
    None,
    /// `count` region tiles mapped to NULL; `offset` is ignored.
    Null,
    /// `count` region tiles left as they are; `offset` is ignored.
    Skip,
}

/// One tile range of a mapping call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Range {
    pub flag: RangeFlag,
    pub offset: u64,
    pub count: u64,
}

/// The values a mapping call passes, in the units D3D12 checks them.
pub struct Request<'a> {
    /// The resource was created with `CreateReservedResource`.
    pub reserved: bool,
    /// `pResource` is not NULL.
    pub resource_present: bool,
    /// `D3D12_TILE_MAPPING_FLAG_NULL`: every tile of the resource is unmapped and the
    /// regions and ranges are not read.
    pub null_mapping: bool,
    /// Tiles of the whole resource (`GetResourceTiling`), None when not known.
    pub total_tiles: Option<u64>,
    /// Tiles of the target heap, None when `pHeap` is NULL.
    pub heap_tiles: Option<u64>,
    pub regions: &'a [Region],
    pub ranges: &'a [Range],
}

/// Why a mapping call was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reject {
    /// `pResource` is NULL without `D3D12_TILE_MAPPING_FLAG_NULL`.
    NullResourceWithoutFlag,
    /// The resource exists but was not created with `CreateReservedResource`.
    NotReserved,
    /// A region has no tiles.
    EmptyRegion { region: usize },
    /// A region's tile count, or the sum of the counts, overflows 64 bits.
    Overflow { what: &'static str },
    /// A region starts or ends outside the resource.
    RegionOverflow { region: usize, start: u64, tiles: u64, total: u64 },
    /// The region and range totals differ: D3D12 requires one mapping per region tile.
    CountsMismatch { region_tiles: u64, range_tiles: u64 },
    /// A `NONE` range needs a heap and none was given.
    NoHeap { range: usize },
    /// A `NONE` range starts or ends outside the heap.
    RangeOutOfHeap { range: usize, offset: u64, count: u64, heap: u64 },
}

impl Reject {
    /// The check that failed, for the log line.
    pub fn check(&self) -> &'static str {
        match self {
            Reject::NullResourceWithoutFlag => "null resource without D3D12_TILE_MAPPING_FLAG_NULL",
            Reject::NotReserved => "resource is not reserved",
            Reject::EmptyRegion { .. } => "empty region",
            Reject::Overflow { .. } => "tile count overflow",
            Reject::RegionOverflow { .. } => "region outside the resource",
            Reject::CountsMismatch { .. } => "region and range tile totals differ",
            Reject::NoHeap { .. } => "heap range without a heap",
            Reject::RangeOutOfHeap { .. } => "heap range outside the heap",
        }
    }

    /// The numbers behind the failure, for the log line.
    pub fn detail(&self) -> String {
        match *self {
            Reject::NullResourceWithoutFlag | Reject::NotReserved => String::new(),
            Reject::EmptyRegion { region } => format!("region {region} has 0 tiles"),
            Reject::Overflow { what } => format!("{what} overflows"),
            Reject::RegionOverflow { region, start, tiles, total } => {
                format!("region {region}: start {start} + {tiles} tiles, resource has {total}")
            }
            Reject::CountsMismatch { region_tiles, range_tiles } => {
                format!("{region_tiles} region tiles, {range_tiles} range tiles")
            }
            Reject::NoHeap { range } => format!("range {range}"),
            Reject::RangeOutOfHeap { range, offset, count, heap } => {
                format!("range {range}: offset {offset} + {count} tiles, heap has {heap}")
            }
        }
    }

    /// Both parts on one line.
    pub fn summary(&self) -> String {
        let detail = self.detail();
        if detail.is_empty() {
            self.check().to_string()
        } else {
            format!("{}: {detail}", self.check())
        }
    }
}

/// Whether rejection number `n` (1-based) is logged: the first 16 and every power of two
/// after them, so a repeating failure cannot flood the log.
pub fn should_log(n: u64) -> bool {
    n <= 16 || n.is_power_of_two()
}

/// Count a rejected call in `rejected` and, when [`should_log`] says so, log it.
pub fn log_reject(site: &str, rejected: &std::sync::atomic::AtomicU64, r: &Reject) {
    let n = rejected.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if should_log(n) {
        crate::xlog_warn!("D3D12.X: {site}: skipped an invalid UpdateTileMappings ({n} rejected): {}", r.summary());
    }
}

fn sum(what: &'static str, a: u64, b: u64) -> Result<u64, Reject> {
    a.checked_add(b).ok_or(Reject::Overflow { what })
}

/// Check a mapping call against the D3D12 rules. `Ok` means the call is valid as passed.
pub fn check(r: &Request) -> Result<(), Reject> {
    if r.null_mapping {
        // every tile of the resource is unmapped; regions and ranges are not read
        return Ok(());
    }
    if !r.resource_present {
        return Err(Reject::NullResourceWithoutFlag);
    }
    if !r.reserved {
        return Err(Reject::NotReserved);
    }
    let mut region_tiles = 0u64;
    for (i, region) in r.regions.iter().enumerate() {
        let tiles = match region.kind {
            RegionKind::Tiles(n) => n,
            RegionKind::Box { width, height, depth } => width
                .checked_mul(height)
                .and_then(|v| v.checked_mul(depth))
                .ok_or(Reject::Overflow { what: "a box region" })?,
        };
        if tiles == 0 {
            return Err(Reject::EmptyRegion { region: i });
        }
        region_tiles = sum("the region tile total", region_tiles, tiles)?;
        if let Some(total) = r.total_tiles {
            let end = sum("a region's end", region.start, tiles)?;
            if end > total {
                return Err(Reject::RegionOverflow { region: i, start: region.start, tiles, total });
            }
        }
    }
    let mut range_tiles = 0u64;
    for (i, range) in r.ranges.iter().enumerate() {
        range_tiles = sum("the range tile total", range_tiles, range.count)?;
        match range.flag {
            RangeFlag::None => {
                let Some(heap) = r.heap_tiles else {
                    return Err(Reject::NoHeap { range: i });
                };
                let end = sum("a heap range's end", range.offset, range.count)?;
                if range.count == 0 || end > heap {
                    return Err(Reject::RangeOutOfHeap { range: i, offset: range.offset, count: range.count, heap });
                }
            }
            // the offset is ignored for these, and they consume no heap tiles
            RangeFlag::Null | RangeFlag::Skip => {}
        }
    }
    if region_tiles != range_tiles {
        return Err(Reject::CountsMismatch { region_tiles, range_tiles });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiles(start: u64, n: u64) -> Region {
        Region { start, kind: RegionKind::Tiles(n) }
    }

    fn mapped(offset: u64, count: u64) -> Range {
        Range { flag: RangeFlag::None, offset, count }
    }

    fn req<'a>(total: u64, heap: u64, regions: &'a [Region], ranges: &'a [Range]) -> Request<'a> {
        Request { reserved: true, resource_present: true, null_mapping: false, total_tiles: Some(total), heap_tiles: Some(heap), regions, ranges }
    }

    #[test]
    fn a_valid_whole_resource_mapping_passes() {
        let regions = [tiles(0, 100)];
        let ranges = [mapped(0, 100)];
        assert_eq!(check(&req(100, 1024, &regions, &ranges)), Ok(()));
    }

    #[test]
    fn several_regions_and_ranges_passes_when_the_totals_match() {
        // two windows, one with a gap skipped for tiles of another heap
        let regions = [tiles(5, 3), tiles(20, 4)];
        let ranges = [
            mapped(0, 2),
            Range { flag: RangeFlag::Skip, offset: 0, count: 1 },
            mapped(2, 3),
            Range { flag: RangeFlag::Null, offset: 0, count: 1 },
        ];
        assert_eq!(check(&req(64, 1024, &regions, &ranges)), Ok(()));
    }

    #[test]
    fn a_region_past_the_resource_is_rejected() {
        let regions = [tiles(60, 5)];
        let ranges = [mapped(0, 5)];
        assert_eq!(
            check(&req(64, 1024, &regions, &ranges)),
            Err(Reject::RegionOverflow { region: 0, start: 60, tiles: 5, total: 64 })
        );
    }

    #[test]
    fn a_region_that_ends_exactly_at_the_resource_passes() {
        let regions = [tiles(60, 4)];
        let ranges = [mapped(0, 4)];
        assert_eq!(check(&req(64, 1024, &regions, &ranges)), Ok(()));
    }

    #[test]
    fn a_heap_range_past_the_heap_is_rejected() {
        let regions = [tiles(0, 10)];
        let ranges = [mapped(1020, 10)];
        assert_eq!(
            check(&req(64, 1024, &regions, &ranges)),
            Err(Reject::RangeOutOfHeap { range: 0, offset: 1020, count: 10, heap: 1024 })
        );
    }

    #[test]
    fn a_null_range_without_a_heap_passes() {
        let regions = [tiles(0, 8)];
        let ranges = [Range { flag: RangeFlag::Null, offset: 0, count: 8 }];
        let r = Request { heap_tiles: None, ..req(64, 0, &regions, &ranges) };
        assert_eq!(check(&r), Ok(()));
    }

    #[test]
    fn a_mapped_range_without_a_heap_is_rejected() {
        let regions = [tiles(0, 8)];
        let ranges = [mapped(0, 8)];
        let r = Request { heap_tiles: None, ..req(64, 0, &regions, &ranges) };
        assert_eq!(check(&r), Err(Reject::NoHeap { range: 0 }));
    }

    #[test]
    fn null_and_skip_offsets_are_ignored() {
        let regions = [tiles(0, 2)];
        let ranges = [Range { flag: RangeFlag::Skip, offset: u64::MAX, count: 1 }, Range { flag: RangeFlag::Null, offset: u64::MAX, count: 1 }];
        assert_eq!(check(&req(2, 1, &regions, &ranges)), Ok(()));
    }

    #[test]
    fn a_tile_total_mismatch_is_rejected() {
        let regions = [tiles(0, 4)];
        let ranges = [mapped(0, 3)];
        assert_eq!(check(&req(64, 1024, &regions, &ranges)), Err(Reject::CountsMismatch { region_tiles: 4, range_tiles: 3 }));
    }

    #[test]
    fn an_empty_region_is_rejected() {
        let regions = [tiles(0, 0)];
        let ranges = [mapped(0, 0)];
        assert_eq!(check(&req(64, 1024, &regions, &ranges)), Err(Reject::EmptyRegion { region: 0 }));
    }

    #[test]
    fn a_box_region_is_checked_by_its_box() {
        let regions = [Region { start: 0, kind: RegionKind::Box { width: 4, height: 2, depth: 1 } }];
        let ranges = [mapped(0, 8)];
        assert_eq!(check(&req(8, 1024, &regions, &ranges)), Ok(()));
        let big = [Region { start: 0, kind: RegionKind::Box { width: 8, height: 2, depth: 1 } }];
        assert_eq!(check(&req(8, 1024, &big, &ranges)), Err(Reject::RegionOverflow { region: 0, start: 0, tiles: 16, total: 8 }));
    }

    #[test]
    fn a_box_product_that_overflows_is_rejected() {
        let regions = [Region { start: 0, kind: RegionKind::Box { width: u64::MAX, height: 2, depth: 1 } }];
        let ranges = [mapped(0, 1)];
        assert_eq!(check(&req(64, 1024, &regions, &ranges)), Err(Reject::Overflow { what: "a box region" }));
    }

    #[test]
    fn a_start_plus_tiles_that_overflows_is_rejected() {
        let regions = [tiles(u64::MAX, 2)];
        let ranges = [mapped(0, 2)];
        assert_eq!(check(&req(64, 1024, &regions, &ranges)), Err(Reject::Overflow { what: "a region's end" }));
    }

    #[test]
    fn a_resource_that_is_not_reserved_is_rejected() {
        let regions = [tiles(0, 1)];
        let ranges = [mapped(0, 1)];
        let r = Request { reserved: false, ..req(1, 1, &regions, &ranges) };
        assert_eq!(check(&r), Err(Reject::NotReserved));
    }

    #[test]
    fn a_null_resource_needs_the_null_flag() {
        let regions = [tiles(0, 1)];
        let ranges = [mapped(0, 1)];
        let r = Request { resource_present: false, ..req(1, 1, &regions, &ranges) };
        assert_eq!(check(&r), Err(Reject::NullResourceWithoutFlag));
        let unmapped = Request { resource_present: false, null_mapping: true, ..req(1, 1, &regions, &ranges) };
        assert_eq!(check(&unmapped), Ok(()));
    }

    #[test]
    fn an_unknown_total_tiles_skips_the_region_bound() {
        let regions = [tiles(1 << 40, 4)];
        let ranges = [mapped(0, 4)];
        let r = Request { total_tiles: None, ..req(64, 1024, &regions, &ranges) };
        assert_eq!(check(&r), Ok(()));
    }

    #[test]
    fn every_queue_type_allows_tile_mapping() {
        for q in [QueueKind::Direct, QueueKind::Compute, QueueKind::Copy] {
            assert!(queue_allows_tile_mapping(q));
        }
    }

    #[test]
    fn the_first_sixteen_and_powers_of_two_are_logged() {
        let logged: Vec<u64> = (1..=64).filter(|&n| should_log(n)).collect();
        assert_eq!(logged, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64]);
    }

    #[test]
    fn a_rejection_names_its_check_and_numbers() {
        let r = Reject::RegionOverflow { region: 2, start: 60, tiles: 5, total: 64 };
        assert_eq!(r.summary(), "region outside the resource: region 2: start 60 + 5 tiles, resource has 64");
        assert_eq!(Reject::NotReserved.summary(), "resource is not reserved");
    }
}
