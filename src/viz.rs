use polars::prelude::*;

#[inline]
fn cell(x: f64, y: f64, inv_cell: f64) -> (i32, i32) {
    // To find the grid cell of a point, we want
    // gx = floor(u / r)
    // Instead of doing division, we pre-compute 1/r, which is inv_cell
    // floor(x / r) == floor(x * (1/r))
    ((x * inv_cell).floor() as i32, (y * inv_cell).floor() as i32)
}

#[inline]
fn get_bitpacked_bool(ca: &BooleanChunked, i: usize) -> bool {
    // after rechunk(), single chunk is typical; take first chunk
    ca.downcast_iter().next().unwrap().value(i)
}

/// Greedily select a subset of points such that no two selected points are
/// closer than a given minimum distance.
///
/// # Overview
///
/// The algorithm proceeds in three conceptual steps:
///
/// 1. **Define a processing order**
///    Points are processed in a priority order (e.g., input order, a user-
///    provided rank, or with `always_keep` points first). Earlier points have
///    higher priority and suppress later points that are too close.
///
/// 2. **Greedy selection**
///    Points are considered one by one in this order. A point is kept if and
///    only if there is no previously kept point within the specified minimum
///    distance. Once a point is kept, it is never removed.
///
/// 3. **Spatial acceleration using a grid**
///    To avoid comparing each point to all previously kept points, the plane
///    is partitioned into square grid cells whose side length equals the
///    minimum distance. When deciding whether to keep a point, the algorithm
///    only checks for conflicts against points stored in the same grid cell
///    and the eight neighboring cells. Any point closer than the minimum
///    distance must lie in one of these cells.
///
/// # Grid construction
///
/// Each point at coordinates `(u, v)` (after any optional coordinate
/// transformation) is assigned to a grid cell:
///
/// ```text
/// cell_x = floor(u / r)
/// cell_y = floor(v / r)
/// ```
///
/// where `r` is the minimum distance. The grid maps each cell index to the
/// list of points already selected in that cell. Only selected points are
/// stored in the grid.
///
/// # Correctness
///
/// The grid cell size is chosen to equal the minimum distance. With this
/// choice, any two points closer than `r` must lie in either the same grid
/// cell or one of the eight adjacent cells. The algorithm therefore performs
/// an exact distance check against all possible conflicting points while
/// avoiding unnecessary comparisons.
///
/// # Complexity
///
/// Let `n` be the number of input points and `k` the average number of selected
/// points per grid cell. The algorithm runs in approximately `O(n · k)` time
/// and is close to linear for well-distributed data. Memory usage scales with
/// the number of selected points.
fn _thin_points_greedy(
    x: &[f64],
    y: &[f64],
    min_distance: f64,
    always_keep: &BooleanChunked,
    order: Option<&[u64]>, // lower = earlier
) -> Vec<bool> {
    let n = x.len();
    let r2 = min_distance * min_distance;
    let inv_cell = 1.0 / min_distance;

    let idx = match order {
        None => {
            let mut idx = Vec::with_capacity(n);

            for i in 0..n {
                if get_bitpacked_bool(always_keep, i) {
                    idx.push(i);
                }
            }

            for i in 0..n {
                if !get_bitpacked_bool(always_keep, i) {
                    idx.push(i);
                }
            }

            idx
        }
        Some(ord) => {
            let mut idx: Vec<usize> = (0..n).collect();

            idx.sort_unstable_by(|&i, &j| {
                match (
                    get_bitpacked_bool(always_keep, i),
                    get_bitpacked_bool(always_keep, j),
                ) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => ord[i].cmp(&ord[j]).then_with(|| i.cmp(&j)),
                }
            });

            idx
        }
    };

    // Grid: (gx, gy) -> list of kept point indices in that cell
    let mut grid: PlHashMap<(i32, i32), Vec<usize>> = PlHashMap::with_capacity(n / 4 + 1);

    let mut keep = vec![false; n];

    for &i in &idx {
        let (gx, gy) = cell(x[i], y[i], inv_cell);

        // Always-keep: skip checks, keep and insert
        if get_bitpacked_bool(always_keep, i) {
            keep[i] = true;
            grid.entry((gx, gy)).or_default().push(i);
            continue;
        }

        // Check neighbors (3x3 cells)
        let mut too_close = false;
        'neighbors: for dx in -1..=1 {
            for dy in -1..=1 {
                let key = (gx + dx, gy + dy);
                if let Some(cands) = grid.get(&key) {
                    for &j in cands {
                        // only compare to already kept points
                        // (grid only stores kept points, so this is redundant but explicit)
                        if !keep[j] {
                            continue;
                        }
                        let dx = x[i] - x[j];
                        let dy = y[i] - y[j];
                        if dx * dx + dy * dy <= r2 {
                            too_close = true;
                            break 'neighbors;
                        }
                    }
                }
            }
        }

        if !too_close {
            keep[i] = true;
            grid.entry((gx, gy)).or_default().push(i);
        }
    }

    keep
}

pub fn thin_points_greedy(
    df: DataFrame,
    x: &str,
    y: &str,
    min_distance: f64,
    always_keep: &str,
    order: Option<&str>,
) -> Vec<bool> {
    let order_s = order.map(|col| df[col].u64().unwrap().rechunk());
    let order_slice: Option<&[u64]> = order_s.as_ref().map(|s| s.cont_slice().unwrap());

    _thin_points_greedy(
        df[x].f64().unwrap().rechunk().cont_slice().unwrap(),
        df[y].f64().unwrap().rechunk().cont_slice().unwrap(),
        min_distance,
        df[always_keep].bool().unwrap().rechunk().as_ref(),
        order_slice,
    )
}
